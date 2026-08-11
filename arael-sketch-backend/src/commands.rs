// Command system: parse and execute text commands for the sketch.
// Decoupled from GUI -- operates on CommandContext which holds sketch state.

use arael::simple_lm::RootProblem;
use std::collections::HashMap;
use arael::model::JacobianModel;
use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::{Action, Created};
use crate::geometry::{arc_start_pos, arc_end_pos};
use crate::history::{History, CursorState};
use crate::ids::Selection;

/// Weight applied to the drag-pull pseudo-constraint when the user drags
/// a point; kept here with the command/action layer that sets it so the
/// GUI can import the same value.
pub const DRAG_PULL_WEIGHT: f64 = 1.0;

// ---------------------------------------------------------------------------
// CommandContext: GUI-free state for command execution
// ---------------------------------------------------------------------------

pub struct CommandContext {
    pub sketch: SketchCell,
    pub history: History,
    pub selection: Vec<Selection>,
    pub session_vars: HashMap<String, f64>,
    pub session_vecs: HashMap<String, vect2d>,
    pub session_names: HashMap<String, String>, // variable -> entity name aliases
    pub cursor: Option<vect2d>,
    pub cursor_tangent: Option<vect2d>,
    pub saved_cursor: CursorState,
    pub status_error: Option<String>,
    /// When a constraint was rejected with a blocker analysis, the
    /// user-facing names (`C<n>`, `d<n>`, "CL0H") of the conflicting
    /// existing constraints. Consumed by the GUI to flash those
    /// constraints briefly. Cleared on the next successful action.
    pub status_blocker_names: Option<Vec<String>>,
    pub last_cost: f64,
    pub skip_dof_check: bool,
    // View state (used by center/zoom commands; GUI overrides with real values)
    pub scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub pending_fit: bool,
    /// Commands blocked in this context (e.g. "save", "load" for MCP).
    pub blocked_commands: Vec<&'static str>,
    /// Set by the `exit` command to signal the app should close.
    pub exit_requested: bool,
    /// Raw-drag mode (see --drag-raw). When true, `cmd_drag` hard-pins
    /// the dragged endpoint to the target via a fixed helper; the sketch
    /// deforms when the target is infeasible. When false (default), the
    /// helper is optimizable so hard constraints stay satisfied and the
    /// dragged endpoint lands at the nearest feasible point.
    pub drag_raw: bool,
    /// Stream each command's echo + output to stdout as it executes,
    /// flushing line by line. Lets `eprintln!` LM verbose traces and
    /// `println!` command echo interleave chronologically when piped.
    pub echo_stdout: bool,
}

#[allow(dead_code)]
impl CommandContext {
    pub fn new() -> Self {
        let sketch = Sketch::new();
        let history = History::new(&sketch);
        CommandContext {
            sketch: sketch.into(), history,
            selection: Vec::new(),
            session_vars: HashMap::new(),
            session_vecs: HashMap::new(),
            session_names: HashMap::new(),
            cursor: None,
            cursor_tangent: None,
            saved_cursor: CursorState::default(),
            status_error: None,
            status_blocker_names: None,
            last_cost: 0.0,
            skip_dof_check: false,
            scale: 80.0,
            offset_x: 400.0,
            offset_y: 300.0,
            pending_fit: false,
            blocked_commands: Vec::new(),
            exit_requested: false,
            drag_raw: false,
            echo_stdout: false,
        }
    }

    pub fn with_sketch(sketch: Sketch) -> Self {
        let history = History::new(&sketch);
        CommandContext {
            sketch: sketch.into(), history,
            selection: Vec::new(),
            session_vars: HashMap::new(),
            session_vecs: HashMap::new(),
            session_names: HashMap::new(),
            cursor: None,
            cursor_tangent: None,
            saved_cursor: CursorState::default(),
            status_error: None,
            status_blocker_names: None,
            last_cost: 0.0,
            skip_dof_check: false,
            scale: 80.0,
            offset_x: 400.0,
            offset_y: 300.0,
            pending_fit: false,
            blocked_commands: Vec::new(),
            exit_requested: false,
            drag_raw: false,
            echo_stdout: false,
        }
    }

    /// Set skip_dof_check for the duration of a constraint command.
    fn set_force(&mut self, force: bool) {
        self.skip_dof_check = force;
    }

    pub fn begin_group(&mut self) {
        self.saved_cursor = CursorState { pos: self.cursor, tangent: self.cursor_tangent };
        self.history.begin_group();
    }
}

/// Build a hint string for dimension rejection errors showing the current measured value.
fn dimension_rejection_hint(sketch: &Sketch, action: &Action) -> String {
    let (kind, requested) = match action {
        Action::AddDimension { kind, value, .. } => (Some(kind), Some(*value)),
        Action::UpdateDimension { did, value, .. } => {
            if let Some(dim) = sketch.dimension_index_by_did(*did)
                .and_then(|i| sketch.dimensions.get(i)) {
                (Some(&dim.kind), Some(*value))
            } else { (None, None) }
        }
        _ => (None, None),
    };
    let (Some(kind), Some(requested)) = (kind, requested) else { return String::new() };
    let current = match kind {
        DimensionKind::LineLength(r) => {
            let l = &sketch.lines[*r];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            Some(("length", (dx * dx + dy * dy).sqrt()))
        }
        DimensionKind::ArcRadius(r) => Some(("radius", sketch.arcs[*r].radius.value)),
        DimensionKind::ArcRadiusB(r) => Some(("radius_b", sketch.arcs[*r].radius_b.value)),
        DimensionKind::ArcSweep(r) => {
            let a = &sketch.arcs[*r];
            Some(("sweep", arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs())))
        }
        DimensionKind::ArcRotation(r) => {
            Some(("xangle", arael::utils::rad2deg(sketch.arcs[*r].rotation.value)))
        }
        DimensionKind::Angle(a, b, supplement) => {
            let la = &sketch.lines[*a];
            let lb = &sketch.lines[*b];
            let dx1 = la.p2.value.x - la.p1.value.x;
            let dy1 = la.p2.value.y - la.p1.value.y;
            let dx2 = lb.p2.value.x - lb.p1.value.x;
            let dy2 = lb.p2.value.y - lb.p1.value.y;
            let cross = dx1 * dy2 - dy1 * dx2;
            let dot = dx1 * dx2 + dy1 * dy2;
            let angle_rad = cross.atan2(dot).abs();
            let angle_deg = if *supplement { 180.0 - arael::utils::rad2deg(angle_rad) } else { arael::utils::rad2deg(angle_rad) };
            Some(("angle", angle_deg))
        }
        DimensionKind::PointPointDistance(a, b) => {
            let pa = dim_endpoint_pos_from_sketch(sketch, a);
            let pb = dim_endpoint_pos_from_sketch(sketch, b);
            let dx = pb.x - pa.x;
            let dy = pb.y - pa.y;
            Some(("distance", (dx * dx + dy * dy).sqrt()))
        }
        DimensionKind::PointLineDistance(pt, line) => {
            let p = dim_endpoint_pos_from_sketch(sketch, pt);
            let l = &sketch.lines[*line];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { None } else {
                let dist = ((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx).abs() / len;
                Some(("distance", dist))
            }
        }
        DimensionKind::HDistance(a, b) => {
            let pa = dim_endpoint_pos_from_sketch(sketch, a);
            let pb = dim_endpoint_pos_from_sketch(sketch, b);
            Some(("hdistance", (pa.x - pb.x).abs()))
        }
        DimensionKind::VDistance(a, b) => {
            let pa = dim_endpoint_pos_from_sketch(sketch, a);
            let pb = dim_endpoint_pos_from_sketch(sketch, b);
            Some(("vdistance", (pa.y - pb.y).abs()))
        }
        DimensionKind::LineAngle(r) => {
            let l = &sketch.lines[*r];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            Some(("xangle", arael::utils::rad2deg(dy.atan2(dx))))
        }
        DimensionKind::ConcentricDistance(a, b) => {
            let ra = sketch.arcs[*a].radius.value;
            let rb = sketch.arcs[*b].radius.value;
            Some(("distance", (rb - ra).abs()))
        }
        DimensionKind::LineLineDistance(a, b) => {
            let la = &sketch.lines[*a];
            let lb = &sketch.lines[*b];
            let dx = la.p2.value.x - la.p1.value.x;
            let dy = la.p2.value.y - la.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { None } else {
                let dist = ((lb.p1.value.x - la.p1.value.x) * dy
                          - (lb.p1.value.y - la.p1.value.y) * dx).abs() / len;
                Some(("distance", dist))
            }
        }
    };
    if let Some((label, current_val)) = current {
        format!(". Current {} is {:.4}, requested {:.4}", label, current_val, requested)
    } else {
        String::new()
    }
}

/// Get position of a dimension endpoint from sketch (without EditorApp).
fn dim_endpoint_pos_from_sketch(sketch: &Sketch, ep: &DimensionEndpoint) -> vect2d {
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

/// Rejection info returned by `validate_and_apply_constraint` when a
/// constraint is refused. Carries the human-readable message plus,
/// for DOF-rejection, the user-facing names of the existing
/// constraints the blocker analysis identified as conflicting
/// (`C<n>` / `d<n>` / "CL0H") so the GUI can flash them via the
/// existing `find_constraint_by_name` / dimension-name resolvers.
pub struct Rejection {
    pub message: String,
    pub blocker_names: Vec<String>,
}

impl Rejection {
    fn msg(message: impl Into<String>) -> Self {
        Rejection { message: message.into(), blocker_names: Vec::new() }
    }
}

impl From<String> for Rejection {
    fn from(s: String) -> Self { Rejection::msg(s) }
}

/// Validate and apply a constraint action on a sketch.
/// Returns Ok(new_cost) on success, Err(Rejection) on rejection.
/// Handles snapshot/restore, cost checking, and DOF checking.
/// Drop selection entries whose referents no longer exist: stale arena
/// refs and out-of-range indices, left behind by deletes and sketch
/// replacement. Constraint validity reuses constraint_id_name's
/// collection mapping; the flag and helper variants hold raw refs and
/// are checked against their arenas directly.
fn prune_selection(ctx: &mut CommandContext) {
    use crate::ids::ConstraintId;
    let sketch = &ctx.sketch;
    ctx.selection.retain(|s| match *s {
        Selection::Point(r) => sketch.points.get(r).is_some(),
        Selection::Line(r)
        | Selection::LineP1(r)
        | Selection::LineP2(r) => sketch.lines.get(r).is_some(),
        Selection::Arc(r)
        | Selection::ArcCenter(r)
        | Selection::ArcStart(r)
        | Selection::ArcEnd(r) => sketch.arcs.get(r).is_some(),
        Selection::Dimension(did) => sketch.dimension_index_by_did(did).is_some(),
        Selection::Constraint(id) => match id {
            ConstraintId::Horizontal(r) | ConstraintId::Vertical(r) => {
                sketch.lines.get(r).is_some()
            }
            ConstraintId::HelperBridge(p) => sketch.points.get(p).is_some(),
            other => crate::ids::constraint_id_name(sketch, other).is_some(),
        },
    });
}

pub fn validate_and_apply_constraint(
    sketch: &mut Sketch,
    action: &Action,
    skip_dof_check: bool,
) -> Result<f64, Rejection> {
    use arael::simple_lm::LmProblem;

    let snapshot = bincode::serialize(sketch).ok();
    let old_cost = {
        let mut params = Vec::new();
        sketch.serialize(&mut params);
        sketch.calc_cost(&params)
    };

    // Skip DOF check for internal/non-constraining actions
    let should_check_dof = !skip_dof_check && match action {
        Action::UpdateDimension { .. } => false,
        Action::AddDimension { derived: true, .. } => false,
        // Range dimensions: penalty residual, doesn't remove DOF from the
        // linear-algebra point of view.
        Action::AddDimension { range: Some(_), .. } => false,
        Action::ApplyCoincidentPP { a, .. } => !sketch.points.get(*a).is_some_and(|p| p.helper),
        Action::ApplyCoincidentLP1 { point, .. } | Action::ApplyCoincidentLP2 { point, .. } =>
            !sketch.points.get(*point).is_some_and(|p| p.helper),
        Action::ApplyCoincidentArcCenter { point, .. } | Action::ApplyCoincidentArcStart { point, .. } |
        Action::ApplyCoincidentArcEnd { point, .. } =>
            !sketch.points.get(*point).is_some_and(|p| p.helper),
        _ => true,
    };

    let old_dof = if should_check_dof {
        Some(sketch.dof()?)
    } else {
        None
    };

    action.apply(sketch);
    sketch.dedup_constraints();

    // Quick cost check
    let quick_cost = {
        let mut params = Vec::new();
        sketch.serialize(&mut params);
        sketch.calc_cost(&params)
    };
    let new_cost = if quick_cost <= old_cost + 1e-6 {
        quick_cost
    } else {
        sketch.solve().end_cost
    };

    // Cost rejection
    if new_cost > old_cost + 1e-3
        && let Some(ref snap) = snapshot
            && let Ok(restored) = bincode::deserialize::<Sketch>(snap) {
                *sketch = restored;
                let hint = dimension_rejection_hint(sketch, action);
                return Err(Rejection::msg(format!(
                    "Constraint rejected: could not satisfy all constraints{}",
                    hint)));
            }

    // Negative radius rejection
    for r in sketch.arcs.refs() {
        let a = &sketch.arcs[r];
        let bad = if a.radius.value < 0.0 {
            Some(("radius", a.radius.value))
        } else if a.is_ellipse && a.radius_b.value < 0.0 {
            Some(("radius_b", a.radius_b.value))
        } else {
            None
        };
        if let Some((which, val)) = bad {
            let name = a.name.clone();
            if let Some(ref snap) = snapshot
                && let Ok(restored) = bincode::deserialize::<Sketch>(snap) {
                    *sketch = restored;
                    return Err(Rejection::msg(format!(
                        "Constraint rejected: {} got negative {} ({:.4}). This is likely a solver bug -- please report it.",
                        name, which, val)));
                }
        }
    }

    // DOF rejection
    if let Some(old_dof) = old_dof {
        let new_dof = sketch.dof()?;
        if new_dof >= old_dof
            && let Some(ref snap) = snapshot {
                let (blocker_hint, blocker_names) = blocker_hint_for_rejection(sketch, snap);
                if let Ok(restored) = bincode::deserialize::<Sketch>(snap) {
                    *sketch = restored;
                    return Err(Rejection {
                        message: format!(
                            "Constraint rejected: DOF unchanged at {}. {}Use 'force' to override.",
                            new_dof, blocker_hint),
                        blocker_names,
                    });
                }
            }
    }

    Ok(new_cost)
}

/// Run blocker analysis on a post-apply sketch. Returns the
/// human-readable hint (embedded in the rejection message) and the
/// user-facing constraint names (`"C<n>"` / `"d<n>"` / `"CL0H"`) of
/// the conflicting existing constraints so the GUI can flash them.
/// Returns `(String::new(), Vec::new())` if no blocker can be
/// identified (empty sketch, numerical edge case, cutoff reached).
fn blocker_hint_for_rejection(sketch: &mut Sketch, pre_snap: &[u8]) -> (String, Vec<String>) {
    // Pre-apply identifiers. Nids are serialised so they survive
    // deserialize without re-running calc_jacobian. Expression
    // constraints don't have nids, so they're identified by
    // description (also stable across rebuilds).
    let mut pre: Sketch = match bincode::deserialize::<Sketch>(pre_snap) {
        Ok(s) => s,
        Err(_) => return (String::new(), Vec::new()),
    };
    pre.prepare_expr_constraints();
    let pre_nids: std::collections::HashSet<u32> = pre.constraint_nid_cid_pairs()
        .into_iter().map(|(nid, _)| nid).collect();
    let pre_expr_descs: std::collections::HashSet<String> = pre.expr_constraints
        .iter().map(|ec| ec.description.clone()).collect();

    // Post-apply jacobian, with drift suppressed so the weak
    // regularizer doesn't dominate rowspan. Uses current params so
    // the rowspan analysis matches what DOF-rejection just saw --
    // at a tangent-aligned config the blocker hint then correctly
    // names the rows whose instantaneous linear dependence caused
    // the rejection.
    let saved_drift = sketch.drift_isigma;
    sketch.drift_isigma = 0.0;
    let mut post_params = Vec::new();
    sketch.serialize(&mut post_params);
    let post_jac = sketch.calc_jacobian(&post_params);
    sketch.drift_isigma = saved_drift;

    // Candidate cids: collection constraints whose nid is new, and
    // expression constraints whose description is new.
    let mut candidate_cids: std::collections::HashSet<u32> = sketch.constraint_nid_cid_pairs()
        .into_iter()
        .filter(|(nid, _)| !pre_nids.contains(nid))
        .map(|(_, cid)| cid)
        .collect();
    for ec in &sketch.expr_constraints {
        if !pre_expr_descs.contains(&ec.description) {
            candidate_cids.insert(ec.cid);
        }
    }
    // Entity-flag candidates. Flag actions (`horizontal`, `vertical`,
    // `length`, `lock`, `radius`, `sweep`, `angle`) just flip a bool
    // on the host entity rather than pushing into a constraint
    // collection, so they don't show up in the nid or expr diffs.
    // Compare pre vs post entity flags directly; any flag flipped
    // from false -> true adds the entity's cid as a candidate. Rows
    // on that entity share the entity's cid, and at this point the
    // only non-zero row there is the one the flag just enabled
    // (drift rows are zero under drift_isigma=0, earlier active
    // flags are already in both pre and post so they cancel out of
    // the blocker analysis -- a pre-existing flag is NOT a blocker
    // candidate, only the just-added one).
    for r in sketch.lines.refs() {
        let post = &sketch.lines[r];
        let Some(pre_line) = pre.lines.get(r) else { continue };
        let flag_added =
            (post.constraints.horizontal && !pre_line.constraints.horizontal)
            || (post.constraints.vertical && !pre_line.constraints.vertical)
            || (post.constraints.has_length && !pre_line.constraints.has_length)
            || (post.constraints.has_angle && !pre_line.constraints.has_angle)
            || (!post.p1.optimize && pre_line.p1.optimize)
            || (!post.p2.optimize && pre_line.p2.optimize);
        if flag_added {
            candidate_cids.insert(post.cid);
        }
    }
    for r in sketch.points.refs() {
        let post = &sketch.points[r];
        let Some(pre_pt) = pre.points.get(r) else { continue };
        let flag_added =
            (post.constraints.has_fix_x && !pre_pt.constraints.has_fix_x)
            || (post.constraints.has_fix_y && !pre_pt.constraints.has_fix_y)
            || (!post.pos.optimize && pre_pt.pos.optimize);
        if flag_added {
            candidate_cids.insert(post.cid);
        }
    }
    for r in sketch.arcs.refs() {
        let post = &sketch.arcs[r];
        let Some(pre_arc) = pre.arcs.get(r) else { continue };
        let flag_added =
            (post.constraints.has_target_radius && !pre_arc.constraints.has_target_radius)
            || (post.constraints.has_target_sweep && !pre_arc.constraints.has_target_sweep)
            || (!post.center.optimize && pre_arc.center.optimize);
        if flag_added {
            candidate_cids.insert(post.cid);
        }
    }
    if candidate_cids.is_empty() {
        return (String::new(), Vec::new());
    }

    let report = match analyze_blockers(&post_jac, &candidate_cids) {
        Some(r) => r,
        None => return (String::new(), Vec::new()),
    };
    if arael_sketch_solver::verbose() {
        let s = &report.stats;
        eprintln!("[BLOCKER] candidates={} existing_constraints {}->{} (prune {:.3}ms) existing_rows={} rej_check={:.3}ms total={:.3}ms",
            s.candidate_rows, s.existing_before_prune, s.existing_after_prune,
            s.component_prune_ms, s.existing_rows, s.rejection_check_ms, s.total_ms);
        for kstat in &s.per_k {
            let tag = if kstat.skipped { "SKIPPED (pool > cutoff)" } else { "" };
            eprintln!("[BLOCKER]   k={} tested={} blockers={} time={:.3}ms {}",
                kstat.k, kstat.subsets_tested, kstat.blockers_found, kstat.time_ms, tag);
        }
        eprintln!("[BLOCKER] result: min_size={} sets={} truncated={} existing_redundant={}",
            report.minimum_size, report.sets.len(), report.truncated, report.existing_redundant);
    }
    let label_map = build_cid_display_map(sketch, &post_jac);
    let name_map = build_cid_name_map(sketch, &post_jac);
    let hint = format_blocker_report(&label_map, &report);
    // Flatten all distinct blocker names across all minimum-size sets
    // into a single list (dedup while preserving first-seen order).
    let mut names: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for set in &report.sets {
        for cid in set {
            if let Some(n) = name_map.get(cid)
                && seen.insert(n.clone()) {
                names.push(n.clone());
            }
        }
    }
    (hint, names)
}

/// Same as `build_cid_display_map` but returns only the bare name
/// ("C<n>" / "d<n>" / "CL0H") without the descriptive parenthesis.
/// Used to surface blocker identifiers to the GUI for highlighting.
fn build_cid_name_map(
    sketch: &Sketch,
    jac: &arael::model::Jacobian<f64>,
) -> HashMap<u32, String> {
    let base = sketch.constraint_labels();
    let dim_names = sketch.dimension_cid_name_map();
    let mut out: HashMap<u32, String> = HashMap::new();
    for (nid, cid) in sketch.constraint_nid_cid_pairs() {
        let name = dim_names.get(&cid).cloned().unwrap_or_else(|| format!("C{}", nid));
        out.insert(cid, name);
    }
    // Expression-backed dimension constraints don't appear in
    // constraint_nid_cid_pairs (they're not in a named collection);
    // pick them up directly from dim_names so their d<N> handle
    // reaches the GUI flash path.
    for (&cid, name) in &dim_names {
        out.entry(cid).or_insert_with(|| name.clone());
    }
    let mut per_cid_labels: HashMap<u32, Vec<&'static str>> = HashMap::new();
    for row in &jac.rows {
        let v = per_cid_labels.entry(row.constraint).or_default();
        if !v.contains(&row.label) { v.push(row.label); }
    }
    for (cid, row_labels) in per_cid_labels {
        if out.contains_key(&cid) { continue; }
        let Some(base_name) = base.get(&cid).cloned() else { continue; };
        let entity_name = base_name
            .strip_prefix("point:")
            .or_else(|| base_name.strip_prefix("line:"))
            .or_else(|| base_name.strip_prefix("arc:"))
            .map(|s| s.to_string())
            .unwrap_or(base_name);
        let attrs: Vec<&str> = row_labels.iter()
            .copied()
            .filter(|l| *l != "drift" && *l != "drag_pull")
            .collect();
        let flag_char = match attrs.as_slice() {
            ["horizontal"] => Some('H'),
            ["vertical"] => Some('V'),
            _ => None,
        };
        if let Some(ch) = flag_char {
            out.insert(cid, arael_sketch_solver::format_flag_name(&entity_name, ch));
        } else {
            // No stable name -- skip rather than expose an internal cid.
            // The hint string already names the entity; highlighting
            // falls back to entity-level (future work).
        }
    }
    out
}

/// Build a CID -> user-visible label map. Each label is in the form
/// `<name> (<description>)` where `<name>` is the deletable handle
/// (`C{nid}` or `CL0H` / `CL0V` for synthetic flag constraints) and
/// `<description>` is the same descriptive text the user sees in
/// `list` / `info <name>`. For entity cids that carry attributes
/// without a stable name (`fix_x`, `has_length`, etc.), fall back to
/// `<entity> (<attrs>)` rendered from the jacobian row labels.
fn build_cid_display_map(
    sketch: &Sketch,
    jac: &arael::model::Jacobian<f64>,
) -> HashMap<u32, String> {
    let base = sketch.constraint_labels();
    let dim_names = sketch.dimension_cid_name_map();
    let mut out: HashMap<u32, String> = HashMap::new();
    // Collection-backed constraints: prefer the dimension name
    // (`d<N>`) when the constraint is dimension-backed, since
    // `delete d<N>` is the actionable handle; fall back to `C<nid>`
    // otherwise. In both cases append the descriptive text from
    // `find_constraint_description` (same text as `info` output).
    for (nid, cid) in sketch.constraint_nid_cid_pairs() {
        let name = match dim_names.get(&cid) {
            Some(dname) => dname.clone(),
            None => format!("C{}", nid),
        };
        let desc = sketch.find_constraint_description(&format!("C{}", nid));
        let display = match desc {
            Some(d) => format!("{} ({})", name, d),
            None => name,
        };
        out.insert(cid, display);
    }
    // Expression-backed dimension constraints. The backing lives in
    // `sketch.expr_constraints`, not a named collection, so
    // `constraint_nid_cid_pairs` skips them. Use `dim_names` to get
    // the actionable `d<N>` handle; take the description from the
    // expr_constraint, stripping its leading "d<N> " since it would
    // duplicate the handle.
    for ec in &sketch.expr_constraints {
        let Some(dim_name) = dim_names.get(&ec.cid) else { continue };
        let remainder = ec.description
            .strip_prefix(&format!("{} ", dim_name))
            .unwrap_or(&ec.description);
        let display = format!("{} ({})", dim_name, remainder);
        out.insert(ec.cid, display);
    }
    // Entity-attached attributes. Collect the row labels active on
    // each unnamed entity cid; emit `<name> (<attr>)` for flags that
    // have a stable C{entity}{flag} name, otherwise fall back to
    // `<entity> (<attrs>)`. Drop drift/drag_pull.
    let mut per_cid_labels: HashMap<u32, Vec<&'static str>> = HashMap::new();
    for row in &jac.rows {
        let v = per_cid_labels.entry(row.constraint).or_default();
        if !v.contains(&row.label) { v.push(row.label); }
    }
    for (cid, row_labels) in per_cid_labels {
        if out.contains_key(&cid) { continue; }
        let Some(base_name) = base.get(&cid).cloned() else { continue; };
        let entity_name = base_name
            .strip_prefix("point:")
            .or_else(|| base_name.strip_prefix("line:"))
            .or_else(|| base_name.strip_prefix("arc:"))
            .map(|s| s.to_string())
            .unwrap_or(base_name);
        let attrs: Vec<&str> = row_labels.iter()
            .copied()
            .filter(|l| *l != "drift" && *l != "drag_pull")
            .collect();
        if attrs.is_empty() {
            out.insert(cid, entity_name);
            continue;
        }
        // If there's a single attribute with a known flag name
        // (horizontal/vertical), use the C<entity>{flag} form so
        // `info` resolves it directly.
        let flag_char = match attrs.as_slice() {
            ["horizontal"] => Some('H'),
            ["vertical"] => Some('V'),
            _ => None,
        };
        let display = match flag_char {
            Some(ch) => {
                let flag_name = arael_sketch_solver::format_flag_name(&entity_name, ch);
                match sketch.find_constraint_description(&flag_name) {
                    Some(desc) => format!("{} ({})", flag_name, desc),
                    None => format!("{} ({})", entity_name, attrs.join(",")),
                }
            }
            None => format!("{} ({})", entity_name, attrs.join(",")),
        };
        out.insert(cid, display);
    }
    out
}

fn format_blocker_report(
    labels: &HashMap<u32, String>,
    report: &BlockerReport,
) -> String {
    let name = |cid: u32| -> String {
        labels.get(&cid).cloned().unwrap_or_else(|| format!("C<cid={}>", cid))
    };
    if report.sets.is_empty() {
        if report.truncated {
            return "Could not isolate a small blocker set within the search cutoff. ".to_string();
        }
        return String::new();
    }
    let mut s = String::new();
    match report.minimum_size {
        1 => {
            if report.sets.len() == 1 {
                s.push_str(&format!("Blocked by: {}. ", name(report.sets[0][0])));
            } else {
                let names: Vec<String> = report.sets.iter()
                    .map(|set| name(set[0]))
                    .collect();
                s.push_str(&format!(
                    "Blocked by one of: {} (removing any one unblocks). ",
                    names.join(", ")));
            }
        }
        k => {
            let fmt_set = |set: &[u32]| -> String {
                let names: Vec<String> = set.iter().map(|&c| name(c)).collect();
                format!("{{{}}}", names.join(", "))
            };
            if report.sets.len() == 1 {
                s.push_str(&format!(
                    "Blocked jointly by: {} (all {} must be removed). ",
                    fmt_set(&report.sets[0]), k));
            } else {
                let alts: Vec<String> = report.sets.iter().map(|set| fmt_set(set)).collect();
                s.push_str(&format!(
                    "Blocked (min {} constraints); equivalent sets: {}. ",
                    k, alts.join(", ")));
            }
        }
    }
    if report.existing_redundant {
        s.push_str("(Existing constraints contain internal redundancies; blocker set may be approximate.) ");
    }
    s
}

impl CommandContext {
    /// Execute an action: apply to sketch and record in history.
    /// For constraint actions: validates by solving, checking cost, and optionally checking DOF.
    /// Run an action through validation and history. Returns what the
    /// action created -- the ONLY correct identity of a new entity
    /// (arena slots are reused, `refs().last()` names the wrong one
    /// after any delete).
    pub fn exec(&mut self, action: Action) -> Created {
        self.status_error = None;
        self.status_blocker_names = None;

        // Logical conflicts and degenerate geometry, for every action
        // on every path. Not overridable with force -- a contradiction
        // stays a contradiction.
        if let Some(err) = crate::conflicts::validate_action(&self.sketch, &action) {
            self.status_error = Some(err);
            return Created::Nothing;
        }

        if action.is_constraint_action() {
            match validate_and_apply_constraint(
                self.sketch.get_mut(), &action, self.skip_dof_check)
            {
                Ok(new_cost) => {
                    self.last_cost = new_cost;
                    self.history.push(action, &self.sketch, self.saved_cursor.clone());
                }
                Err(rejection) => {
                    self.status_error = Some(rejection.message);
                    if !rejection.blocker_names.is_empty() {
                        self.status_blocker_names = Some(rejection.blocker_names);
                    }
                }
            }
            Created::Nothing
        } else {
            let created = action.apply(self.sketch.get_mut());
            self.sketch.get_mut().dedup_constraints();
            self.history.push(action, &self.sketch, CursorState { pos: self.cursor, tangent: self.cursor_tangent });
            created
        }
    }
}

pub struct CommandResult {
    pub output: String,
    pub is_error: bool,
    pub no_echo: bool,
    pub markdown: bool,
}

fn ok(msg: impl Into<String>) -> CommandResult {
    CommandResult { output: msg.into(), is_error: false, no_echo: false, markdown: false }
}

/// Return ok or the status_error if the last exec was rejected.
/// Also resets skip_dof_check to false.
fn ok_or_status(ctx: &mut CommandContext, msg: impl Into<String>) -> CommandResult {
    ctx.skip_dof_check = false;
    if let Some(e) = ctx.status_error.take() {
        CommandResult { output: e, is_error: true, no_echo: false, markdown: false }
    } else {
        let m = msg.into();
        if m.is_empty() {
            CommandResult { output: m, is_error: false, no_echo: true, markdown: false }
        } else {
            CommandResult { output: m, is_error: false, no_echo: false, markdown: false }
        }
    }
}

/// Format "<name>: <description>" for a just-applied constraint,
/// looking up the canonical descriptive text from
/// `Sketch::find_constraint_description`. Falls back to the supplied
/// fallback string if the name doesn't resolve (shouldn't happen for
/// freshly-applied constraints, but keeps the UX path safe).
fn applied_msg(sketch: &Sketch, name: &str, fallback: &str) -> String {
    match sketch.find_constraint_description(name) {
        Some(desc) => format!("{}: {}", name, desc),
        None => fallback.to_string(),
    }
}

fn err(msg: impl Into<String>) -> CommandResult {
    CommandResult { output: msg.into(), is_error: true, no_echo: false, markdown: false }
}

/// Name of the last dimension in the sketch -- the one just added by
/// the caller's `Action::AddDimension`. Used by the dim-creation
/// commands so their success messages surface the newly-minted
/// `d<N>` identifier.
fn last_dim_name(ctx: &CommandContext) -> String {
    ctx.sketch.dimensions.last().map(|d| d.name.clone()).unwrap_or_default()
}

/// Peel recognized trailing keywords off the token list, any order,
/// any count. Returns one flag per `keys` entry, in order.
fn peel_keywords<const N: usize>(tokens: &mut Vec<&str>, keys: [&str; N]) -> [bool; N] {
    let mut flags = [false; N];
    while let Some(&last) = tokens.last() {
        match keys.iter().position(|&k| k == last) {
            Some(i) => { flags[i] = true; tokens.pop(); }
            None => break,
        }
    }
    flags
}

/// Run one `driven`-keyword AddDimension and produce its message
/// fragment honestly: the created dimension's name on success, the
/// rejection reason otherwise. Reading last_dim_name without checking
/// for rejection would report the previous dimension as if it were
/// the new one.
fn driven_dim_fragment(ctx: &mut CommandContext, action: Action, label: &str, value: f64) -> String {
    ctx.exec(action);
    match ctx.status_error.take() {
        Some(e) => format!(" [driven {} rejected: {}]", label, e),
        None => format!(" [driven {} {}={:.4}]", last_dim_name(ctx), label, value),
    }
}

// ---------------------------------------------------------------------------
// Entity resolution
// ---------------------------------------------------------------------------

fn resolve_line(sketch: &Sketch, name: &str) -> Result<Ref<Line>, String> {
    for r in sketch.lines.refs() {
        if sketch.lines[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown line: {}", name))
}

fn resolve_point(sketch: &Sketch, name: &str) -> Result<Ref<Point>, String> {
    for r in sketch.points.refs() {
        if sketch.points[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown point: {}", name))
}

/// Check if a name looks like an arc/ellipse reference.
fn is_arc_name(name: &str) -> bool {
    name.starts_with('A') || name.starts_with("EA")
}

fn resolve_arc(sketch: &Sketch, name: &str) -> Result<Ref<Arc>, String> {
    for r in sketch.arcs.refs() {
        if sketch.arcs[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown arc: {}", name))
}

/// True when two arcs' current centers coincide within a small
/// geometric tolerance. Used to gate the concentric-distance
/// dimension: the dim installs its own center-coincidence residual,
/// so callers only need to check that the pair is already
/// (approximately) concentric rather than that a paired `Concentric`
/// constraint exists.
fn arcs_are_concentric(sketch: &Sketch, a: Ref<Arc>, b: Ref<Arc>) -> bool {
    let ca = sketch.arcs[a].center.value;
    let cb = sketch.arcs[b].center.value;
    let dx = ca.x - cb.x;
    let dy = ca.y - cb.y;
    (dx * dx + dy * dy).sqrt() < 1e-3
}

/// Resolve an endpoint reference like "L0.p1", "P0", "A0.center"
fn resolve_endpoint_pos(sketch: &Sketch, name: &str) -> Result<vect2d, String> {
    if let Some((entity, field)) = name.split_once('.') {
        if entity.starts_with('L') {
            let r = resolve_line(sketch, entity)?;
            let l = &sketch.lines[r];
            match field {
                "p1" => return Ok(l.p1.value),
                "p2" => return Ok(l.p2.value),
                "p1.x" | "p1.y" | "p2.x" | "p2.y" => {
                    return Err(format!("Use {} as scalar, not coordinate", name));
                }
                _ => return Err(format!("Unknown field: {}.{}", entity, field)),
            }
        } else if is_arc_name(entity) {
            let r = resolve_arc(sketch, entity)?;
            let a = &sketch.arcs[r];
            match field {
                "center" => return Ok(a.center.value),
                "start" => return Ok(vect2d::new(
                    a.center.value.x + a.radius.value * a.start_angle.value.cos(),
                    a.center.value.y + a.radius.value * a.start_angle.value.sin(),
                )),
                "end" => return Ok(vect2d::new(
                    a.center.value.x + a.radius.value * a.end_angle.value.cos(),
                    a.center.value.y + a.radius.value * a.end_angle.value.sin(),
                )),
                _ => return Err(format!("Unknown field: {}.{}", entity, field)),
            }
        }
    }
    // Try as point name
    if name.starts_with('P') {
        let r = resolve_point(sketch, name)?;
        return Ok(sketch.points[r].pos.value);
    }
    Err(format!("Cannot resolve '{}' as coordinate", name))
}

// ---------------------------------------------------------------------------
// Expression evaluation context
// ---------------------------------------------------------------------------

fn eval_context(sketch: &Sketch) -> HashMap<String, f64> {
    let mut vars = HashMap::new();
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        vars.insert(format!("{}.x", p.name), p.pos.value.x);
        vars.insert(format!("{}.y", p.name), p.pos.value.y);
        vars.insert(format!("{}.pos.x", p.name), p.pos.value.x);
        vars.insert(format!("{}.pos.y", p.name), p.pos.value.y);
    }
    for r in sketch.lines.refs() {
        let l = &sketch.lines[r];
        vars.insert(format!("{}.p1.x", l.name), l.p1.value.x);
        vars.insert(format!("{}.p1.y", l.name), l.p1.value.y);
        vars.insert(format!("{}.p2.x", l.name), l.p2.value.x);
        vars.insert(format!("{}.p2.y", l.name), l.p2.value.y);
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        vars.insert(format!("{}.length", l.name), (dx*dx + dy*dy).sqrt());
        vars.insert(format!("{}.angle", l.name), dy.atan2(dx).to_degrees());
    }
    for r in sketch.arcs.refs() {
        let a = &sketch.arcs[r];
        vars.insert(format!("{}.center.x", a.name), a.center.value.x);
        vars.insert(format!("{}.center.y", a.name), a.center.value.y);
        vars.insert(format!("{}.radius", a.name), a.radius.value);
        vars.insert(format!("{}.radius_b", a.name), a.radius_b.value);
        vars.insert(format!("{}.rotation", a.name), a.rotation.value);
        vars.insert(format!("{}.diameter", a.name), a.radius.value * 2.0);
        vars.insert(format!("{}.start_angle", a.name), a.start_angle.value);
        vars.insert(format!("{}.end_angle", a.name), a.end_angle.value);
        vars.insert(format!("{}.sweep", a.name), (a.end_angle.value - a.start_angle.value).abs().to_degrees());
        let sp = crate::geometry::arc_start_pos(a);
        let ep = crate::geometry::arc_end_pos(a);
        vars.insert(format!("{}.start.x", a.name), sp.x);
        vars.insert(format!("{}.start.y", a.name), sp.y);
        vars.insert(format!("{}.end.x", a.name), ep.x);
        vars.insert(format!("{}.end.y", a.name), ep.y);
    }
    for d in &sketch.dimensions {
        vars.insert(d.name.clone(), d.value);
    }
    for p in &sketch.user_params {
        vars.insert(p.name.clone(), p.value);
    }
    vars
}

/// Pre-substitute geometric function calls (angle, dist) in an expression string
/// with their numeric values, so the symbolic parser can handle them.
fn presubst_geo_functions(sketch: &Sketch, expr: &str) -> String {
    let mut result = expr.to_string();
    for fname in &["angle", "dist"] {
        loop {
            let Some(start) = result.find(&format!("{}(", fname)) else { break };
            let after_paren = start + fname.len() + 1;
            // Find matching closing paren
            let mut depth = 1;
            let mut end = after_paren;
            for (i, ch) in result[after_paren..].char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => { depth -= 1; if depth == 0 { end = after_paren + i; break; } }
                    _ => {}
                }
            }
            if depth != 0 { break; }
            let call = &result[start..=end];
            if let Some(Ok(val)) = eval_geo_scalar(sketch, call) {
                result = format!("{}{}{}", &result[..start], val, &result[end + 1..]);
            } else {
                break; // can't evaluate, leave as-is for the parser to report the error
            }
        }
    }
    result
}

fn eval_expr_with(sketch: &Sketch, expr_str: &str, extra: &HashMap<String, f64>) -> Result<f64, String> {
    let expr_str = presubst_geo_functions(sketch, expr_str);
    let parsed = arael_sym::parse(&expr_str).map_err(|e| e.msg)?;
    let mut ctx = eval_context(sketch);
    for (k, v) in extra { ctx.insert(k.clone(), *v); }
    let vars: HashMap<&str, f64> = ctx.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    parsed.eval(&vars)
}

pub fn eval_expr(sketch: &Sketch, expr_str: &str) -> Result<f64, String> {
    eval_expr_with(sketch, expr_str, &HashMap::new())
}

// ---------------------------------------------------------------------------
// Geometric functions (return coordinates or scalars)
// ---------------------------------------------------------------------------

fn eval_geo_coord(sketch: &Sketch, call: &str) -> Option<Result<vect2d, String>> {
    let call = call.trim();
    if !call.contains('(') { return None; }
    let (fname, args_str) = call.split_once('(')?;
    let args_str = args_str.strip_suffix(')')?;
    let fname = fname.trim();
    let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

    Some(match fname {
        "intersect" => {
            if args.len() != 2 { return Some(Err("intersect(L0, L1)".into())); }
            let la = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let lb = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let a = &sketch.lines[la]; let b = &sketch.lines[lb];
            let d1 = vect2d::new(a.p2.value.x - a.p1.value.x, a.p2.value.y - a.p1.value.y);
            let d2 = vect2d::new(b.p2.value.x - b.p1.value.x, b.p2.value.y - b.p1.value.y);
            let cross = d1.x * d2.y - d1.y * d2.x;
            if cross.abs() < 1e-12 { return Some(Err("Lines are parallel".into())); }
            let dx = b.p1.value.x - a.p1.value.x;
            let dy = b.p1.value.y - a.p1.value.y;
            let t = (dx * d2.y - dy * d2.x) / cross;
            Ok(vect2d::new(a.p1.value.x + t * d1.x, a.p1.value.y + t * d1.y))
        }
        "midpoint" => {
            if args.len() != 1 { return Some(Err("midpoint(L0)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            Ok(vect2d::new((line.p1.value.x + line.p2.value.x) / 2.0,
                           (line.p1.value.y + line.p2.value.y) / 2.0))
        }
        "project" => {
            if args.len() != 2 { return Some(Err("project(P0, L0)".into())); }
            let pt = match resolve_endpoint_pos(sketch, args[0]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let l = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len2 = dx * dx + dy * dy;
            if len2 < 1e-24 { return Some(Ok(line.p1.value)); }
            let t = ((pt.x - line.p1.value.x) * dx + (pt.y - line.p1.value.y) * dy) / len2;
            Ok(vect2d::new(line.p1.value.x + t * dx, line.p1.value.y + t * dy))
        }
        "along" => {
            if args.len() != 2 { return Some(Err("along(L0, 0.5)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let t = match eval_expr(sketch, args[1]) { Ok(v) => v, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            Ok(vect2d::new(line.p1.value.x + t * (line.p2.value.x - line.p1.value.x),
                           line.p1.value.y + t * (line.p2.value.y - line.p1.value.y)))
        }
        "arc_point" => {
            if args.len() != 2 { return Some(Err("arc_point(A0, 45)".into())); }
            let a = match resolve_arc(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let angle_deg = match eval_expr(sketch, args[1]) { Ok(v) => v, Err(e) => return Some(Err(e)) };
            let arc = &sketch.arcs[a];
            let angle = angle_deg.to_radians();
            Ok(vect2d::new(arc.center.value.x + arc.radius.value * angle.cos(),
                           arc.center.value.y + arc.radius.value * angle.sin()))
        }
        "rotate" => {
            // rotate(point, center, angle_deg)
            if args.len() != 3 { return Some(Err("rotate(P0, center, angle_deg)".into())); }
            let pt = match resolve_endpoint_pos(sketch, args[0]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let center = match resolve_endpoint_pos(sketch, args[1]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let angle = match eval_expr(sketch, args[2]) { Ok(v) => v.to_radians(), Err(e) => return Some(Err(e)) };
            let dx = pt.x - center.x;
            let dy = pt.y - center.y;
            let c = angle.cos(); let s = angle.sin();
            Ok(vect2d::new(center.x + dx * c - dy * s, center.y + dx * s + dy * c))
        }
        "mirror" => {
            if args.len() != 2 { return Some(Err("mirror(P0, L0)".into())); }
            let pt = match resolve_endpoint_pos(sketch, args[0]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let l = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len2 = dx * dx + dy * dy;
            if len2 < 1e-24 { return Some(Ok(pt)); }
            let t = ((pt.x - line.p1.value.x) * dx + (pt.y - line.p1.value.y) * dy) / len2;
            let proj = vect2d::new(line.p1.value.x + t * dx, line.p1.value.y + t * dy);
            Ok(vect2d::new(2.0 * proj.x - pt.x, 2.0 * proj.y - pt.y))
        }
        "tangent" => {
            if args.len() != 1 { return Some(Err("tangent(L0)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { return Some(Err("Zero-length line".into())); }
            Ok(vect2d::new(dx / len, dy / len))
        }
        "normal" => {
            if args.len() != 1 { return Some(Err("normal(L0)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { return Some(Err("Zero-length line".into())); }
            Ok(vect2d::new(-dy / len, dx / len))
        }
        _ => return None,
    })
}

fn eval_geo_scalar(sketch: &Sketch, call: &str) -> Option<Result<f64, String>> {
    let call = call.trim();
    if !call.contains('(') { return None; }
    let (fname, args_str) = call.split_once('(')?;
    let args_str = args_str.strip_suffix(')')?;
    let fname = fname.trim();
    let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

    Some(match fname {
        "dist" => {
            if args.len() == 2 {
                // Try point-point
                if let (Ok(a), Ok(b)) = (resolve_endpoint_pos(sketch, args[0]), resolve_endpoint_pos(sketch, args[1])) {
                    let dx = a.x - b.x; let dy = a.y - b.y;
                    return Some(Ok((dx * dx + dy * dy).sqrt()));
                }
                // Try point-to-line
                if let (Ok(pt), Ok(lr)) = (resolve_endpoint_pos(sketch, args[0]), resolve_line(sketch, args[1])) {
                    let l = &sketch.lines[lr];
                    let dx = l.p2.value.x - l.p1.value.x;
                    let dy = l.p2.value.y - l.p1.value.y;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len < 1e-12 { return Some(Ok(((l.p1.value.x - pt.x).powi(2) + (l.p1.value.y - pt.y).powi(2)).sqrt())); }
                    let cross = ((pt.x - l.p1.value.x) * dy - (pt.y - l.p1.value.y) * dx).abs();
                    return Some(Ok(cross / len));
                }
                Err(format!("Cannot resolve dist({}, {})", args[0], args[1]))
            } else {
                Err("dist(P0, P1) or dist(P0, L0)".into())
            }
        }
        "angle" => {
            if args.len() != 2 { return Some(Err("angle(L0, L1)".into())); }
            let la = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let lb = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let a = &sketch.lines[la]; let b = &sketch.lines[lb];
            let d1x = a.p2.value.x - a.p1.value.x; let d1y = a.p2.value.y - a.p1.value.y;
            let d2x = b.p2.value.x - b.p1.value.x; let d2y = b.p2.value.y - b.p1.value.y;
            let cross = d1x * d2y - d1y * d2x;
            let dot = d1x * d2x + d1y * d2y;
            Ok(cross.atan2(dot).to_degrees())
        }
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Coordinate parsing
// ---------------------------------------------------------------------------

fn parse_coord(ctx: &CommandContext, arg: &str, cursor: Option<vect2d>) -> Result<vect2d, String> {
    let sketch = &ctx.sketch;
    let arg = arg.trim();
    // Special @-keywords
    if arg == "@tangent" {
        return ctx.cursor_tangent.ok_or("No tangent direction available (set by add_line/add_arc/add_earc)".into());
    }
    if arg == "@cursor" || arg == "cursor" {
        return cursor.ok_or("No cursor position available".into());
    }
    // Relative coordinate: @dx,dy
    if let Some(rest) = arg.strip_prefix('@') {
        let prev = cursor.ok_or("No previous point for relative coordinate")?;
        if let Some((x_str, y_str)) = rest.split_once(',') {
            let dx = eval_expr_with(sketch, x_str.trim(), &ctx.session_vars)?;
            let dy = eval_expr_with(sketch, y_str.trim(), &ctx.session_vars)?;
            return Ok(vect2d::new(prev.x + dx, prev.y + dy));
        }
        return Err(format!("Relative coordinate needs @dx,dy format: {}", arg));
    }
    // Cursor keyword (kept for backward compatibility)
    if arg == "cursor" {
        return ctx.cursor.ok_or("Cursor not set".into());
    }
    // Geometric function: intersect(L0, L1), midpoint(L0), tangent(L0), etc.
    if let Some(result) = eval_geo_coord(sketch, arg) {
        return result;
    }
    // Session vector variable
    if let Some(v) = ctx.session_vecs.get(arg) {
        return Ok(*v);
    }
    // Endpoint reference: L0.p2, P0, A0.center
    if let Ok(pos) = resolve_endpoint_pos(sketch, arg) {
        return Ok(pos);
    }
    // x,y with possible expressions
    if let Some((x_str, y_str)) = arg.split_once(',') {
        let x = eval_expr_with(sketch, x_str.trim(), &ctx.session_vars)?;
        let y = eval_expr_with(sketch, y_str.trim(), &ctx.session_vars)?;
        return Ok(vect2d::new(x, y));
    }
    // Vector expression: L0.p2 + normal(L0) * 3, etc.
    // Pre-expand vector sub-expressions to temp vars, then eval as two scalar expressions.
    if let Ok(v) = parse_vec_expr(ctx, arg) {
        return Ok(v);
    }
    Err(format!("Cannot parse coordinate: {}", arg))
}

/// Parse a vector expression like `L0.p2 + normal(L0) * 3`.
/// Scans for vector-valued sub-expressions, replaces them with temp vars,
/// then evaluates x and y components as separate scalar expressions.
fn parse_vec_expr(ctx: &CommandContext, expr: &str) -> Result<vect2d, String> {
    let sketch = &ctx.sketch;
    let mut tmp_vars: HashMap<String, f64> = ctx.session_vars.clone();
    let mut work = expr.to_string();
    let mut counter = 0u32;

    // Repeatedly scan for vector-valued tokens and replace them
    for _ in 0..32 {
        let mut found = false;
        // Find function calls: name(args)
        if let Some(start) = find_func_call(&work) {
            let call = &work[start.0..start.1];
            if let Some(Ok(v)) = eval_geo_coord(sketch, call) {
                let name = format!("__v{}", counter);
                counter += 1;
                tmp_vars.insert(format!("{}.x", name), v.x);
                tmp_vars.insert(format!("{}.y", name), v.y);
                work = format!("{}{}{}", &work[..start.0], name, &work[start.1..]);
                found = true;
            }
        }
        if found { continue; }
        // Find endpoint refs and session vecs: scan for identifiers that resolve to vec2
        let tokens = extract_identifiers(&work);
        let mut replaced = false;
        for (tstart, tend, token) in tokens.iter().rev() {
            // Skip if already a temp var
            if token.starts_with("__v") { continue; }
            // Try session vec
            if let Some(v) = ctx.session_vecs.get(*token) {
                let name = format!("__v{}", counter);
                counter += 1;
                tmp_vars.insert(format!("{}.x", name), v.x);
                tmp_vars.insert(format!("{}.y", name), v.y);
                work = format!("{}{}{}", &work[..*tstart], name, &work[*tend..]);
                replaced = true;
                break;
            }
            // Try endpoint ref (L0.p2, P0, A0.center)
            if let Ok(v) = resolve_endpoint_pos(sketch, token) {
                let name = format!("__v{}", counter);
                counter += 1;
                tmp_vars.insert(format!("{}.x", name), v.x);
                tmp_vars.insert(format!("{}.y", name), v.y);
                work = format!("{}{}{}", &work[..*tstart], name, &work[*tend..]);
                replaced = true;
                break;
            }
        }
        if replaced { continue; }
        break;
    }

    // Now build x and y expressions by appending .x/.y to __vN tokens
    let x_expr = work.replace("__v", "__v").split("__v").enumerate().map(|(i, part)| {
        if i == 0 { part.to_string() }
        else {
            // part starts with the number and rest
            let num_end = part.find(|c: char| !c.is_ascii_digit()).unwrap_or(part.len());
            format!("__v{}.x{}", &part[..num_end], &part[num_end..])
        }
    }).collect::<String>();
    let y_expr = work.split("__v").enumerate().map(|(i, part)| {
        if i == 0 { part.to_string() }
        else {
            let num_end = part.find(|c: char| !c.is_ascii_digit()).unwrap_or(part.len());
            format!("__v{}.y{}", &part[..num_end], &part[num_end..])
        }
    }).collect::<String>();

    let x = eval_expr_with(sketch, &x_expr, &tmp_vars)?;
    let y = eval_expr_with(sketch, &y_expr, &tmp_vars)?;
    Ok(vect2d::new(x, y))
}

/// Find the first function call `name(...)` in the string, return (start, end) byte offsets.
fn find_func_call(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // Walk back to find function name start
            let paren_pos = i;
            if paren_pos == 0 || !bytes[paren_pos - 1].is_ascii_alphanumeric() {
                i += 1; continue;
            }
            let mut name_start = paren_pos;
            while name_start > 0 && (bytes[name_start - 1].is_ascii_alphanumeric() || bytes[name_start - 1] == b'_') {
                name_start -= 1;
            }
            // Find matching closing paren
            let mut depth = 1;
            let mut j = paren_pos + 1;
            while j < bytes.len() && depth > 0 {
                if bytes[j] == b'(' { depth += 1; }
                if bytes[j] == b')' { depth -= 1; }
                j += 1;
            }
            if depth == 0 {
                let fname = &s[name_start..paren_pos];
                // Only match known geo functions
                if matches!(fname, "intersect" | "midpoint" | "project" | "along" | "arc_point" |
                    "rotate" | "mirror" | "tangent" | "normal") {
                    return Some((name_start, j));
                }
            }
        }
        i += 1;
    }
    None
}

/// Extract identifiers from an expression string as (start, end, &str) triples.
fn extract_identifiers(s: &str) -> Vec<(usize, usize, &str)> {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.') {
                i += 1;
            }
            result.push((start, i, &s[start..i]));
        } else {
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Endpoint resolution for coincident/constraints
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum EndpointRef {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
}

fn resolve_endpoint_ref(sketch: &Sketch, name: &str) -> Result<EndpointRef, String> {
    if let Some((entity, field)) = name.split_once('.') {
        if entity.starts_with('L') {
            let r = resolve_line(sketch, entity)?;
            match field {
                "p1" => return Ok(EndpointRef::LineP1(r)),
                "p2" => return Ok(EndpointRef::LineP2(r)),
                _ => return Err(format!("Line endpoint must be p1 or p2: {}", name)),
            }
        } else if is_arc_name(entity) {
            let r = resolve_arc(sketch, entity)?;
            match field {
                "center" => return Ok(EndpointRef::ArcCenter(r)),
                "start" => return Ok(EndpointRef::ArcStart(r)),
                "end" => return Ok(EndpointRef::ArcEnd(r)),
                _ => return Err(format!("Arc endpoint must be center/start/end: {}", name)),
            }
        }
    }
    if name.starts_with('P') {
        let r = resolve_point(sketch, name)?;
        return Ok(EndpointRef::Point(r));
    }
    Err(format!("Cannot parse endpoint: {}", name))
}

// ---------------------------------------------------------------------------
// Command execution
// ---------------------------------------------------------------------------

/// Replace session name aliases in a string (word-boundary aware).
/// Replace session-name aliases with their entity names. One pass,
/// left to right, longest alias first at each position. Replaced text
/// is never rescanned, so an alias whose entity name collides with
/// another alias's name cannot chain, and the result does not depend
/// on map iteration order.
fn substitute_aliases(ctx: &CommandContext, input: &str) -> String {
    if ctx.session_names.is_empty() { return input.to_string(); }
    let mut aliases: Vec<(&str, &str)> = ctx.session_names.iter()
        .map(|(a, r)| (a.as_str(), r.as_str())).collect();
    aliases.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(b.0)));
    let bytes = input.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let boundary_before = i == 0 || !is_word(bytes[i - 1]);
        if boundary_before {
            if let Some((alias, real)) = aliases.iter().find(|(a, _)| {
                input[i..].starts_with(a)
                    && (i + a.len() >= input.len() || !is_word(bytes[i + a.len()]))
            }) {
                out.push_str(real);
                i += alias.len();
                continue;
            }
        }
        let ch_len = input[i..].chars().next().map_or(1, |c| c.len_utf8());
        out.push_str(&input[i..i + ch_len]);
        i += ch_len;
    }
    out
}

pub fn execute(ctx: &mut CommandContext, input: &str) -> Vec<CommandResult> {
    use std::io::Write;
    let mut results = Vec::new();
    for cmd in input.split(';') {
        let cmd = cmd.trim();
        if cmd.is_empty() { continue; }
        if ctx.echo_stdout && !cmd.starts_with('#') {
            println!("> {}", cmd);
            let _ = std::io::stdout().flush();
        }
        let mut r = execute_one(ctx, cmd);
        if r.is_error && !r.output.starts_with('>') {
            r.output = format!("'{}': {}", cmd, r.output);
        }
        if ctx.echo_stdout && !r.output.is_empty() {
            println!("{}", r.output);
            let _ = std::io::stdout().flush();
        }
        let is_err = r.is_error;
        results.push(r);
        // Commands that delete entities or replace the sketch (delete,
        // clear, load, undo, redo, goto) leave the selection pointing
        // at freed arena slots or shifted indices; any later read
        // panics. Prune after every command.
        prune_selection(ctx);
        if is_err { break; }
    }
    if results.is_empty() {
        results.push(ok(""));
    }
    let dof_start = results.len();
    append_dof_tail(ctx, input, &mut results);
    if ctx.echo_stdout {
        for r in &results[dof_start..] {
            if !r.output.is_empty() {
                println!("{}", r.output);
                let _ = std::io::stdout().flush();
            }
        }
    }
    results
}

/// Append a `DOF: <n>` summary line at the end of a command
/// sequence so the user always knows the current degree-of-freedom
/// state after their edits -- especially important when the
/// sequence errored out partway through, since the user needs to
/// see how their edits left the sketch before diagnosing the
/// failure. Skipped only when:
///
/// - the sketch is empty (no entities -> no meaningful DOF);
/// - every command in the sequence is purely observational
///   (`dof*`, `list`, `info`, `find`, `help`, `print`, `cost`,
///   `history`, `measure`, `msg`) -- DOF hasn't changed, no need
///   to re-announce it.
fn append_dof_tail(ctx: &mut CommandContext, input: &str, results: &mut Vec<CommandResult>) {
    if ctx.sketch.lines.refs().next().is_none()
        && ctx.sketch.points.refs().next().is_none()
        && ctx.sketch.arcs.refs().next().is_none() {
        return;
    }
    const OBSERVATIONAL: &[&str] = &[
        "dof", "list", "info", "find", "help", "print", "cost",
        "history", "measure", "msg",
    ];
    let all_observational = input.split(';').map(|c| c.trim()).filter(|c| !c.is_empty())
        .all(|c| {
            let head = c.split_whitespace().next().unwrap_or("");
            OBSERVATIONAL.contains(&head)
        });
    if all_observational { return; }
    let Ok(dof) = ctx.sketch.dof() else { return };
    results.push(CommandResult {
        output: format!("DOF: {}", dof),
        is_error: false,
        no_echo: false,
        markdown: false,
    });
}

fn execute_one(ctx: &mut CommandContext, input: &str) -> CommandResult {
    let input = input.trim();

    // Strip inline comments (# not inside quotes) first, then the
    // trailing "force" keyword, so `horizontal L0 force # note`
    // parses. `msg` is exempt from both: its text is verbatim.
    let (input, force) = if input.starts_with("msg ") || input == "msg" {
        (input, false)
    } else {
        strip_force(strip_inline_comment(input))
    };
    ctx.skip_dof_check = force;
    let input = input.trim();

    // Comments (entire line)
    if input.is_empty() || input.starts_with('#') {
        return CommandResult { output: String::new(), is_error: false, no_echo: true, markdown: false };
    }

    // Assignment: "name = command args" or "let name = ..."
    let assign_input = input.strip_prefix("let ").map(|s| (true, s)).unwrap_or((false, input));
    if let Some((lhs, rhs)) = assign_input.1.split_once('=') {
        let var_name = lhs.trim();
        let rhs = rhs.trim();
        // Multi-assignment: "a, b, c = add_line ..."
        if var_name.contains(',') {
            let names: Vec<&str> = var_name.split(',').map(|s| s.trim()).collect();
            if names.iter().all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')) {
                let result = execute_one(ctx, rhs);
                if !result.is_error {
                    for (i, name) in names.iter().enumerate() {
                        if let Some(entity) = ctx.session_names.get(&format!("_{}", i)).cloned() {
                            ctx.session_names.insert(name.to_string(), entity);
                        }
                    }
                }
                return result;
            }
        }
        // Check if lhs is a simple identifier
        if !var_name.is_empty()
            && !var_name.contains('.')
            && !var_name.contains(' ')
            && var_name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            && !var_name.bytes().next().unwrap_or(b'0').is_ascii_digit()
        {
            // Try as a command first (e.g. "base = add_line 0,0 5,0")
            let first_word = rhs.split_whitespace().next().unwrap_or("");
            let is_command = matches!(first_word,
                "add_line" | "add_rect" | "add_rect3" | "add_rectcenter" |
                "add_point" | "add_circle" | "add_circle2" | "add_circle3" |
                "add_circle2t" | "add_circle3t" | "add_ellipse" | "add_arc" |
                "add_earc" | "add_earc3" | "add_earc_center" | "add_earc_tangent" | "add_earc_rtangent" | "offset_line" | "offset" | "fillet" | "chamfer" | "mirror" |
                "length" | "radius" | "radius_b" | "sweep" | "angle" | "distance");
            if is_command {
                let dim_count_before = ctx.sketch.dimensions.len();
                let prev_entity = ctx.session_names.get("_").cloned();
                let result = execute_one(ctx, rhs);
                if !result.is_error {
                    // Check for new entity name (only if "_" was updated by this command)
                    let cur_entity = ctx.session_names.get("_").cloned();
                    let entity_captured = if cur_entity != prev_entity {
                        if let Some(entity_name) = cur_entity {
                            ctx.session_names.insert(var_name.to_string(), entity_name);
                            true
                        } else { false }
                    } else {
                        false
                    };
                    // Check for new dimension — dimension commands (length, angle, etc.)
                    // don't set "_" like geometry commands do, so we detect new dimensions
                    // by comparing count before/after and capture the dimension name.
                    // Only when the command didn't already produce an entity (e.g. "driven"
                    // on geometry commands creates dimensions as a side effect).
                    if !entity_captured && ctx.sketch.dimensions.len() > dim_count_before
                        && let Some(dim) = ctx.sketch.dimensions.last() {
                            ctx.session_names.insert(var_name.to_string(), dim.name.clone());
                        }
                }
                return result;
            }
            // Otherwise treat as scalar/vector assignment (existing let behavior)
            return cmd_let(ctx, &format!("{} = {}", var_name, rhs));
        }
    }

    // Substitute session_names aliases in arguments
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0];

    // Check if command is blocked in this context
    if ctx.blocked_commands.contains(&cmd) {
        return err(format!("'{}' is not allowed in this context", cmd));
    }
    let raw_args = if parts.len() > 1 { parts[1].trim() } else { "" };

    // Replace known aliases in args (word-boundary aware)
    let args_str = substitute_aliases(ctx, raw_args);
    let args_str = args_str.as_str();

    match cmd {
        "explain" => cmd_explain(ctx, args_str),
        "add_line" => cmd_add_line(ctx, args_str),
        "add_rect" => cmd_add_rect(ctx, args_str),
        "add_rect3" => cmd_add_rect3(ctx, args_str),
        "add_rectcenter" => cmd_add_rectcenter(ctx, args_str),
        "add_point" => cmd_add_point(ctx, args_str),
        "add_circle" => cmd_add_circle(ctx, args_str),
        "add_circle2" => cmd_add_circle2(ctx, args_str),
        "add_circle3" => cmd_add_circle3(ctx, args_str),
        "add_circle2t" => cmd_add_circle2t(ctx, args_str),
        "add_circle3t" => cmd_add_circle3t(ctx, args_str),
        "add_ellipse" => cmd_add_ellipse(ctx, args_str),
        "add_arc" => cmd_add_arc(ctx, args_str),
        "add_earc" => cmd_add_earc(ctx, args_str),
        "add_earc3" => cmd_add_earc3(ctx, args_str),
        "add_earc_center" => cmd_add_earc_center(ctx, args_str),
        "add_earc_tangent" => cmd_add_earc_tangent(ctx, args_str),
        "add_earc_rtangent" => cmd_add_earc_rtangent(ctx, args_str),
        "offset_line" | "offset" => cmd_offset_line(ctx, args_str),
        "fillet" => cmd_fillet(ctx, args_str),
        "chamfer" => cmd_chamfer(ctx, args_str),
        "delete" => cmd_delete(ctx, args_str),
        "horizontal" => cmd_horizontal(ctx, args_str),
        "vertical" => cmd_vertical(ctx, args_str),
        "parallel" => cmd_parallel(ctx, args_str),
        "perpendicular" | "perp" => cmd_perpendicular(ctx, args_str),
        "equal" => cmd_equal(ctx, args_str),
        "collinear" => cmd_collinear(ctx, args_str),
        "tangent" => cmd_tangent(ctx, args_str),
        "coincident" => cmd_coincident(ctx, args_str),
        "concentric" => cmd_concentric(ctx, args_str),
        "midpoint" => cmd_midpoint(ctx, args_str),
        "symmetry" => cmd_symmetry(ctx, args_str),
        "mirror" => cmd_mirror(ctx, args_str),
        "point_on" => cmd_point_on(ctx, args_str),
        "length" => cmd_length(ctx, args_str),
        "radius" => cmd_radius(ctx, args_str),
        "radius_b" => cmd_radius_b(ctx, args_str),
        "sweep" => cmd_sweep(ctx, args_str),
        "angle" => cmd_angle(ctx, args_str),
        "distance" => cmd_distance(ctx, args_str),
        "hdistance" => cmd_hdistance(ctx, args_str),
        "vdistance" => cmd_vdistance(ctx, args_str),
        "xangle" => cmd_xangle(ctx, args_str),
        "lock" => cmd_lock(ctx, args_str),
        "unlock" => cmd_unlock(ctx, args_str),
        "param" => cmd_param(ctx, args_str),
        "del_param" => cmd_del_param(ctx, args_str),
        "rename_param" => cmd_rename_param(ctx, args_str),
        "style" => cmd_style(ctx, args_str),
        "quiet" => cmd_quiet(ctx, args_str),
        "constr" => cmd_constr(ctx, args_str),
        "drag" => cmd_drag(ctx, args_str),
        "select" => cmd_select(ctx, args_str),
        "deselect" => cmd_deselect(ctx, args_str),
        "print" => cmd_print(ctx, args_str),
        "info" => cmd_info(ctx, args_str),
        "measure" => cmd_measure(ctx, args_str),
        "list" => cmd_list(ctx, args_str),
        "find" => cmd_find(ctx, args_str),
        "dof" => cmd_dof(ctx, args_str),
        "cost" => {
            let cost = ctx.sketch.current_cost();
            ok(format!("Cost: {:.6}", cost))
        }
        "undo" => cmd_undo(ctx, args_str),
        "redo" => cmd_redo(ctx, args_str),
        "history" => cmd_history(ctx, args_str),
        "goto" => cmd_goto(ctx, args_str),
        "center" => cmd_center(ctx, args_str),
        "zoom" => cmd_zoom(ctx, args_str),
        "msg" => CommandResult {
            output: args_str.replace("\\n", "\n"), is_error: false, no_echo: true, markdown: true,
        },
        "cursor" => cmd_cursor(ctx, args_str),
        "dim_pos" => cmd_dim_pos(ctx, args_str),
        "set_derived" => cmd_set_derived(ctx, args_str),
        "set_driven" => cmd_set_driven(ctx, args_str),
        "freeze" => cmd_freeze(ctx, args_str),
        "clear" => { ctx.sketch = Sketch::new().into(); ctx.history = crate::history::History::new(&ctx.sketch); ok("Cleared") },
        "let" => cmd_let(ctx, args_str),
        "save" => cmd_save(ctx, args_str),
        "load" => cmd_load(ctx, args_str),
        "help" => cmd_help(args_str),
        "exit" | "quit" => { ctx.exit_requested = true; ok("Exiting") },
        "ai" => ok("AI assistant not yet configured. Use --mcp to enable MCP server for external AI agents."),
        _ if cmd.starts_with('!') => ok("AI assistant not yet configured. Use --mcp to enable MCP server for external AI agents."),
        _ => err(format!("Unknown command: {}. Type 'help' for commands.", cmd)),
    }
}

// ---------------------------------------------------------------------------
// Geometry commands
// ---------------------------------------------------------------------------

const SNAP_THRESHOLD: f64 = 1e-3;

fn snap_near(a: vect2d, b: vect2d) -> bool {
    (a.x - b.x).abs() < SNAP_THRESHOLD && (a.y - b.y).abs() < SNAP_THRESHOLD
}

/// Auto-connect endpoints of the last created line to nearby existing endpoints.
fn auto_coincident_line(ctx: &mut CommandContext, line_ref: Ref<Line>) -> Vec<String> {
    let mut actions: Vec<(Action, String)> = Vec::new();
    let l = &ctx.sketch.lines[line_ref];
    let p1 = l.p1.value;
    let p2 = l.p2.value;
    let this_name = l.name.clone();

    for r in ctx.sketch.lines.refs() {
        if r == line_ref { continue; }
        let other = &ctx.sketch.lines[r];
        if snap_near(p1, other.p1.value) {
            actions.push((Action::ApplyCoincidentLL11 { a: line_ref, b: r },
                format!("{}.p1={}.p1", this_name, other.name)));
        } else if snap_near(p1, other.p2.value) {
            actions.push((Action::ApplyCoincidentLL12 { a: line_ref, b: r },
                format!("{}.p1={}.p2", this_name, other.name)));
        }
        if snap_near(p2, other.p1.value) {
            actions.push((Action::ApplyCoincidentLL21 { a: line_ref, b: r },
                format!("{}.p2={}.p1", this_name, other.name)));
        } else if snap_near(p2, other.p2.value) {
            actions.push((Action::ApplyCoincidentLL22 { a: line_ref, b: r },
                format!("{}.p2={}.p2", this_name, other.name)));
        }
    }
    for r in ctx.sketch.points.refs() {
        let pt = &ctx.sketch.points[r];
        if pt.helper { continue; }
        if snap_near(p1, pt.pos.value) {
            actions.push((Action::ApplyCoincidentLP1 { line: line_ref, point: r },
                format!("{}.p1={}", this_name, pt.name)));
        }
        if snap_near(p2, pt.pos.value) {
            actions.push((Action::ApplyCoincidentLP2 { line: line_ref, point: r },
                format!("{}.p2={}", this_name, pt.name)));
        }
    }
    for r in ctx.sketch.arcs.refs() {
        let arc = &ctx.sketch.arcs[r];
        let ac = arc.center.value;
        let a_start = arc_start_pos(arc);
        let a_end = arc_end_pos(arc);
        if snap_near(p1, ac) {
            actions.push((Action::ApplyCoincidentLP1ArcCenter { line: line_ref, arc: r },
                format!("{}.p1={}.center", this_name, arc.name)));
        }
        if snap_near(p1, a_start) {
            actions.push((Action::ApplyCoincidentLP1ArcStart { line: line_ref, arc: r },
                format!("{}.p1={}.start", this_name, arc.name)));
        }
        if snap_near(p1, a_end) {
            actions.push((Action::ApplyCoincidentLP1ArcEnd { line: line_ref, arc: r },
                format!("{}.p1={}.end", this_name, arc.name)));
        }
        if snap_near(p2, ac) {
            actions.push((Action::ApplyCoincidentLP2ArcCenter { line: line_ref, arc: r },
                format!("{}.p2={}.center", this_name, arc.name)));
        }
        if snap_near(p2, a_start) {
            actions.push((Action::ApplyCoincidentLP2ArcStart { line: line_ref, arc: r },
                format!("{}.p2={}.start", this_name, arc.name)));
        }
        if snap_near(p2, a_end) {
            actions.push((Action::ApplyCoincidentLP2ArcEnd { line: line_ref, arc: r },
                format!("{}.p2={}.end", this_name, arc.name)));
        }
    }
    let mut connected = Vec::new();
    let saved = ctx.skip_dof_check;
    ctx.skip_dof_check = true; // auto-coincident is positional, don't DOF-check
    for (action, desc) in actions {
        let watermark = ctx.sketch.next_constraint_id;
        ctx.exec(action);
        // A rejected auto-coincident is not reported as connected.
        if ctx.status_error.take().is_some() { continue; }
        let ids = constraint_names_since(ctx, watermark);
        connected.push(if ids.is_empty() { desc } else { format!("{} ({})", desc, ids.join(" ")) });
    }
    ctx.skip_dof_check = saved;
    connected
}

/// Auto-connect arc endpoints to nearby existing geometry.
/// center_only=true for circles (start/end are edge points, not snap targets).
fn auto_coincident_arc(ctx: &mut CommandContext, arc_ref: Ref<Arc>, center_only: bool) -> Vec<String> {
    let mut actions: Vec<(Action, String)> = Vec::new();
    let arc = &ctx.sketch.arcs[arc_ref];
    let center = arc.center.value;
    let start = arc_start_pos(arc);
    let end = arc_end_pos(arc);
    let this_name = arc.name.clone();

    // Check against line endpoints
    for r in ctx.sketch.lines.refs() {
        let line = &ctx.sketch.lines[r];
        let lp1 = line.p1.value;
        let lp2 = line.p2.value;
        if snap_near(center, lp1) {
            actions.push((Action::ApplyCoincidentLP1ArcCenter { line: r, arc: arc_ref },
                format!("{}.center={}.p1", this_name, line.name)));
        }
        if snap_near(center, lp2) {
            actions.push((Action::ApplyCoincidentLP2ArcCenter { line: r, arc: arc_ref },
                format!("{}.center={}.p2", this_name, line.name)));
        }
        if !center_only {
            if snap_near(start, lp1) {
                actions.push((Action::ApplyCoincidentLP1ArcStart { line: r, arc: arc_ref },
                    format!("{}.start={}.p1", this_name, line.name)));
            }
            if snap_near(start, lp2) {
                actions.push((Action::ApplyCoincidentLP2ArcStart { line: r, arc: arc_ref },
                    format!("{}.start={}.p2", this_name, line.name)));
            }
            if snap_near(end, lp1) {
                actions.push((Action::ApplyCoincidentLP1ArcEnd { line: r, arc: arc_ref },
                    format!("{}.end={}.p1", this_name, line.name)));
            }
            if snap_near(end, lp2) {
                actions.push((Action::ApplyCoincidentLP2ArcEnd { line: r, arc: arc_ref },
                    format!("{}.end={}.p2", this_name, line.name)));
            }
        }
    }

    // Check against other arc endpoints
    for r in ctx.sketch.arcs.refs() {
        if r == arc_ref { continue; }
        let other = &ctx.sketch.arcs[r];
        let oc = other.center.value;
        let os = arc_start_pos(other);
        let oe = arc_end_pos(other);
        if snap_near(center, oc) {
            actions.push((Action::ApplyConcentric { a: arc_ref, b: r },
                format!("{}.center={}.center", this_name, other.name)));
        }
        if snap_near(center, os) {
            actions.push((Action::ApplyCoincidentArcCenterStart { a: arc_ref, b: r },
                format!("{}.center={}.start", this_name, other.name)));
        }
        if snap_near(center, oe) {
            actions.push((Action::ApplyCoincidentArcCenterEnd { a: arc_ref, b: r },
                format!("{}.center={}.end", this_name, other.name)));
        }
        if !center_only {
            if snap_near(start, oc) {
                actions.push((Action::ApplyCoincidentArcStartCenter { a: arc_ref, b: r },
                    format!("{}.start={}.center", this_name, other.name)));
            }
            if snap_near(start, os) {
                actions.push((Action::ApplyCoincidentArcStartStart { a: arc_ref, b: r },
                    format!("{}.start={}.start", this_name, other.name)));
            }
            if snap_near(start, oe) {
                actions.push((Action::ApplyCoincidentArcStartEnd { a: arc_ref, b: r },
                    format!("{}.start={}.end", this_name, other.name)));
            }
            if snap_near(end, oc) {
                actions.push((Action::ApplyCoincidentArcEndCenter { a: arc_ref, b: r },
                    format!("{}.end={}.center", this_name, other.name)));
            }
            if snap_near(end, os) {
                actions.push((Action::ApplyCoincidentArcEndStart { a: arc_ref, b: r },
                    format!("{}.end={}.start", this_name, other.name)));
            }
            if snap_near(end, oe) {
                actions.push((Action::ApplyCoincidentArcEndEnd { a: arc_ref, b: r },
                    format!("{}.end={}.end", this_name, other.name)));
            }
        }
    }

    // Check against free points (skip helpers)
    for r in ctx.sketch.points.refs() {
        let pt = &ctx.sketch.points[r];
        if pt.helper { continue; }
        if snap_near(center, pt.pos.value) {
            actions.push((Action::ApplyCoincidentArcCenter { point: r, arc: arc_ref },
                format!("{}.center={}", this_name, pt.name)));
        }
        if !center_only {
            if snap_near(start, pt.pos.value) {
                actions.push((Action::ApplyCoincidentArcStart { point: r, arc: arc_ref },
                    format!("{}.start={}", this_name, pt.name)));
            }
            if snap_near(end, pt.pos.value) {
                actions.push((Action::ApplyCoincidentArcEnd { point: r, arc: arc_ref },
                    format!("{}.end={}", this_name, pt.name)));
            }
        }
    }

    let mut connected = Vec::new();
    let saved = ctx.skip_dof_check;
    ctx.skip_dof_check = true; // auto-coincident is positional, don't DOF-check
    for (action, desc) in actions {
        let watermark = ctx.sketch.next_constraint_id;
        ctx.exec(action);
        // A rejected auto-coincident is not reported as connected.
        if ctx.status_error.take().is_some() { continue; }
        let ids = constraint_names_since(ctx, watermark);
        connected.push(if ids.is_empty() { desc } else { format!("{} ({})", desc, ids.join(" ")) });
    }
    ctx.skip_dof_check = saved;
    connected
}

/// Pop the last tangent constraint matching the action type.
fn pop_tangent(sketch: &mut Sketch, action: &Action) {
    match action {
        Action::ApplyTangentLA { .. } => { sketch.tangent_la.pop(); }
        Action::ApplyTangentAA { .. } => { sketch.tangent_aa.pop(); }
        _ => {}
    }
}

/// Check if two direction vectors are nearly parallel (within ~1 degree).
fn nearly_tangent(d1: vect2d, d2: vect2d) -> bool {
    let len1 = (d1.x * d1.x + d1.y * d1.y).sqrt();
    let len2 = (d2.x * d2.x + d2.y * d2.y).sqrt();
    if len1 < 1e-12 || len2 < 1e-12 { return false; }
    let cross = (d1.x * d2.y - d1.y * d2.x).abs() / (len1 * len2);
    cross < 0.018 // sin(1 deg) ~ 0.01745
}

/// Try to apply auto-tangent constraints for a newly created line.
/// Checks if the line's endpoints are coincident with arc endpoints
/// and the geometry is already tangent.
fn auto_tangent_line(ctx: &mut CommandContext, line_ref: Ref<Line>) -> Vec<String> {
    let snap_threshold = 1e-3;
    let cost_threshold = 1e-6;
    let l = &ctx.sketch.lines[line_ref];
    let lp1 = l.p1.value;
    let lp2 = l.p2.value;
    let ld = vect2d::new(lp2.x - lp1.x, lp2.y - lp1.y);

    let mut candidates: Vec<(Action, String)> = Vec::new();

    for r in ctx.sketch.arcs.refs() {
        let a = &ctx.sketch.arcs[r];
        // Line p1 near arc start
        let sp = crate::geometry::arc_start_pos(a);
        if (lp1.x - sp.x).abs() < snap_threshold && (lp1.y - sp.y).abs() < snap_threshold {
            let at = crate::geometry::arc_tangent_at(a, a.start_angle.value);
            if nearly_tangent(ld, at) {
                candidates.push((Action::ApplyTangentLA { line: line_ref, arc: r },
                    format!("{}.tangent.{}", ctx.sketch.lines[line_ref].name, a.name)));
            }
        }
        // Line p1 near arc end
        let ep = crate::geometry::arc_end_pos(a);
        if (lp1.x - ep.x).abs() < snap_threshold && (lp1.y - ep.y).abs() < snap_threshold {
            let at = crate::geometry::arc_tangent_at(a, a.end_angle.value);
            if nearly_tangent(ld, at) {
                candidates.push((Action::ApplyTangentLA { line: line_ref, arc: r },
                    format!("{}.tangent.{}", ctx.sketch.lines[line_ref].name, a.name)));
            }
        }
        // Line p2 near arc start
        if (lp2.x - sp.x).abs() < snap_threshold && (lp2.y - sp.y).abs() < snap_threshold {
            let at = crate::geometry::arc_tangent_at(a, a.start_angle.value);
            if nearly_tangent(ld, at) {
                candidates.push((Action::ApplyTangentLA { line: line_ref, arc: r },
                    format!("{}.tangent.{}", ctx.sketch.lines[line_ref].name, a.name)));
            }
        }
        // Line p2 near arc end
        if (lp2.x - ep.x).abs() < snap_threshold && (lp2.y - ep.y).abs() < snap_threshold {
            let at = crate::geometry::arc_tangent_at(a, a.end_angle.value);
            if nearly_tangent(ld, at) {
                candidates.push((Action::ApplyTangentLA { line: line_ref, arc: r },
                    format!("{}.tangent.{}", ctx.sketch.lines[line_ref].name, a.name)));
            }
        }
    }

    // Stage 2: cost check — push constraint without solving, check cost, pop if bad
    let mut applied = Vec::new();
    for (action, desc) in candidates {
        let old_cost = ctx.sketch.current_cost();
        // Push constraint directly (no solve)
        action.apply_without_solve(ctx.sketch.get_mut());
        let new_cost = ctx.sketch.current_cost();
        // Pop the probe either way; an accepted candidate is
        // re-applied through exec so it lands in history (it used to
        // be invisible to undo).
        pop_tangent(ctx.sketch.get_mut(), &action);
        if new_cost <= old_cost + cost_threshold {
            let saved = ctx.skip_dof_check;
            ctx.skip_dof_check = true;
            let watermark = ctx.sketch.next_constraint_id;
            ctx.exec(action);
            ctx.skip_dof_check = saved;
            if ctx.status_error.take().is_none() {
                let ids = constraint_names_since(ctx, watermark);
                applied.push(if ids.is_empty() { desc } else { format!("{} ({})", desc, ids.join(" ")) });
            }
        }
    }
    applied
}

/// Try to apply auto-tangent constraints for a newly created arc.
/// Checks against lines and other arcs at shared endpoints.
fn auto_tangent_arc(ctx: &mut CommandContext, arc_ref: Ref<Arc>) -> Vec<String> {
    let snap_threshold = 1e-3;
    let cost_threshold = 1e-6;
    let a = &ctx.sketch.arcs[arc_ref];
    let a_sp = crate::geometry::arc_start_pos(a);
    let a_ep = crate::geometry::arc_end_pos(a);
    let a_st = crate::geometry::arc_tangent_at(a, a.start_angle.value);
    let a_et = crate::geometry::arc_tangent_at(a, a.end_angle.value);
    let a_name = a.name.clone();

    let mut candidates: Vec<(Action, String)> = Vec::new();

    // Against lines
    for r in ctx.sketch.lines.refs() {
        let l = &ctx.sketch.lines[r];
        let ld = vect2d::new(l.p2.value.x - l.p1.value.x, l.p2.value.y - l.p1.value.y);
        // Arc start near line p1 or p2
        if ((a_sp.x - l.p1.value.x).abs() < snap_threshold && (a_sp.y - l.p1.value.y).abs() < snap_threshold
            || (a_sp.x - l.p2.value.x).abs() < snap_threshold && (a_sp.y - l.p2.value.y).abs() < snap_threshold)
            && nearly_tangent(a_st, ld) {
                candidates.push((Action::ApplyTangentLA { line: r, arc: arc_ref },
                    format!("{}.tangent.{}", l.name, a_name)));
            }
        // Arc end near line p1 or p2
        if ((a_ep.x - l.p1.value.x).abs() < snap_threshold && (a_ep.y - l.p1.value.y).abs() < snap_threshold
            || (a_ep.x - l.p2.value.x).abs() < snap_threshold && (a_ep.y - l.p2.value.y).abs() < snap_threshold)
            && nearly_tangent(a_et, ld) {
                candidates.push((Action::ApplyTangentLA { line: r, arc: arc_ref },
                    format!("{}.tangent.{}", l.name, a_name)));
            }
    }

    // Against other arcs
    for r in ctx.sketch.arcs.refs() {
        if r == arc_ref { continue; }
        let b = &ctx.sketch.arcs[r];
        let b_sp = crate::geometry::arc_start_pos(b);
        let b_ep = crate::geometry::arc_end_pos(b);
        let b_st = crate::geometry::arc_tangent_at(b, b.start_angle.value);
        let b_et = crate::geometry::arc_tangent_at(b, b.end_angle.value);
        // Arc start near other arc start/end
        if (a_sp.x - b_sp.x).abs() < snap_threshold && (a_sp.y - b_sp.y).abs() < snap_threshold
            && nearly_tangent(a_st, b_st) {
                candidates.push((Action::ApplyTangentAA { a: arc_ref, b: r },
                    format!("{}.tangent.{}", a_name, b.name)));
            }
        if (a_sp.x - b_ep.x).abs() < snap_threshold && (a_sp.y - b_ep.y).abs() < snap_threshold
            && nearly_tangent(a_st, b_et) {
                candidates.push((Action::ApplyTangentAA { a: arc_ref, b: r },
                    format!("{}.tangent.{}", a_name, b.name)));
            }
        // Arc end near other arc start/end
        if (a_ep.x - b_sp.x).abs() < snap_threshold && (a_ep.y - b_sp.y).abs() < snap_threshold
            && nearly_tangent(a_et, b_st) {
                candidates.push((Action::ApplyTangentAA { a: arc_ref, b: r },
                    format!("{}.tangent.{}", a_name, b.name)));
            }
        if (a_ep.x - b_ep.x).abs() < snap_threshold && (a_ep.y - b_ep.y).abs() < snap_threshold
            && nearly_tangent(a_et, b_et) {
                candidates.push((Action::ApplyTangentAA { a: arc_ref, b: r },
                    format!("{}.tangent.{}", a_name, b.name)));
            }
    }

    // Stage 2: cost check
    let mut applied = Vec::new();
    for (action, desc) in candidates {
        let old_cost = ctx.sketch.current_cost();
        action.apply_without_solve(ctx.sketch.get_mut());
        let new_cost = ctx.sketch.current_cost();
        // See auto_tangent_line: probe popped, accepted candidates
        // re-applied through exec for history coverage.
        pop_tangent(ctx.sketch.get_mut(), &action);
        if new_cost <= old_cost + cost_threshold {
            let saved = ctx.skip_dof_check;
            ctx.skip_dof_check = true;
            let watermark = ctx.sketch.next_constraint_id;
            ctx.exec(action);
            ctx.skip_dof_check = saved;
            if ctx.status_error.take().is_none() {
                let ids = constraint_names_since(ctx, watermark);
                applied.push(if ids.is_empty() { desc } else { format!("{} ({})", desc, ids.join(" ")) });
            }
        }
    }
    applied
}

fn cmd_add_line(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, notangent, driven, quiet, constr] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "notangent", "driven", "quiet", "constr"]);

    // Parse all coordinate tokens
    let points: Vec<vect2d> = if tokens.len() >= 2 {
        let mut pts = Vec::new();
        let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) {
            Ok(p) => p, Err(e) => return err(e),
        };
        pts.push(p1);
        for i in 1..tokens.len() {
            let prev = *pts.last().unwrap();
            let p = match parse_coord(ctx, tokens[i], Some(prev)) {
                Ok(p) => p, Err(e) => return err(e),
            };
            pts.push(p);
        }
        pts
    } else if tokens.len() == 1 {
        let prev = match ctx.cursor {
            Some(p) => p,
            None => return err("No previous point. Use: add_line x1,y1 x2,y2"),
        };
        let p2 = match parse_coord(ctx, tokens[0], Some(prev)) {
            Ok(p) => p, Err(e) => return err(e),
        };
        vec![prev, p2]
    } else {
        return err("Usage: add_line x1,y1 x2,y2 [x3,y3 ...] [noconnect] [notangent] [nocursor] [driven]");
    };

    ctx.begin_group();
    let mut msgs = Vec::new();
    let n_segments = points.len() - 1;
    for i in 0..n_segments {
        let p1 = points[i];
        let p2 = points[i + 1];
        let Some(line_ref) = ctx.exec(Action::AddLine { p1, p2 }).line() else {
            return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
        };
        if quiet { ctx.exec(Action::SetQuietLine { line: line_ref, on: true }); }
        if constr { ctx.exec(Action::SetConstructionLine { line: line_ref, on: true }); }
        let name = ctx.sketch.lines[line_ref].name.clone();
        ctx.session_names.insert("_".into(), name.clone());
        // For multi-segment, also set _0, _1, _2, ... for multi-assignment
        if n_segments > 1 {
            ctx.session_names.insert(format!("_{}", i), name.clone());
        }
        let mut msg = format!("Added {}: ({:.2},{:.2})-({:.2},{:.2})", name, p1.x, p1.y, p2.x, p2.y);
        if !noconnect {
            let connected = auto_coincident_line(ctx, line_ref);
            if !connected.is_empty() {
                msg += &format!(" [connected: {}]", connected.join(", "));
            }
            if !notangent {
                let tangents = auto_tangent_line(ctx, line_ref);
                if !tangents.is_empty() {
                    msg += &format!(" [tangent: {}]", tangents.join(", "));
                }
            }
        }
        if driven {
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let len = (dx * dx + dy * dy).sqrt();
            msg += &driven_dim_fragment(ctx, Action::AddDimension {
                kind: DimensionKind::LineLength(line_ref),
                value: len, expr: None, derived: false, range: None,
            }, "length", len);
        }
        msgs.push(msg);
    }
    if !nocursor {
        ctx.cursor = Some(*points.last().unwrap());
        if points.len() >= 2 {
            let p1 = points[points.len() - 2];
            let p2 = points[points.len() - 1];
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 1e-12 {
                ctx.cursor_tangent = Some(vect2d::new(dx / len, dy / len));
            }
        }
    }
    ok(msgs.join("\n"))
}

/// Set cursor_tangent from an arc's end angle.
fn set_cursor_tangent_from_arc(ctx: &mut CommandContext, arc_ref: arael::refs::Ref<arael_sketch_solver::Arc>) {
    let a = &ctx.sketch.arcs[arc_ref];
    let t = crate::geometry::arc_tangent_at(a, a.end_angle.value);
    let len = (t.x * t.x + t.y * t.y).sqrt();
    if len > 1e-12 {
        ctx.cursor_tangent = Some(vect2d::new(t.x / len, t.y / len));
    }
}

/// Helper: execute a constraint/dimension action inside a compound command.
/// On success, pushes `desc` to `applied`. On failure in strict mode, returns Err.
/// In non-strict mode, collects warning and continues.
/// `C<n>` names of constraints minted at or after `watermark` (the
/// sketch's next_constraint_id captured before an exec).
fn constraint_names_since(ctx: &CommandContext, watermark: u32) -> Vec<String> {
    let mut nids: Vec<u32> = ctx.sketch.constraint_nid_cid_pairs().into_iter()
        .map(|(nid, _)| nid).filter(|&n| n >= watermark).collect();
    nids.sort_unstable();
    nids.dedup();
    nids.into_iter().map(|n| format!("C{}", n)).collect()
}

fn rect_exec(ctx: &mut CommandContext, action: Action, strict: bool, desc: &str, applied: &mut Vec<String>, warnings: &mut Vec<String>) -> Result<(), String> {
    // Surface the id of what the action minted: flag constraints have
    // synthetic names, dimensions their d-name, everything else a
    // watermarked C<n>.
    let flag_ids: Vec<String> = match &action {
        Action::ApplyHorizontal { lines } => lines.iter()
            .map(|r| arael_sketch_solver::format_flag_name(&ctx.sketch.lines[*r].name, 'H')).collect(),
        Action::ApplyVertical { lines } => lines.iter()
            .map(|r| arael_sketch_solver::format_flag_name(&ctx.sketch.lines[*r].name, 'V')).collect(),
        _ => Vec::new(),
    };
    let is_dim = matches!(action, Action::AddDimension { .. });
    let watermark = ctx.sketch.next_constraint_id;
    ctx.exec(action);
    if let Some(e) = ctx.status_error.take() {
        if strict {
            return Err(e);
        }
        warnings.push(e);
    } else {
        let mut ids = flag_ids;
        if is_dim {
            // The backing constraint is addressed through the
            // dimension, so only the d-name is surfaced.
            ids.push(last_dim_name(ctx));
        } else {
            ids.extend(constraint_names_since(ctx, watermark));
        }
        if ids.is_empty() {
            applied.push(desc.to_string());
        } else {
            applied.push(format!("{} ({})", desc, ids.join(" ")));
        }
    }
    Ok(())
}

/// Shared logic: create 4 lines from corners, apply constraints and optional driven dims.
/// corners: [bl, br, tr, tl] (or any 4-corner cycle).
fn build_rect(
    ctx: &mut CommandContext,
    corners: [vect2d; 4],
    noconnect: bool,
    noconstraint: bool,
    hv: bool,
    driven: bool,
    strict: bool,
) -> CommandResult {
    ctx.begin_group();
    let mut warnings = Vec::new();
    let mut applied = Vec::new();

    // Create 4 lines: 0-1, 1-2, 2-3, 3-0
    let mut line_refs = Vec::new();
    let mut line_names = Vec::new();
    for i in 0..4 {
        let p1 = corners[i];
        let p2 = corners[(i + 1) % 4];
        let Some(r) = ctx.exec(Action::AddLine { p1, p2 }).line() else {
            return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
        };
        let name = ctx.sketch.lines[r].name.clone();
        if !noconnect {
            auto_coincident_line(ctx, r);
        }
        line_refs.push(r);
        line_names.push(name);
    }

    if !noconstraint {
        if hv {
            let desc = format!("horizontal {} {}", line_names[0], line_names[2]);
            if let Err(e) = rect_exec(ctx, Action::ApplyHorizontal { lines: vec![line_refs[0], line_refs[2]] }, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
            let desc = format!("vertical {} {}", line_names[1], line_names[3]);
            if let Err(e) = rect_exec(ctx, Action::ApplyVertical { lines: vec![line_refs[1], line_refs[3]] }, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
        } else {
            let desc = format!("perpendicular {} {}", line_names[0], line_names[1]);
            if let Err(e) = rect_exec(ctx, Action::ApplyPerpendicular { a: line_refs[0], b: line_refs[1] }, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
            let desc = format!("parallel {} {}", line_names[0], line_names[2]);
            if let Err(e) = rect_exec(ctx, Action::ApplyParallel { a: line_refs[0], b: line_refs[2] }, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
            let desc = format!("parallel {} {}", line_names[1], line_names[3]);
            if let Err(e) = rect_exec(ctx, Action::ApplyParallel { a: line_refs[1], b: line_refs[3] }, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
        }
    }

    if driven {
        for &i in &[0, 1] {
            let l = &ctx.sketch.lines[line_refs[i]];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            let kind = DimensionKind::LineLength(line_refs[i]);
            let desc = format!("driven length {} = {:.4}", line_names[i], len);
            if let Err(e) = rect_exec(ctx, Action::AddDimension { kind, value: len, expr: None, derived: false, range: None,  }, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
        }
    }

    ctx.cursor = Some(corners[2]);
    ctx.session_names.insert("_".into(), line_names[0].clone());
    for (i, name) in line_names.iter().enumerate() {
        ctx.session_names.insert(format!("_{}", i), name.clone());
    }

    let mut msg = format!("Added rect: {} {} {} {}", line_names[0], line_names[1], line_names[2], line_names[3]);
    for a in &applied {
        msg += &format!("\n  {}", a);
    }
    for w in &warnings {
        msg += &format!("\n  warning: {}", w);
    }
    ok(msg)
}

struct RectKeywords {
    noconnect: bool,
    noconstraint: bool,
    hv: bool,
    driven: bool,
    strict: bool,
}

/// Parse trailing rect keywords. Returns error string on conflict.
fn parse_rect_keywords(tokens: &mut Vec<&str>, allow_hv: bool) -> Result<RectKeywords, String> {
    let [noconnect, noconstraint, hv, driven, strict] = peel_keywords(tokens,
        ["noconnect", "noconstraint", "hv", "driven", "strict"]);
    let kw = RectKeywords { noconnect, noconstraint, hv, driven, strict };
    if kw.hv && !allow_hv {
        return Err("hv keyword is not supported for this command".into());
    }
    if kw.noconstraint && (kw.hv || kw.driven || kw.strict) {
        return Err("noconstraint conflicts with hv, driven, and strict".into());
    }
    Ok(kw)
}

fn cmd_add_rect(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let kw = match parse_rect_keywords(&mut tokens, true) { Ok(k) => k, Err(e) => return err(e) };
    if tokens.len() != 2 {
        return err("Usage: add_rect x1,y1 x2,y2 [hv] [noconnect] [noconstraint] [driven] [strict]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let bl = p1;
    let br = vect2d::new(p2.x, p1.y);
    let tr = p2;
    let tl = vect2d::new(p1.x, p2.y);
    build_rect(ctx, [bl, br, tr, tl], kw.noconnect, kw.noconstraint, kw.hv, kw.driven, kw.strict)
}

fn cmd_add_rect3(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let kw = match parse_rect_keywords(&mut tokens, false) { Ok(k) => k, Err(e) => return err(e) };
    if tokens.len() != 3 {
        return err("Usage: add_rect3 p1 p2 p3 [noconnect] [noconstraint] [driven] [strict]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let p3 = match parse_coord(ctx, tokens[2], Some(p2)) { Ok(p) => p, Err(e) => return err(e) };
    // Reject collinear points (cross product of p1->p2 and p2->p3 ~ 0)
    let cross = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);
    if cross.abs() < 1e-9 {
        return err("Points are collinear, cannot form a rectangle");
    }
    // p4 = p1 + (p3 - p2)
    let p4 = vect2d::new(p1.x + (p3.x - p2.x), p1.y + (p3.y - p2.y));
    build_rect(ctx, [p1, p2, p3, p4], kw.noconnect, kw.noconstraint, kw.hv, kw.driven, kw.strict)
}

fn cmd_add_rectcenter(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let kw = match parse_rect_keywords(&mut tokens, true) { Ok(k) => k, Err(e) => return err(e) };
    if tokens.len() != 2 {
        return err("Usage: add_rectcenter cx,cy px,py [hv] [noconnect] [noconstraint] [driven] [strict]");
    }
    let center = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let corner = match parse_coord(ctx, tokens[1], Some(center)) { Ok(p) => p, Err(e) => return err(e) };
    // Axis-aligned: bl=corner, tr=opposite corner (reflected through center)
    let bl = corner;
    let tr = vect2d::new(2.0 * center.x - corner.x, 2.0 * center.y - corner.y);
    let br = vect2d::new(tr.x, bl.y);
    let tl = vect2d::new(bl.x, tr.y);
    build_rect(ctx, [bl, br, tr, tl], kw.noconnect, kw.noconstraint, kw.hv, kw.driven, kw.strict)
}

fn cmd_add_point(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let nocursor = tokens.last() == Some(&"nocursor");
    if nocursor { tokens.pop(); }
    if tokens.len() != 1 { return err("Usage: add_point x,y [nocursor]"); }
    let pos = match parse_coord(ctx, tokens[0], ctx.cursor) {
        Ok(p) => p, Err(e) => return err(e),
    };
    ctx.begin_group();
    ctx.exec(Action::AddPoint { pos });
    let name = ctx.sketch.points.refs().last().map(|r| ctx.sketch.points[r].name.clone()).unwrap_or_default();
    if !nocursor { ctx.cursor = Some(pos); }
    ctx.session_names.insert("_".into(), name.clone());
    ok(format!("Added {}: ({:.2},{:.2})", name, pos.x, pos.y))
}

fn cmd_add_circle(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 2 {
        return err("Usage: add_circle cx,cy radius [noconnect] [nocursor] [driven]");
    }
    let center = match parse_coord(ctx, tokens[0], ctx.cursor) {
        Ok(p) => p, Err(e) => return err(e),
    };
    let r = match eval_expr(&ctx.sketch, tokens[1]) {
        Ok(v) => v, Err(e) => return err(e),
    };
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(center); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) r={:.2}", name, center.x, center.y, r);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, "radius", r);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_circle2(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 2 {
        return err("Usage: add_circle2 p1 p2 [noconnect] [nocursor] [driven]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let center = vect2d::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
    let r = ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt() / 2.0;
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(center); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) r={:.2}", name, center.x, center.y, r);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, "radius", r);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_circle3(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 3 {
        return err("Usage: add_circle3 p1 p2 p3 [noconnect] [nocursor] [driven]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let p3 = match parse_coord(ctx, tokens[2], Some(p2)) { Ok(p) => p, Err(e) => return err(e) };
    let (center, r, _, _, _) = match crate::geometry::circumscribed_arc(p1, p2, p3) {
        Some(v) => v,
        None => return err("Points are collinear, cannot define a circle"),
    };
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(center); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) r={:.2}", name, center.x, center.y, r);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, "radius", r);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_ellipse(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 4 {
        return err("Usage: add_ellipse cx,cy rx ry rotation [noconnect] [nocursor] [driven]");
    }
    let center = match parse_coord(ctx, tokens[0], ctx.cursor) {
        Ok(p) => p, Err(e) => return err(e),
    };
    let rx = match eval_expr(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let ry = match eval_expr(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };
    let rot = match eval_expr(&ctx.sketch, tokens[3]) { Ok(v) => v, Err(e) => return err(e) };
    let rot_rad = arael::utils::deg2rad(rot);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipse { center, rx, ry, rotation: rot_rad }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(center); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) rx={:.2} ry={:.2} rot={:.2}deg",
        name, center.x, center.y, rx, ry, rot);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: rx, expr: None, derived: false, range: None,
        }, "rx", rx);
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadiusB(arc_ref),
            value: ry, expr: None, derived: false, range: None,
        }, "ry", ry);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_earc(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven, large, cw] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven", "large", "cw"]);
    if tokens.len() != 5 {
        return err("Usage: add_earc p1 p2 rx ry rot_deg [large] [cw] [noconnect] [notangent] [nocursor] [driven]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let rx = match eval_expr(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };
    let ry = match eval_expr(&ctx.sketch, tokens[3]) { Ok(v) => v, Err(e) => return err(e) };
    let rot_deg = match eval_expr(&ctx.sketch, tokens[4]) { Ok(v) => v, Err(e) => return err(e) };
    let rot = arael::utils::deg2rad(rot_deg);
    let sweep = !cw; // SVG sweep-flag: true = CCW
    let result = crate::geometry::svg_arc_to_center(p1, p2, rx, ry, rot, large, sweep);
    let (center, sa, ea, rx, ry) = match result {
        Some(v) => v,
        None => return err("Cannot compute elliptic arc (degenerate or zero radii)"),
    };
    let ccw = !cw;
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start: sa, end: ea, ccw }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor {
        ctx.cursor = Some(p2);
        set_cursor_tangent_from_arc(ctx, arc_ref);
    }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: rx={:.2} ry={:.2} rot={:.2}deg", name, rx, ry, rot_deg);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, false);
        if !connected.is_empty() { msg += &format!(" [connected: {}]", connected.join(", ")); }
        if !notangent {
            let tangents = auto_tangent_arc(ctx, arc_ref);
            if !tangents.is_empty() { msg += &format!(" [tangent: {}]", tangents.join(", ")); }
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: rx, expr: None, derived: false, range: None,
        }, "rx", rx);
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadiusB(arc_ref),
            value: ry, expr: None, derived: false, range: None,
        }, "ry", ry);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_earc3(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven"]);
    if tokens.len() != 5 {
        return err("Usage: add_earc3 p1 p2 pmid rx ry [noconnect] [notangent] [nocursor] [driven]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let pmid = match parse_coord(ctx, tokens[2], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let rx = match eval_expr(&ctx.sketch, tokens[3]) { Ok(v) => v, Err(e) => return err(e) };
    let ry = match eval_expr(&ctx.sketch, tokens[4]) { Ok(v) => v, Err(e) => return err(e) };
    // Estimate rotation from midpoint geometry
    let rot = (p2.y - p1.y).atan2(p2.x - p1.x);
    // Determine ccw from midpoint: same logic as circumscribed_arc
    // Use SVG conversion with rotation=estimated, find which arc passes near pmid
    let mut best = None;
    let mut best_dist = f64::MAX;
    for &large in &[false, true] {
        for &sweep in &[true, false] {
            if let Some((center, sa, ea, rx_out, ry_out)) =
                crate::geometry::svg_arc_to_center(p1, p2, rx, ry, rot, large, sweep)
            {
                // Check how close the midpoint of this arc is to pmid
                let mid_angle = (sa + ea) / 2.0;
                let cr = rot.cos();
                let sr = rot.sin();
                let ct = mid_angle.cos();
                let st = mid_angle.sin();
                let mid_pt = vect2d::new(
                    center.x + rx_out * ct * cr - ry_out * st * sr,
                    center.y + rx_out * ct * sr + ry_out * st * cr,
                );
                let dist = ((mid_pt.x - pmid.x).powi(2) + (mid_pt.y - pmid.y).powi(2)).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best = Some((center, sa, ea, rx_out, ry_out, sweep));
                }
            }
        }
    }
    let (center, sa, ea, rx, ry, ccw) = match best {
        Some((c, sa, ea, rx, ry, sweep)) => (c, sa, ea, rx, ry, sweep),
        None => return err("Cannot compute elliptic arc from given points and radii"),
    };
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start: sa, end: ea, ccw }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor {
        ctx.cursor = Some(p2);
        set_cursor_tangent_from_arc(ctx, arc_ref);
    }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: rx={:.2} ry={:.2}", name, rx, ry);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, false);
        if !connected.is_empty() { msg += &format!(" [connected: {}]", connected.join(", ")); }
        if !notangent {
            let tangents = auto_tangent_arc(ctx, arc_ref);
            if !tangents.is_empty() { msg += &format!(" [tangent: {}]", tangents.join(", ")); }
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: rx, expr: None, derived: false, range: None,
        }, "rx", rx);
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadiusB(arc_ref),
            value: ry, expr: None, derived: false, range: None,
        }, "ry", ry);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_earc_center(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven, cw] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven", "cw"]);
    if tokens.len() != 6 {
        return err("Usage: add_earc_center cx,cy rx ry rot_deg start_deg end_deg [cw] [noconnect] [notangent] [nocursor] [driven]");
    }
    let center = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let rx = match eval_expr(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let ry = match eval_expr(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };
    let rot_deg = match eval_expr(&ctx.sketch, tokens[3]) { Ok(v) => v, Err(e) => return err(e) };
    let start_deg = match eval_expr(&ctx.sketch, tokens[4]) { Ok(v) => v, Err(e) => return err(e) };
    let end_deg = match eval_expr(&ctx.sketch, tokens[5]) { Ok(v) => v, Err(e) => return err(e) };
    let rot = arael::utils::deg2rad(rot_deg);
    let start = arael::utils::deg2rad(start_deg);
    let end = arael::utils::deg2rad(end_deg);
    let ccw = !cw;
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start, end, ccw }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(center); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: rx={:.2} ry={:.2} rot={:.2}deg start={:.2}deg end={:.2}deg",
        name, rx, ry, rot_deg, start_deg, end_deg);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() { msg += &format!(" [connected: {}]", connected.join(", ")); }
        if !notangent {
            let tangents = auto_tangent_arc(ctx, arc_ref);
            if !tangents.is_empty() { msg += &format!(" [tangent: {}]", tangents.join(", ")); }
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: rx, expr: None, derived: false, range: None,
        }, "rx", rx);
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadiusB(arc_ref),
            value: ry, expr: None, derived: false, range: None,
        }, "ry", ry);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_earc_tangent(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, notangent, driven, quiet, constr] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "notangent", "driven", "quiet", "constr"]);
    // Syntax: add_earc_tangent p1 t1 p2 t2 [w]
    if tokens.len() < 4 || tokens.len() > 5 {
        return err("Usage: add_earc_tangent p1 t1 p2 t2 [w] [noconnect] [notangent] [nocursor] [quiet] [driven]");
    }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let t1 = match parse_coord(ctx, tokens[1], None) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[2], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let t2 = match parse_coord(ctx, tokens[3], None) { Ok(p) => p, Err(e) => return err(e) };
    let w = if tokens.len() == 5 {
        match eval_expr(&ctx.sketch, tokens[4]) { Ok(v) => v, Err(e) => return err(e) }
    } else { 1.0 };

    let result = crate::earc_fit::fit_earc_tangent(p1, t1, p2, t2, w);
    let (center, rx, ry, rot, sa, ea, ccw) = match result {
        Some(v) => v,
        None => return err("Cannot fit elliptic arc (degenerate tangent configuration)"),
    };
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start: sa, end: ea, ccw }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor {
        ctx.cursor = Some(p2);
        set_cursor_tangent_from_arc(ctx, arc_ref);
    }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: rx={:.2} ry={:.2} bulge={:.2}", name, rx, ry, w);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, false);
        if !connected.is_empty() { msg += &format!(" [connected: {}]", connected.join(", ")); }
        if !notangent {
            let tangents = auto_tangent_arc(ctx, arc_ref);
            if !tangents.is_empty() { msg += &format!(" [tangent: {}]", tangents.join(", ")); }
        }
    }
    if driven {
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: rx, expr: None, derived: false, range: None,
        }, "rx", rx);
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadiusB(arc_ref),
            value: ry, expr: None, derived: false, range: None,
        }, "ry", ry);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

fn cmd_add_earc_rtangent(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let p1 = match ctx.cursor {
        Some(p) => p,
        None => return err("No cursor position (add a line or arc first)"),
    };
    let t1 = match ctx.cursor_tangent {
        Some(t) => t,
        None => return err("No tangent direction (add a line or arc first)"),
    };
    // Prepend p1 and t1 as coordinate strings and delegate
    let p1_str = format!("{},{}", p1.x, p1.y);
    let t1_str = format!("{},{}", t1.x, t1.y);
    let new_args = format!("{} {} {}", p1_str, t1_str, args);
    cmd_add_earc_tangent(ctx, &new_args)
}

/// Check if the perpendicular projection of `center` onto the line segment (p1,p2)
/// falls within the segment (0 <= t <= 1).
fn tangent_touches_segment(center: vect2d, p1: vect2d, p2: vect2d) -> bool {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-24 { return false; }
    let t = ((center.x - p1.x) * dx + (center.y - p1.y) * dy) / len_sq;
    (-1e-6..=1.0 + 1e-6).contains(&t)
}

/// Compute center of circle tangent to two line segments with given radius.
/// Only considers candidates whose tangent points fall on the actual segments.
/// Returns Ok(center) if exactly one candidate, Err with message otherwise.
fn circle_tangent_2lines(sketch: &Sketch, la: Ref<Line>, lb: Ref<Line>, radius: f64) -> Result<vect2d, String> {
    let a = &sketch.lines[la];
    let b = &sketch.lines[lb];
    let a1 = a.p1.value; let a2 = a.p2.value;
    let b1 = b.p1.value; let b2 = b.p2.value;
    let da = vect2d::new(a2.x - a1.x, a2.y - a1.y);
    let db = vect2d::new(b2.x - b1.x, b2.y - b1.y);
    let la_len = (da.x * da.x + da.y * da.y).sqrt();
    let lb_len = (db.x * db.x + db.y * db.y).sqrt();
    if la_len < 1e-12 || lb_len < 1e-12 {
        return Err("Degenerate line (zero length)".into());
    }
    let na = vect2d::new(-da.y / la_len, da.x / la_len);
    let nb = vect2d::new(-db.y / lb_len, db.x / lb_len);
    // Try all 4 offset combinations (+/- normal for each line)
    let mut candidates = Vec::new();
    for &sa in &[1.0_f64, -1.0] {
        for &sb in &[1.0_f64, -1.0] {
            let oa1 = vect2d::new(a1.x + sa * radius * na.x, a1.y + sa * radius * na.y);
            let ob1 = vect2d::new(b1.x + sb * radius * nb.x, b1.y + sb * radius * nb.y);
            if let Some(c) = line_line_intersect(oa1, da, ob1, db) {
                // Check if tangent points fall on actual segments
                if tangent_touches_segment(c, a1, a2) && tangent_touches_segment(c, b1, b2) {
                    // Deduplicate (parallel offsets can give same point)
                    if !candidates.iter().any(|p: &vect2d| (p.x - c.x).abs() < 1e-6 && (p.y - c.y).abs() < 1e-6) {
                        candidates.push(c);
                    }
                }
            }
        }
    }
    match candidates.len() {
        0 => Err("No tangent circle touches both line segments at this radius".into()),
        1 => Ok(candidates[0]),
        n => Err(format!("Ambiguous: {} possible placements, extend or shorten lines to disambiguate", n)),
    }
}

/// Compute circle tangent to 3 line segments (incircle or excircle).
/// Tries the incircle and 3 excircles, keeps only those whose tangent points
/// all fall on the actual segments. Returns exactly 1 or errors.
fn circle_tangent_3lines(sketch: &Sketch, la: Ref<Line>, lb: Ref<Line>, lc: Ref<Line>) -> Result<(vect2d, f64), String> {
    let a = &sketch.lines[la];
    let b = &sketch.lines[lb];
    let c = &sketch.lines[lc];
    let a1 = a.p1.value; let a2 = a.p2.value;
    let b1 = b.p1.value; let b2 = b.p2.value;
    let c1 = c.p1.value; let c2 = c.p2.value;
    let da = vect2d::new(a2.x - a1.x, a2.y - a1.y);
    let db = vect2d::new(b2.x - b1.x, b2.y - b1.y);
    let dc = vect2d::new(c2.x - c1.x, c2.y - c1.y);
    let la_len = (da.x * da.x + da.y * da.y).sqrt();
    let lb_len = (db.x * db.x + db.y * db.y).sqrt();
    let lc_len = (dc.x * dc.x + dc.y * dc.y).sqrt();
    if la_len < 1e-12 || lb_len < 1e-12 || lc_len < 1e-12 {
        return Err("Degenerate line (zero length)".into());
    }
    let na = vect2d::new(-da.y / la_len, da.x / la_len);
    let nb = vect2d::new(-db.y / lb_len, db.x / lb_len);
    let nc = vect2d::new(-dc.y / lc_len, dc.x / lc_len);
    // Try all 8 sign combinations for offset directions (incircle + 3 excircles + extras)
    let mut candidates = Vec::new();
    for &sa in &[1.0_f64, -1.0] {
        for &sb in &[1.0_f64, -1.0] {
            for &sc in &[1.0_f64, -1.0] {
                // Offset each line by sign * r * normal, where r is unknown.
                // The 3 offset lines must intersect at one point (the center).
                // Intersect offset-A and offset-B to get center, then check offset-C passes through it.
                // Since r is unknown, we solve: distance from center to each line = same value.
                // Use two pairs of lines to find center, then compute r.
                // Equidistance: sa*(na.(p-a1)) = sb*(nb.(p-b1))
                let eq_ab_x = sa * na.x - sb * nb.x;
                let eq_ab_y = sa * na.y - sb * nb.y;
                let eq_ab_c = sa * (na.x * a1.x + na.y * a1.y) - sb * (nb.x * b1.x + nb.y * b1.y);
                let eq_bc_x = sb * nb.x - sc * nc.x;
                let eq_bc_y = sb * nb.y - sc * nc.y;
                let eq_bc_c = sb * (nb.x * b1.x + nb.y * b1.y) - sc * (nc.x * c1.x + nc.y * c1.y);
                let det = eq_ab_x * eq_bc_y - eq_ab_y * eq_bc_x;
                if det.abs() < 1e-12 { continue; }
                let cx = (eq_ab_c * eq_bc_y - eq_ab_y * eq_bc_c) / det;
                let cy = (eq_ab_x * eq_bc_c - eq_ab_c * eq_bc_x) / det;
                let center = vect2d::new(cx, cy);
                let r = ((center.x - a1.x) * na.x + (center.y - a1.y) * na.y).abs();
                if r < 1e-12 { continue; }
                // Check tangent points fall on all 3 segments
                if tangent_touches_segment(center, a1, a2)
                    && tangent_touches_segment(center, b1, b2)
                    && tangent_touches_segment(center, c1, c2)
                    && !candidates.iter().any(|(p, _): &(vect2d, f64)|
                        (p.x - cx).abs() < 1e-6 && (p.y - cy).abs() < 1e-6)
                    {
                        candidates.push((center, r));
                    }
            }
        }
    }
    match candidates.len() {
        0 => Err("No tangent circle touches all 3 line segments".into()),
        1 => Ok(candidates[0]),
        n => Err(format!("Ambiguous: {} possible placements, extend or shorten lines to disambiguate", n)),
    }
}

/// Intersect two lines given as point+direction. Returns None if parallel.
fn line_line_intersect(p1: vect2d, d1: vect2d, p2: vect2d, d2: vect2d) -> Option<vect2d> {
    let cross = d1.x * d2.y - d1.y * d2.x;
    if cross.abs() < 1e-12 { return None; }
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let t = (dx * d2.y - dy * d2.x) / cross;
    Some(vect2d::new(p1.x + t * d1.x, p1.y + t * d1.y))
}

fn cmd_add_circle2t(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [noconnect, quiet, constr, noconstraint, driven, strict] = peel_keywords(&mut tokens,
        ["noconnect", "quiet", "constr", "noconstraint", "driven", "strict"]);
    if noconstraint && (driven || strict) {
        return err("noconstraint conflicts with driven and strict");
    }
    if tokens.len() != 3 {
        return err("Usage: add_circle2t L0 L1 radius [noconnect] [noconstraint] [driven] [strict]");
    }
    let la = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let lb = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let r = match eval_expr(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };
    let center = match circle_tangent_2lines(&ctx.sketch, la, lb, r) {
        Ok(c) => c,
        Err(e) => return err(e),
    };
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let mut warnings = Vec::new();
    let mut applied = Vec::new();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) r={:.2}", name, center.x, center.y, r);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    if !noconstraint {
        let desc = format!("tangent {} {}", tokens[0], name);
        if let Err(e) = rect_exec(ctx, Action::ApplyTangentLA { line: la, arc: arc_ref }, strict, &desc, &mut applied, &mut warnings) {
            return err(e);
        }
        let desc = format!("tangent {} {}", tokens[1], name);
        if let Err(e) = rect_exec(ctx, Action::ApplyTangentLA { line: lb, arc: arc_ref }, strict, &desc, &mut applied, &mut warnings) {
            return err(e);
        }
    }
    if driven {
        let desc = format!("driven radius {} = {:.4}", name, r);
        if let Err(e) = rect_exec(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, strict, &desc, &mut applied, &mut warnings) {
            return err(e);
        }
    }
    for a in &applied {
        msg += &format!("\n  {}", a);
    }
    for w in &warnings {
        msg += &format!("\n  warning: {}", w);
    }
    ok(msg)
}

fn cmd_add_circle3t(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [noconnect, quiet, constr, noconstraint, driven, strict] = peel_keywords(&mut tokens,
        ["noconnect", "quiet", "constr", "noconstraint", "driven", "strict"]);
    if noconstraint && (driven || strict) {
        return err("noconstraint conflicts with driven and strict");
    }
    if tokens.len() != 3 {
        return err("Usage: add_circle3t L0 L1 L2 [noconnect] [noconstraint] [driven] [strict]");
    }
    let la = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let lb = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let lc = match resolve_line(&ctx.sketch, tokens[2]) { Ok(r) => r, Err(e) => return err(e) };
    let (center, r) = match circle_tangent_3lines(&ctx.sketch, la, lb, lc) {
        Ok(v) => v,
        Err(e) => return err(e),
    };
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let mut warnings = Vec::new();
    let mut applied_list = Vec::new();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) r={:.2}", name, center.x, center.y, r);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    if !noconstraint {
        let line_names = [tokens[0], tokens[1], tokens[2]];
        for (&line, ln) in [la, lb, lc].iter().zip(line_names.iter()) {
            let desc = format!("tangent {} {}", ln, name);
            if let Err(e) = rect_exec(ctx, Action::ApplyTangentLA { line, arc: arc_ref }, strict, &desc, &mut applied_list, &mut warnings) {
                return err(e);
            }
        }
    }
    if driven {
        let desc = format!("driven radius {} = {:.4}", name, r);
        if let Err(e) = rect_exec(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, strict, &desc, &mut applied_list, &mut warnings) {
            return err(e);
        }
    }
    for a in &applied_list {
        msg += &format!("\n  {}", a);
    }
    for w in &warnings {
        msg += &format!("\n  warning: {}", w);
    }
    ok(msg)
}

/// Unified `delete` command: removes entities (L0, P0, A0, EA0),
/// constraints (C3, CL0H, CL0V, or the relational form
/// `L0 L1 parallel`), and dimensions (d0, d1, ...). This is the
/// single removal command; the old `remove_constraint` (alias `rc`)
/// and `remove_dim` were merged into it.
///
/// When an entity is deleted, the response message also lists which
/// constraints and dimensions got cascade-removed alongside it
/// (e.g. deleting L0 removes its horizontal flag, any coincidences
/// touching L0.p1 / L0.p2, any length dimension on L0, etc.). The
/// report is computed by diffing `list_constraints()` and
/// `dimensions` before vs after the delete, so it catches every
/// cascade path the solver walks internally.
fn cmd_delete(ctx: &mut CommandContext, args: &str) -> CommandResult {
    use crate::ids::find_constraint_by_name;
    let cleaned = args.trim();
    if cleaned.is_empty() {
        return err("Usage: delete L0 | delete P0 | delete A0 | delete C3 | delete CL0H | delete d0 | delete L0 L1 parallel");
    }
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();

    // Multi-token relational form: `delete L0 L1 parallel`.
    if tokens.len() > 1 {
        return delete_relational(ctx, cleaned);
    }

    let name = tokens[0];

    // 1) Named constraint: C<n>, CL0H, CL0V.
    if let Some(id) = find_constraint_by_name(&ctx.sketch, name) {
        let prefix = format!("{}: ", name);
        let desc = ctx.sketch.list_constraints().into_iter()
            .find(|l| l.starts_with(&prefix))
            .unwrap_or_else(|| name.to_string());
        ctx.begin_group();
        ctx.exec(Action::DeleteConstraint { id });
        return ok(format!("Deleted {}", desc));
    }

    // 2) Dimension: d<n>.
    if let Some(idx) = ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        ctx.begin_group();
        ctx.exec(Action::RemoveDimension { did: ctx.sketch.dimensions[idx].did });
        return ok(format!("Deleted dimension {}", name));
    }

    // 3) Entity: L0, P0, A0, EA0.
    let action = if name.starts_with('L') && !name.starts_with("LineAngle") {
        // Resolve as line.
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => Some(("line", Action::DeleteLine { line: r })),
            Err(e) => return err(e),
        }
    } else if name.starts_with('P') {
        match resolve_point(&ctx.sketch, name) {
            Ok(r) => Some(("point", Action::DeletePoint { point: r })),
            Err(e) => return err(e),
        }
    } else if is_arc_name(name) {
        match resolve_arc(&ctx.sketch, name) {
            Ok(r) => Some(("arc", Action::DeleteArc { arc: r })),
            Err(e) => return err(e),
        }
    } else {
        None
    };

    let Some((kind, action)) = action else {
        return err(format!("Unknown name: {}", name));
    };

    // Snapshot both views of the current state so we can report
    // what got cascade-removed. list_constraints() covers flag,
    // lock, and C<n>-named constraints; a per-dim description pass
    // covers d<n>-named dimensions. When a dim has a backing entity
    // flag (LineLength -> "length L0 = V", ArcSweep -> "sweep A0 =
    // V deg", etc.), its list_constraints phrase duplicates the
    // dim: the next pair of snapshots captures each separately so
    // we can de-dupe them in the output.
    let before_constraints: std::collections::BTreeSet<String> =
        ctx.sketch.list_constraints().into_iter().collect();
    let before_dim_phrases: Vec<(String, String)> = ctx.sketch.dimensions.iter()
        .map(|d| (d.name.clone(), dim_phrase(&ctx.sketch, d)))
        .collect();

    ctx.begin_group();
    ctx.exec(action);

    let after_constraints: std::collections::BTreeSet<String> =
        ctx.sketch.list_constraints().into_iter().collect();
    let after_dim_names: std::collections::BTreeSet<String> =
        ctx.sketch.dimensions.iter().map(|d| d.name.clone()).collect();

    let mut removed_constraints: Vec<String> = before_constraints
        .difference(&after_constraints).cloned().collect();
    let removed_dims: Vec<(String, String)> = before_dim_phrases.iter()
        .filter(|(n, _)| !after_dim_names.contains(n))
        .cloned().collect();

    // Any list_constraints phrase that exactly duplicates a removed
    // dim's backing description gets dropped so it isn't shown
    // twice. The remaining entries are real constraints (C<n>,
    // CL<n>H/V, lock, horizontal/vertical flags not backed by a dim).
    let dup: std::collections::HashSet<&str> = removed_dims.iter()
        .map(|(_, phrase)| phrase.as_str())
        .collect();
    removed_constraints.retain(|c| !dup.contains(c.as_str()));

    let mut msg = format!("Deleted {} {}", kind, name);
    if !removed_constraints.is_empty() || !removed_dims.is_empty() {
        msg.push_str("\n  cascade:");
        for c in &removed_constraints {
            msg.push_str(&format!("\n    {}", c));
        }
        for (dname, phrase) in &removed_dims {
            if phrase.is_empty() {
                msg.push_str(&format!("\n    {}", dname));
            } else {
                msg.push_str(&format!("\n    {}: {}", dname, phrase));
            }
        }
    }
    ok(msg)
}

/// Human-readable backing description for a dimension, matching the
/// phrase `list_constraints()` would emit for its entity-flag or
/// scalar constraint. Used by the delete-cascade reporter so each
/// removed dim is shown as `d<n>: <what it was>` and any identical
/// phrase in the list_constraints diff is suppressed. Returns an
/// empty string for dimensions that don't have a canonical phrase
/// (range-only dims without a matching kind string, for example).
fn dim_phrase(sketch: &Sketch, dim: &Dimension) -> String {
    use arael_sketch_solver::DimensionKind as K;
    use arael_sketch_solver::DimensionEndpoint as E;
    let ep_name = |e: &E| -> String {
        match e {
            E::Point(r) => sketch.point_display_name(*r),
            E::LineP1(r) => format!("{}.p1", sketch.lines[*r].name),
            E::LineP2(r) => format!("{}.p2", sketch.lines[*r].name),
            E::ArcCenter(r) => format!("{}.center", sketch.arcs[*r].name),
            E::ArcStart(r) => format!("{}.start", sketch.arcs[*r].name),
            E::ArcEnd(r) => format!("{}.end", sketch.arcs[*r].name),
        }
    };
    match dim.kind {
        // Entity-flag-backed: match the exact `list_constraints` format
        // so the de-dupe string compare succeeds.
        K::LineLength(r) => {
            let l = &sketch.lines[r];
            format!("length {} = {}", l.name, l.constraints.length)
        }
        K::LineAngle(r) => {
            let l = &sketch.lines[r];
            format!("xangle {} = {:.4}", l.name, l.constraints.target_angle.to_degrees())
        }
        K::ArcRadius(r) => {
            let a = &sketch.arcs[r];
            format!("radius {} = {}", a.name, a.constraints.target_radius)
        }
        K::ArcRadiusB(r) => {
            let a = &sketch.arcs[r];
            format!("radius_b {} = {}", a.name, a.constraints.target_radius_b)
        }
        K::ArcSweep(r) => {
            let a = &sketch.arcs[r];
            format!("sweep {} = {:.2} deg", a.name, a.constraints.target_sweep.to_degrees())
        }
        K::ArcRotation(r) => {
            let a = &sketch.arcs[r];
            format!("rotation {} = {:.4}", a.name, a.constraints.target_rotation.to_degrees())
        }
        // Collection-backed: phrase the info from the dim itself
        // (the C<n> form appears separately in list_constraints).
        K::PointPointDistance(a, b) => format!("distance {} {} = {}", ep_name(&a), ep_name(&b), dim.value),
        K::PointLineDistance(ep, l) => format!("distance {} {} = {}", ep_name(&ep), sketch.lines[l].name, dim.value),
        K::Angle(a, b, sup) => format!("angle {} {} = {}{}", sketch.lines[a].name, sketch.lines[b].name, dim.value, if sup { " supplement" } else { "" }),
        K::HDistance(a, b) => format!("hdistance {} {} = {}", ep_name(&a), ep_name(&b), dim.value),
        K::VDistance(a, b) => format!("vdistance {} {} = {}", ep_name(&a), ep_name(&b), dim.value),
        K::ConcentricDistance(a, b) => format!("distance {} {} = {}", sketch.arcs[a].name, sketch.arcs[b].name, dim.value),
        K::LineLineDistance(a, b) => format!("distance {} {} = {}", sketch.lines[a].name, sketch.lines[b].name, dim.value),
    }
}

// ---------------------------------------------------------------------------
// Constraint commands
// ---------------------------------------------------------------------------

/// Strip trailing "force" keyword from args, return (cleaned_args, is_force).
/// Strip inline comment: everything after `#` that is not inside quotes.
fn strip_inline_comment(input: &str) -> &str {
    let mut in_quote = false;
    for (i, ch) in input.char_indices() {
        if ch == '"' { in_quote = !in_quote; }
        else if ch == '#' && !in_quote { return &input[..i]; }
    }
    input
}

fn strip_force(args: &str) -> (&str, bool) {
    if args.trim().ends_with(" force") || args.trim() == "force" {
        let trimmed = args.trim();
        if trimmed == "force" {
            ("", true)
        } else {
            (&trimmed[..trimmed.len() - 6], true)
        }
    } else {
        (args, false)
    }
}

/// Dry-run wrapper: snapshot the sketch, execute the inner command
/// (which may add a constraint or dimension), capture whether it was
/// accepted or rejected (and any blocker info), then restore the
/// sketch so no state is kept. Returns a human-readable explanation.
/// Intended for the `explain` command and for UI code that wants to
/// preview whether a proposed constraint would succeed.
pub fn dry_run(ctx: &mut CommandContext, input: &str) -> DryRunOutcome {
    let snapshot = bincode::serialize(&ctx.sketch).ok();
    let history_cursor_before = ctx.history.cursor;
    let status_err_before = ctx.status_error.take();
    let blockers_before = ctx.status_blocker_names.take();
    let last_cost_before = ctx.last_cost;
    let dof_before = ctx.sketch.cached_dof();

    let result = execute_one(ctx, input);
    let err = ctx.status_error.take().or_else(||
        if result.is_error { Some(result.output.clone()) } else { None });
    let blockers = ctx.status_blocker_names.take();

    if let Some(snap) = snapshot
        && let Ok(restored) = bincode::deserialize::<Sketch>(&snap) {
        ctx.sketch = restored.into();
    }
    // Roll history back: drop any entries the inner command pushed
    // past the original cursor and reset the cursor itself.
    ctx.history.actions.truncate(history_cursor_before);
    ctx.history.snapshots.truncate(history_cursor_before);
    ctx.history.cursors.truncate(history_cursor_before);
    ctx.history.groups.truncate(history_cursor_before);
    ctx.history.cursor = history_cursor_before;
    ctx.status_error = status_err_before;
    ctx.status_blocker_names = blockers_before;
    ctx.last_cost = last_cost_before;
    // Re-seed the cache at the restored sketch's generation; the
    // value is still true for the restored state.
    if let Some(d) = dof_before {
        ctx.sketch.mutate_values(|s| s.set_cached_dof(d));
    }

    DryRunOutcome {
        accepted: err.is_none(),
        message: err.unwrap_or(result.output),
        blocker_names: blockers.unwrap_or_default(),
    }
}

/// Result of a dry-run command evaluation.
#[allow(dead_code)]
pub struct DryRunOutcome {
    /// True if the inner command succeeded.
    pub accepted: bool,
    /// Success output or rejection message (includes blocker hint
    /// when the inner path produced one).
    pub message: String,
    /// User-facing names of conflicting constraints when the inner
    /// path was a DOF-rejection with blocker analysis. Empty
    /// otherwise. Intended for UI consumers (e.g. drag preview) that
    /// want the structured names rather than parsing the message.
    pub blocker_names: Vec<String>,
}

fn cmd_explain(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let inner = args.trim();
    if inner.is_empty() {
        return err("Usage: explain <constraint-command> [args]");
    }
    let outcome = dry_run(ctx, inner);
    let tag = if outcome.accepted { "accepts" } else { "rejects" };
    let msg = if outcome.message.is_empty() {
        format!("'{}': {} (no further detail)", inner, tag)
    } else {
        format!("'{}': {} -- {}", inner, tag, outcome.message)
    };
    ok(msg)
}

fn cmd_horizontal(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut lines = Vec::new();
    for name in args.split_whitespace() {
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => lines.push(r),
            Err(e) => return err(e),
        }
    }
    if lines.is_empty() { return err("Usage: horizontal L0 [L1 ...]"); }
    for &r in &lines {
        if ctx.sketch.lines[r].constraints.horizontal {
            return err(format!("{} is already horizontal", ctx.sketch.lines[r].name));
        }
    }
    ctx.begin_group();
    let lines_copy = lines.clone();
    ctx.exec(Action::ApplyHorizontal { lines });
    let parts: Vec<String> = lines_copy.iter()
        .map(|r| {
            let name = arael_sketch_solver::format_flag_name(&ctx.sketch.lines[*r].name, 'H');
            applied_msg(&ctx.sketch, &name, &format!("{}: horizontal", name))
        })
        .collect();
    ok_or_status(ctx, parts.join(", "))
}

fn cmd_vertical(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut lines = Vec::new();
    for name in args.split_whitespace() {
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => lines.push(r),
            Err(e) => return err(e),
        }
    }
    if lines.is_empty() { return err("Usage: vertical L0 [L1 ...]"); }
    for &r in &lines {
        if ctx.sketch.lines[r].constraints.vertical {
            return err(format!("{} is already vertical", ctx.sketch.lines[r].name));
        }
    }
    ctx.begin_group();
    let lines_copy = lines.clone();
    ctx.exec(Action::ApplyVertical { lines });
    let parts: Vec<String> = lines_copy.iter()
        .map(|r| {
            let name = arael_sketch_solver::format_flag_name(&ctx.sketch.lines[*r].name, 'V');
            applied_msg(&ctx.sketch, &name, &format!("{}: vertical", name))
        })
        .collect();
    ok_or_status(ctx, parts.join(", "))
}

/// Classify a `parallel` argument: may be a line, an ellipse major
/// axis (only ellipses have an optimisable rotation), or fail. A
/// circular arc (non-ellipse) is rejected with a clear message.
enum ParallelArg {
    Line(Ref<Line>),
    Ellipse(Ref<Arc>),
}

fn resolve_parallel_arg(sketch: &Sketch, name: &str) -> Result<ParallelArg, String> {
    if let Ok(arc) = resolve_arc(sketch, name) {
        if !sketch.arcs[arc].is_ellipse {
            return Err(format!("parallel on {name}: only ellipses have an optimisable major-axis rotation; circular arcs have no orientation to align"));
        }
        return Ok(ParallelArg::Ellipse(arc));
    }
    let line = resolve_line(sketch, name)?;
    Ok(ParallelArg::Line(line))
}

fn cmd_parallel(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: parallel <L|EA> <L|EA>"); }
    let a = match resolve_parallel_arg(&ctx.sketch, tokens[0]) { Ok(v) => v, Err(e) => return err(e) };
    let b = match resolve_parallel_arg(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    match (a, b) {
        (ParallelArg::Line(la), ParallelArg::Line(lb)) => {
            if la == lb { return err("Cannot constrain a line parallel to itself"); }
            if ctx.sketch.parallel.iter().any(|c| (c.a == la && c.b == lb) || (c.a == lb && c.b == la)) {
                return err("Parallel constraint already exists");
            }
            ctx.begin_group();
            ctx.exec(Action::ApplyParallel { a: la, b: lb });
            let nid = ctx.sketch.parallel.last().map(|c| c.nid).unwrap_or(0);
            ok_or_status(ctx, applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied parallel"))
        }
        (ParallelArg::Ellipse(ea), ParallelArg::Ellipse(eb)) => {
            if ea == eb { return err("Cannot constrain an ellipse parallel to itself"); }
            if ctx.sketch.arc_arc_parallel.iter().any(|c| (c.a == ea && c.b == eb) || (c.a == eb && c.b == ea)) {
                return err("Parallel constraint already exists");
            }
            ctx.begin_group();
            ctx.exec(Action::ApplyArcArcParallel { a: ea, b: eb });
            let nid = ctx.sketch.arc_arc_parallel.last().map(|c| c.nid).unwrap_or(0);
            ok_or_status(ctx, applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied parallel"))
        }
        (ParallelArg::Ellipse(ea), ParallelArg::Line(lb))
        | (ParallelArg::Line(lb), ParallelArg::Ellipse(ea)) => {
            if ctx.sketch.arc_line_parallel.iter().any(|c| c.arc == ea && c.line == lb) {
                return err("Parallel constraint already exists");
            }
            ctx.begin_group();
            ctx.exec(Action::ApplyArcLineParallel { arc: ea, line: lb });
            let nid = ctx.sketch.arc_line_parallel.last().map(|c| c.nid).unwrap_or(0);
            ok_or_status(ctx, applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied parallel"))
        }
    }
}

fn cmd_perpendicular(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: perpendicular L0 L1"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain a line perpendicular to itself"); }
    if ctx.sketch.perpendicular.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Perpendicular constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyPerpendicular { a, b });
    let nid = ctx.sketch.perpendicular.last().map(|c| c.nid).unwrap_or(0);
    let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied perpendicular");
    ok_or_status(ctx, msg)
}

fn cmd_equal(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: equal L0 L1  or  equal A0 A1"); }
    if tokens[0].starts_with('L') && tokens[1].starts_with('L') {
        let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if a == b { return err("Cannot constrain a line equal to itself"); }
        let exists = ctx.sketch.equal_length.iter().any(|c|
            (c.a == a && c.b == b) || (c.a == b && c.b == a));
        if exists { return err("Equal length constraint already exists"); }
        ctx.begin_group();
        ctx.exec(Action::ApplyEqualLength { a, b });
        let nid = ctx.sketch.equal_length.last().map(|c| c.nid).unwrap_or(0);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied equal length");
        ok_or_status(ctx, msg)
    } else if is_arc_name(tokens[0]) && is_arc_name(tokens[1]) {
        let a = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if a == b { return err("Cannot constrain an arc equal to itself"); }
        let exists = ctx.sketch.equal_radius.iter().any(|c|
            (c.a == a && c.b == b) || (c.a == b && c.b == a));
        if exists { return err("Equal radius constraint already exists"); }
        ctx.begin_group();
        ctx.exec(Action::ApplyEqualRadius { a, b });
        let nid = ctx.sketch.equal_radius.last().map(|c| c.nid).unwrap_or(0);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied equal radius");
        ok_or_status(ctx, msg)
    } else {
        err("equal needs two lines or two arcs")
    }
}

fn cmd_collinear(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: collinear L0 L1"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain a line collinear with itself"); }
    if ctx.sketch.collinear.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Collinear constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyCollinear { a, b });
    let nid = ctx.sketch.collinear.last().map(|c| c.nid).unwrap_or(0);
    let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied collinear");
    ok_or_status(ctx, msg)
}

fn cmd_tangent(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: tangent L0 A0  or  tangent A0 A1"); }
    if tokens[0].starts_with('L') && is_arc_name(tokens[1]) {
        let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let arc = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if ctx.sketch.tangent_la.iter().any(|c| c.line == line && c.arc == arc) {
            return err("Tangent constraint already exists");
        }
        ctx.begin_group();
        ctx.exec(Action::ApplyTangentLA { line, arc });
        let nid = ctx.sketch.tangent_la.last().map(|c| c.nid).unwrap_or(0);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied tangent");
        ok_or_status(ctx, msg)
    } else if is_arc_name(tokens[0]) && is_arc_name(tokens[1]) {
        let a = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if a == b { return err("Cannot constrain an arc tangent to itself"); }
        if ctx.sketch.tangent_aa.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
            return err("Tangent constraint already exists");
        }
        ctx.begin_group();
        ctx.exec(Action::ApplyTangentAA { a, b });
        let nid = ctx.sketch.tangent_aa.last().map(|c| c.nid).unwrap_or(0);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied tangent");
        ok_or_status(ctx, msg)
    } else {
        err("tangent needs line+arc or arc+arc")
    }
}

fn cmd_coincident(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: coincident L0.p2 L1.p1"); }
    let a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain an endpoint coincident with itself"); }
    use EndpointRef::*;
    let s = &ctx.sketch;
    // Check for existing equivalent coincident constraint
    let exists = match (a, b) {
        (Point(a), Point(b)) => s.coincident_pp.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => s.coincident_lp1.iter().any(|c| c.line == l && c.point == p),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => s.coincident_lp2.iter().any(|c| c.line == l && c.point == p),
        (LineP1(a), LineP1(b)) => s.coincident_ll11.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (LineP1(a), LineP2(b)) => s.coincident_ll12.iter().any(|c| c.a == a && c.b == b)
            || s.coincident_ll21.iter().any(|c| c.a == b && c.b == a),
        (LineP2(a), LineP1(b)) => s.coincident_ll21.iter().any(|c| c.a == a && c.b == b)
            || s.coincident_ll12.iter().any(|c| c.a == b && c.b == a),
        (LineP2(a), LineP2(b)) => s.coincident_ll22.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => s.coincident_arc_center.iter().any(|c| c.point == p && c.arc == arc),
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => s.coincident_arc_start.iter().any(|c| c.point == p && c.arc == arc),
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => s.coincident_arc_end.iter().any(|c| c.point == p && c.arc == arc),
        (LineP1(line), ArcCenter(arc)) | (ArcCenter(arc), LineP1(line)) => s.coincident_lp1_arc_center.iter().any(|c| c.line == line && c.arc == arc),
        (LineP2(line), ArcCenter(arc)) | (ArcCenter(arc), LineP2(line)) => s.coincident_lp2_arc_center.iter().any(|c| c.line == line && c.arc == arc),
        (LineP1(line), ArcStart(arc)) | (ArcStart(arc), LineP1(line)) => s.coincident_lp1_arc_start.iter().any(|c| c.line == line && c.arc == arc),
        (LineP2(line), ArcStart(arc)) | (ArcStart(arc), LineP2(line)) => s.coincident_lp2_arc_start.iter().any(|c| c.line == line && c.arc == arc),
        (LineP1(line), ArcEnd(arc)) | (ArcEnd(arc), LineP1(line)) => s.coincident_lp1_arc_end.iter().any(|c| c.line == line && c.arc == arc),
        (LineP2(line), ArcEnd(arc)) | (ArcEnd(arc), LineP2(line)) => s.coincident_lp2_arc_end.iter().any(|c| c.line == line && c.arc == arc),
        _ => false,
    };
    if exists { return err("Coincident constraint already exists"); }
    let action = match (a, b) {
        (Point(a), Point(b)) => Action::ApplyCoincidentPP { a, b },
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => Action::ApplyCoincidentLP1 { line: l, point: p },
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => Action::ApplyCoincidentLP2 { line: l, point: p },
        (LineP1(a), LineP1(b)) => Action::ApplyCoincidentLL11 { a, b },
        (LineP1(a), LineP2(b)) => Action::ApplyCoincidentLL12 { a, b },
        (LineP2(a), LineP1(b)) => Action::ApplyCoincidentLL21 { a, b },
        (LineP2(a), LineP2(b)) => Action::ApplyCoincidentLL22 { a, b },
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => Action::ApplyCoincidentArcCenter { point: p, arc },
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => Action::ApplyCoincidentArcStart { point: p, arc },
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => Action::ApplyCoincidentArcEnd { point: p, arc },
        (LineP1(line), ArcCenter(arc)) | (ArcCenter(arc), LineP1(line)) => Action::ApplyCoincidentLP1ArcCenter { line, arc },
        (LineP2(line), ArcCenter(arc)) | (ArcCenter(arc), LineP2(line)) => Action::ApplyCoincidentLP2ArcCenter { line, arc },
        (LineP1(line), ArcStart(arc)) | (ArcStart(arc), LineP1(line)) => Action::ApplyCoincidentLP1ArcStart { line, arc },
        (LineP2(line), ArcStart(arc)) | (ArcStart(arc), LineP2(line)) => Action::ApplyCoincidentLP2ArcStart { line, arc },
        (LineP1(line), ArcEnd(arc)) | (ArcEnd(arc), LineP1(line)) => Action::ApplyCoincidentLP1ArcEnd { line, arc },
        (LineP2(line), ArcEnd(arc)) | (ArcEnd(arc), LineP2(line)) => Action::ApplyCoincidentLP2ArcEnd { line, arc },
        _ => return err("Unsupported coincident combination"),
    };
    ctx.begin_group();
    ctx.exec(action);
    // Coincident has many backing collections; saturate to the most
    // recently assigned nid via next_constraint_id - 1.
    let nid = ctx.sketch.next_constraint_id.saturating_sub(1);
    let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied coincident");
    ok_or_status(ctx, msg)
}

fn cmd_concentric(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: concentric A0 A1"); }
    let a = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain an arc concentric with itself"); }
    if ctx.sketch.concentric.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Concentric constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyConcentric { a, b });
    let nid = ctx.sketch.concentric.last().map(|c| c.nid).unwrap_or(0);
    let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied concentric");
    ok_or_status(ctx, msg)
}

// ---------------------------------------------------------------------------
// Dimension commands
// ---------------------------------------------------------------------------

/// Parse a dimension value string. Returns (numeric_value, expression_string).
/// - `=expr` or `{expr}` → live expression: (0.0, Some("expr"))
/// - `"expr"` (quoted) → live expression (backwards compat): (0.0, Some("expr"))
/// - numeric literal → (value, None)
/// - anything else → evaluate as expression to number: (value, None)
fn parse_dim_value(sketch: &Sketch, val_str: &str) -> Result<(f64, Option<String>), String> {
    let val_str = val_str.trim().trim_matches('"');
    // Snapshot form: `=expr` evaluates now and stores the result as
    // a literal (the dim won't track further changes).
    if let Some(expr) = val_str.strip_prefix('=') {
        return match eval_expr(sketch, expr.trim()) {
            Ok(value) => Ok((value, None)),
            Err(e) => Err(format!("Cannot evaluate snapshot '{}': {}", expr, e)),
        };
    }
    // Numeric literal.
    if let Ok(value) = val_str.parse::<f64>() {
        return Ok((value, None));
    }
    // Anything else: live expression (re-evaluated every solve).
    arael_sym::parse(val_str).map_err(|e|
        format!("Cannot parse value '{}': {}", val_str, e))?;
    Ok((0.0, Some(val_str.to_string())))
}

/// Find existing dimension index matching the given kind.
/// For Angle, matches regardless of supplement flag. For PointPointDistance, matches either order.
fn find_existing_dimension(sketch: &Sketch, kind: &DimensionKind) -> Option<usize> {
    sketch.dimensions.iter().position(|d| match (&d.kind, kind) {
        (DimensionKind::Angle(da, db, _), DimensionKind::Angle(a, b, _)) =>
            (*da == *a && *db == *b) || (*da == *b && *db == *a),
        (DimensionKind::PointPointDistance(a1, b1), DimensionKind::PointPointDistance(a2, b2)) =>
            (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2),
        (a, b) => a == b,
    })
}

fn cmd_length(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }
    if tokens.len() == 1 && (is_derived || is_driven) {
        let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let l = &ctx.sketch.lines[line];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let len = (dx * dx + dy * dy).sqrt();
        let kind = DimensionKind::LineLength(line);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: len, expr: None, range: None,  });
            let label = if is_derived { "derived" } else { "driven" };
            return ok_or_status(ctx, format!("Updated {} {} length = ({:.4})", name, label, len));
        }
        ctx.exec(Action::AddDimension { kind, value: len, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} length {} = ({:.4})", label, dim_name, tokens[0], len));
    }
    // Range form: `length L0 >= V`, `length L0 <= V`, `length L0 LO to HI`
    let range_opt = if tokens.len() >= 2 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[1..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let l = &ctx.sketch.lines[line];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let measured = (dx * dx + dy * dy).sqrt();
        let kind = DimensionKind::LineLength(line);
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} length {} (current {:.4})", name, bound_desc, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} length {} {} (current {:.4})", dim_name, tokens[0], bound_desc, measured));
    }

    if tokens.len() != 2 { return err("Usage: length L0 5.0 [derived|driven]  or  length L0 >=V | <=V | LO to HI"); }
    let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = DimensionKind::LineLength(line);
    let (value, expr) = match parse_dim_value(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let display = if expr.is_some() { tokens[1].to_string() } else { format!("{}", value) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} length = {}", name, display));
    }
    ctx.exec(Action::AddDimension { kind, value, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} length {} = {}", prefix, dim_name, tokens[0], display))
}

fn cmd_radius(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }
    if tokens.len() == 1 && (is_derived || is_driven) {
        let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let r = ctx.sketch.arcs[arc].radius.value;
        let kind = DimensionKind::ArcRadius(arc);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: r, expr: None, range: None,  });
            let label = if is_derived { "derived" } else { "driven" };
            return ok_or_status(ctx, format!("Updated {} {} radius = ({:.4})", label, name, r));
        }
        ctx.exec(Action::AddDimension { kind, value: r, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} radius {} = ({:.4})", label, dim_name, tokens[0], r));
    }
    // Range form: `radius A0 >= V | <= V | LO to HI`.
    let range_opt = if tokens.len() >= 2 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[1..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let measured = ctx.sketch.arcs[arc].radius.value;
        let kind = DimensionKind::ArcRadius(arc);
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} radius {} (current {:.4})", name, bound_desc, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} radius {} {} (current {:.4})", dim_name, tokens[0], bound_desc, measured));
    }

    if tokens.len() != 2 { return err("Usage: radius A0 1.5 [derived|driven]  or  radius A0 >=V | <=V | LO to HI"); }
    let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = DimensionKind::ArcRadius(arc);
    let (value, expr) = match parse_dim_value(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let display = if expr.is_some() { tokens[1].to_string() } else { format!("{}", value) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} radius = {}", name, display));
    }
    ctx.exec(Action::AddDimension { kind, value, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} radius {} = {}", prefix, dim_name, tokens[0], display))
}

fn cmd_radius_b(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }
    if tokens.len() == 1 && (is_derived || is_driven) {
        let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        if !ctx.sketch.arcs[arc].is_ellipse {
            return err("radius_b only applies to ellipses (use add_ellipse to create one)");
        }
        let r = ctx.sketch.arcs[arc].radius_b.value;
        let kind = DimensionKind::ArcRadiusB(arc);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: r, expr: None, range: None,  });
            let label = if is_derived { "derived" } else { "driven" };
            return ok_or_status(ctx, format!("Updated {} {} radius_b = ({:.4})", label, name, r));
        }
        ctx.exec(Action::AddDimension { kind, value: r, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} radius_b {} = ({:.4})", label, dim_name, tokens[0], r));
    }
    // Range form: `radius_b A0 >= V | <= V | LO to HI`.
    let range_opt = if tokens.len() >= 2 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[1..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        if !ctx.sketch.arcs[arc].is_ellipse {
            return err("radius_b only applies to ellipses (use add_ellipse to create one)");
        }
        let measured = ctx.sketch.arcs[arc].radius_b.value;
        let kind = DimensionKind::ArcRadiusB(arc);
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} radius_b {} (current {:.4})", name, bound_desc, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} radius_b {} {} (current {:.4})", dim_name, tokens[0], bound_desc, measured));
    }

    if tokens.len() != 2 { return err("Usage: radius_b A0 1.5 [derived|driven]  or  radius_b A0 >=V | <=V | LO to HI"); }
    let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    if !ctx.sketch.arcs[arc].is_ellipse {
        return err("radius_b only applies to ellipses (use add_ellipse to create one)");
    }
    let kind = DimensionKind::ArcRadiusB(arc);
    let (value, expr) = match parse_dim_value(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let display = if expr.is_some() { tokens[1].to_string() } else { format!("{}", value) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} radius_b = {}", name, display));
    }
    ctx.exec(Action::AddDimension { kind, value, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} radius_b {} = {}", prefix, dim_name, tokens[0], display))
}

fn cmd_sweep(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }
    if tokens.is_empty() { return err("Usage: sweep A0 180 [derived|driven]"); }
    let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    if ctx.sketch.arcs[arc].closed {
        return err("Cannot set sweep on a full circle (angles are fixed)");
    }
    let kind = DimensionKind::ArcSweep(arc);
    if tokens.len() == 1 && (is_derived || is_driven) {
        let a = &ctx.sketch.arcs[arc];
        let sweep_deg = arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs());
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: sweep_deg, expr: None, range: None,  });
            let label = if is_derived { "derived" } else { "driven" };
            return ok_or_status(ctx, format!("Updated {} {} sweep = ({:.4})", label, name, sweep_deg));
        }
        ctx.exec(Action::AddDimension { kind, value: sweep_deg, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} sweep {} = ({:.4})", label, dim_name, tokens[0], sweep_deg));
    }
    // Range form: `sweep A0 >= V`, `sweep A0 <= V`, `sweep A0 LO to HI`
    let range_opt = if tokens.len() >= 2 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[1..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        let a = &ctx.sketch.arcs[arc];
        let measured = arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs());
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} sweep {} (current {:.4})", name, bound_desc, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} sweep {} {} (current {:.4})", dim_name, tokens[0], bound_desc, measured));
    }

    if tokens.len() != 2 { return err("Usage: sweep A0 180 [derived|driven]  or  sweep A0 >=V | <=V | LO to HI"); }
    let (value, expr) = match parse_dim_value(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let display = if expr.is_some() { tokens[1].to_string() } else { format!("{}", value) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} sweep = {}", name, display));
    }
    ctx.exec(Action::AddDimension { kind, value, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} sweep {} = {}", prefix, dim_name, tokens[0], display))
}

// ---------------------------------------------------------------------------
// Lock/unlock
// ---------------------------------------------------------------------------

fn cmd_lock(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: lock P0  or  lock L0.p1  or  lock L0.p1 x,y"); }
    let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let pos = if tokens.len() > 1 {
        match parse_coord(ctx, tokens[1], None) { Ok(p) => p, Err(e) => return err(e) }
    } else {
        match resolve_endpoint_pos(&ctx.sketch, tokens[0]) { Ok(p) => p, Err(e) => return err(e) }
    };
    let action = match ep {
        EndpointRef::Point(p) => Action::LockPoint { point: p, pos },
        EndpointRef::LineP1(l) => Action::LockLineP1 { line: l, pos },
        EndpointRef::LineP2(l) => Action::LockLineP2 { line: l, pos },
        EndpointRef::ArcCenter(a) => Action::LockArcCenter { arc: a, pos },
        _ => return err("Can only lock points, line endpoints, and arc centers"),
    };
    ctx.begin_group();
    ctx.exec(action);
    ok(format!("Locked {} at ({:.2},{:.2})", tokens[0], pos.x, pos.y))
}

fn cmd_unlock(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    let ep = match resolve_endpoint_ref(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
    let action = match ep {
        EndpointRef::Point(p) => Action::UnlockPoint { point: p },
        EndpointRef::LineP1(l) => Action::UnlockLineP1 { line: l },
        EndpointRef::LineP2(l) => Action::UnlockLineP2 { line: l },
        EndpointRef::ArcCenter(a) => Action::UnlockArcCenter { arc: a },
        _ => return err("Can only unlock points, line endpoints, and arc centers"),
    };
    ctx.begin_group();
    ctx.exec(action);
    ok(format!("Unlocked {}", name))
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

fn cmd_param(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
    if tokens.len() != 2 { return err("Usage: param name value"); }
    let name = tokens[0].trim();
    let expr = tokens[1].trim();
    // Snapshot for rollback
    let snapshot = bincode::serialize(&ctx.sketch).ok();
    let old_cost = ctx.sketch.current_cost();
    // Check if param exists -> update
    let is_update = ctx.sketch.user_params.iter().any(|p| p.name == name);
    if is_update {
        let idx = ctx.sketch.user_params.iter().position(|p| p.name == name).unwrap();
        ctx.begin_group();
        ctx.exec(Action::UpdateUserParam { index: idx, name: name.to_string(), expr_str: expr.to_string() });
    } else {
        if let Err(e) = ctx.sketch.validate_param_name(name, None) {
            return err(e);
        }
        ctx.begin_group();
        ctx.exec(Action::AddUserParam { name: name.to_string(), expr_str: expr.to_string() });
    }
    ctx.sketch.mutate_values(|s| s.update_expr_dim_values());
    let new_cost = ctx.sketch.solve().end_cost;
    ctx.last_cost = new_cost;
    // Reject if cost increased significantly
    if new_cost > old_cost + 1e-3
        && let Some(ref snap) = snapshot
            && let Ok(restored) = bincode::deserialize::<Sketch>(snap) {
                ctx.sketch = restored.into();
                return err("Parameter change rejected: could not satisfy all constraints");
            }
    let val = ctx.sketch.user_params.iter().find(|p| p.name == name).map(|p| p.value).unwrap_or(0.0);
    ok(format!("{} {} = {} ({:.4})", if is_update { "Updated" } else { "Added" }, name, expr, val))
}

fn cmd_del_param(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == name) {
        ctx.begin_group();
        ctx.exec(Action::RemoveUserParam { index: idx });
        ok(format!("Deleted parameter {}", name))
    } else {
        err(format!("Unknown parameter: {}", name))
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

fn cmd_style(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: style L0 [solid|dashed|dashdot]"); }
    let name = tokens[0];
    if tokens.len() == 1 {
        // Query
        if name.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            return ok(format!("{}: {}", name, ctx.sketch.lines[r].style.name()));
        } else if is_arc_name(name) {
            let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            return ok(format!("{}: {}", name, ctx.sketch.arcs[r].style.name()));
        }
        return err("Style applies to lines and arcs");
    }
    let style = match LineStyle::from_name(tokens[1]) {
        Some(s) => s,
        None => return err(format!("Unknown style '{}'. Use: solid, dashed, dashdot", tokens[1])),
    };
    if name.starts_with('L') {
        let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::SetStyleLine { line: r, style });
        ok(format!("{}: {}", name, style.name()))
    } else if is_arc_name(name) {
        let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::SetStyleArc { arc: r, style });
        ok(format!("{}: {}", name, style.name()))
    } else {
        err("Style applies to lines and arcs")
    }
}

fn cmd_quiet(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: quiet L0 [on|off] | quiet A0 EA0 ..."); }
    // Check for explicit on/off
    let explicit = if tokens.last() == Some(&"on") { Some(true) }
        else if tokens.last() == Some(&"off") { Some(false) }
        else { None };
    let names = if explicit.is_some() { &tokens[..tokens.len()-1] } else { &tokens[..] };
    ctx.begin_group();
    let mut msgs = Vec::new();
    for name in names {
        if name.starts_with('P') {
            let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            let q = explicit.unwrap_or(!ctx.sketch.points[r].quiet);
            ctx.exec(Action::SetQuietPoint { point: r, on: q });
            msgs.push(format!("{}: quiet={}", name, q));
        } else if name.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            let q = explicit.unwrap_or(!ctx.sketch.lines[r].quiet);
            ctx.exec(Action::SetQuietLine { line: r, on: q });
            msgs.push(format!("{}: quiet={}", name, q));
        } else if is_arc_name(name) {
            let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            let q = explicit.unwrap_or(!ctx.sketch.arcs[r].quiet);
            ctx.exec(Action::SetQuietArc { arc: r, on: q });
            msgs.push(format!("{}: quiet={}", name, q));
        } else {
            return err(format!("Unknown entity '{}'", name));
        }
    }
    ok(msgs.join(", "))
}

fn cmd_constr(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: constr L0 [on|off] | constr A0 EA0 ..."); }
    let explicit = if tokens.last() == Some(&"on") { Some(true) }
        else if tokens.last() == Some(&"off") { Some(false) }
        else { None };
    let names = if explicit.is_some() { &tokens[..tokens.len()-1] } else { &tokens[..] };
    ctx.begin_group();
    let mut msgs = Vec::new();
    for name in names {
        if name.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            let c = explicit.unwrap_or(!ctx.sketch.lines[r].construction);
            ctx.exec(Action::SetConstructionLine { line: r, on: c });
            msgs.push(format!("{}: construction={}", name, c));
        } else if is_arc_name(name) {
            let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            let c = explicit.unwrap_or(!ctx.sketch.arcs[r].construction);
            ctx.exec(Action::SetConstructionArc { arc: r, on: c });
            msgs.push(format!("{}: construction={}", name, c));
        } else {
            return err(format!("constr applies to lines and arcs, not '{}'", name));
        }
    }
    ok(msgs.join(", "))
}

fn cmd_drag(ctx: &mut CommandContext, args: &str) -> CommandResult {

    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 {
        return err("Usage: drag L0.p1 x,y | drag L0 @dx,dy | drag P0 x,y | drag A0.center x,y");
    }
    let entity_spec = tokens[0];

    // Resolve current position of the drag target
    use arael_sketch_solver::DragTarget;

    let (target, current_pos) = if let Some((ent, field)) = entity_spec.split_once('.') {
        if ent.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, ent) { Ok(r) => r, Err(e) => return err(e) };
            match field {
                "p1" => (DragTarget::LineP1(r), ctx.sketch.lines[r].p1.value),
                "p2" => (DragTarget::LineP2(r), ctx.sketch.lines[r].p2.value),
                _ => return err(format!("Unknown line field '{}'. Use p1 or p2", field)),
            }
        } else if is_arc_name(ent) {
            let r = match resolve_arc(&ctx.sketch, ent) { Ok(r) => r, Err(e) => return err(e) };
            match field {
                "center" => (DragTarget::ArcCenter(r), ctx.sketch.arcs[r].center.value),
                "start" => (DragTarget::ArcStart(r), crate::geometry::arc_start_pos(&ctx.sketch.arcs[r])),
                "end" => (DragTarget::ArcEnd(r), crate::geometry::arc_end_pos(&ctx.sketch.arcs[r])),
                _ => return err(format!("Unknown arc field '{}'. Use center, start, or end", field)),
            }
        } else {
            return err(format!("Unknown entity '{}' in drag target", ent));
        }
    } else if entity_spec.starts_with('P') {
        let r = match resolve_point(&ctx.sketch, entity_spec) { Ok(r) => r, Err(e) => return err(e) };
        (DragTarget::Point(r), ctx.sketch.points[r].pos.value)
    } else if entity_spec.starts_with('L') {
        let r = match resolve_line(&ctx.sketch, entity_spec) { Ok(r) => r, Err(e) => return err(e) };
        let mid = vect2d::new(
            (ctx.sketch.lines[r].p1.value.x + ctx.sketch.lines[r].p2.value.x) / 2.0,
            (ctx.sketch.lines[r].p1.value.y + ctx.sketch.lines[r].p2.value.y) / 2.0,
        );
        (DragTarget::LineBody(r), mid)
    } else if is_arc_name(entity_spec) {
        let r = match resolve_arc(&ctx.sketch, entity_spec) { Ok(r) => r, Err(e) => return err(e) };
        (DragTarget::ArcBody(r), ctx.sketch.arcs[r].center.value)
    } else {
        return err(format!("Unknown entity '{}'. Use P0, L0, L0.p1, A0.center, etc.", entity_spec));
    };

    // Parse target coordinate (absolute or relative)
    let target_pos = match parse_coord(ctx, tokens[1], Some(current_pos)) {
        Ok(p) => p, Err(e) => return err(e),
    };

    // Snapshot
    let snapshot = bincode::serialize(&ctx.sketch).ok();
    let old_cost = ctx.sketch.current_cost();

    // Helper positions per target; LineBody moves both endpoints by
    // the cursor delta.
    let (hpos, hpos2) = match &target {
        DragTarget::LineBody(r) => {
            let offset = vect2d::new(target_pos.x - current_pos.x, target_pos.y - current_pos.y);
            let l = &ctx.sketch.lines[*r];
            (
                vect2d::new(l.p1.value.x + offset.x, l.p1.value.y + offset.y),
                Some(vect2d::new(l.p2.value.x + offset.x, l.p2.value.y + offset.y)),
            )
        }
        _ => (target_pos, None),
    };
    let pull = if ctx.drag_raw { None } else { Some(DRAG_PULL_WEIGHT) };
    let apparatus = ctx.sketch.get_mut().install_drag(target, hpos, hpos2, pull);

    // Solve (drag)
    ctx.sketch.solve();

    ctx.sketch.get_mut().remove_drag(&apparatus);

    // Solve (relax)
    ctx.sketch.solve();

    // Check cost
    let new_cost = ctx.sketch.current_cost();
    if new_cost > old_cost + 1e-3
        && let Some(ref snap) = snapshot
            && let Ok(restored) = bincode::deserialize::<Sketch>(snap) {
                ctx.sketch = restored.into();
                return err("Drag failed: could not satisfy constraints");
            }

    // Record the drag in history the way the GUI does -- as a full
    // state snapshot -- so undo reverts the drag instead of eating
    // the previous action.
    ctx.begin_group();
    if let Ok(snap) = bincode::serialize(&ctx.sketch) {
        ctx.history.push(
            Action::Drag { snapshot: snap },
            &ctx.sketch,
            CursorState { pos: ctx.cursor, tangent: ctx.cursor_tangent },
        );
    }

    // Report new position
    let new_pos = match &target {
        DragTarget::Point(r) => ctx.sketch.points[*r].pos.value,
        DragTarget::LineP1(r) => ctx.sketch.lines[*r].p1.value,
        DragTarget::LineP2(r) => ctx.sketch.lines[*r].p2.value,
        DragTarget::LineBody(r) => vect2d::new(
            (ctx.sketch.lines[*r].p1.value.x + ctx.sketch.lines[*r].p2.value.x) / 2.0,
            (ctx.sketch.lines[*r].p1.value.y + ctx.sketch.lines[*r].p2.value.y) / 2.0,
        ),
        DragTarget::ArcCenter(r) | DragTarget::ArcBody(r) => ctx.sketch.arcs[*r].center.value,
        DragTarget::ArcStart(r) => crate::geometry::arc_start_pos(&ctx.sketch.arcs[*r]),
        DragTarget::ArcEnd(r) => crate::geometry::arc_end_pos(&ctx.sketch.arcs[*r]),
    };
    ok(format!("Dragged {} to ({:.4}, {:.4})", entity_spec, new_pos.x, new_pos.y))
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

fn cmd_select(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();

    // select all
    if tokens.len() == 1 && tokens[0] == "all" {
        ctx.selection.clear();
        for r in ctx.sketch.points.refs() {
            if !ctx.sketch.points[r].helper { ctx.selection.push(Selection::Point(r)); }
        }
        for r in ctx.sketch.lines.refs() { ctx.selection.push(Selection::Line(r)); }
        for r in ctx.sketch.arcs.refs() { ctx.selection.push(Selection::Arc(r)); }
        return ok(format!("Selected {} entities", ctx.selection.len()));
    }

    // select <entity> chain — follow coincident endpoint connections
    if tokens.len() == 2 && tokens[1] == "chain" {
        let seed = tokens[0];
        return cmd_select_chain(ctx, seed);
    }

    // select <entity> linked — follow all constraint relationships
    if tokens.len() == 2 && tokens[1] == "linked" {
        let seed = tokens[0];
        return cmd_select_linked(ctx, seed);
    }

    for name in args.split_whitespace() {
        if name.contains('.') {
            // Endpoint selection
            let sel = match resolve_endpoint_ref(&ctx.sketch, name) {
                Ok(EndpointRef::Point(r)) => Selection::Point(r),
                Ok(EndpointRef::LineP1(r)) => Selection::LineP1(r),
                Ok(EndpointRef::LineP2(r)) => Selection::LineP2(r),
                Ok(EndpointRef::ArcCenter(r)) => Selection::ArcCenter(r),
                Ok(EndpointRef::ArcStart(r)) => Selection::ArcStart(r),
                Ok(EndpointRef::ArcEnd(r)) => Selection::ArcEnd(r),
                Err(e) => return err(e),
            };
            ctx.selection.push(sel);
        } else if name.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            ctx.selection.push(Selection::Line(r));
        } else if name.starts_with('P') {
            let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            ctx.selection.push(Selection::Point(r));
        } else if is_arc_name(name) {
            let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            ctx.selection.push(Selection::Arc(r));
        } else {
            return err(format!("Cannot select: {}", name));
        }
    }
    ok(format!("Selected {} entities", args.split_whitespace().count()))
}

/// Select all entities connected via coincident endpoint constraints, recursively.
fn cmd_select_chain(ctx: &mut CommandContext, seed: &str) -> CommandResult {
    // Resolve seed to a line or arc index
    let mut line_set: std::collections::HashSet<Ref<Line>> = std::collections::HashSet::new();
    let mut arc_set: std::collections::HashSet<Ref<Arc>> = std::collections::HashSet::new();

    if seed.starts_with('L') {
        let r = match resolve_line(&ctx.sketch, seed) { Ok(r) => r, Err(e) => return err(e) };
        line_set.insert(r);
    } else if is_arc_name(seed) {
        let r = match resolve_arc(&ctx.sketch, seed) { Ok(r) => r, Err(e) => return err(e) };
        arc_set.insert(r);
    } else {
        return err("chain requires a line or arc");
    }

    // Flood fill via coincident constraints
    loop {
        let before = line_set.len() + arc_set.len();
        // LL coincident
        for c in &ctx.sketch.coincident_ll11 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_ll12 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_ll21 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_ll22 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        // Line-Arc coincident
        for c in &ctx.sketch.coincident_lp1_arc_start { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp2_arc_start { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp1_arc_end { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp2_arc_end { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp1_arc_center { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp2_arc_center { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        // Arc-Arc coincident
        for c in &ctx.sketch.coincident_arc_center_start { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_center_end { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_start_center { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_end_center { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_start_start { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_start_end { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_end_start { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_end_end { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        // Concentric
        for c in &ctx.sketch.concentric { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        if line_set.len() + arc_set.len() == before { break; }
    }

    ctx.selection.clear();
    let mut names = Vec::new();
    for r in &line_set {
        let r = *r;
        ctx.selection.push(Selection::Line(r));
        names.push(ctx.sketch.lines[r].name.clone());
    }
    for r in &arc_set {
        let r = *r;
        ctx.selection.push(Selection::Arc(r));
        names.push(ctx.sketch.arcs[r].name.clone());
    }
    names.sort();
    ok(format!("Chain: {}", names.join(", ")))
}

/// Select all entities sharing any constraint relationship, recursively.
fn cmd_select_linked(ctx: &mut CommandContext, seed: &str) -> CommandResult {
    // Start with seed entity
    let mut line_set: std::collections::HashSet<Ref<Line>> = std::collections::HashSet::new();
    let mut arc_set: std::collections::HashSet<Ref<Arc>> = std::collections::HashSet::new();

    if seed.starts_with('L') {
        let r = match resolve_line(&ctx.sketch, seed) { Ok(r) => r, Err(e) => return err(e) };
        line_set.insert(r);
    } else if is_arc_name(seed) {
        let r = match resolve_arc(&ctx.sketch, seed) { Ok(r) => r, Err(e) => return err(e) };
        arc_set.insert(r);
    } else {
        return err("linked requires a line or arc");
    }

    // Flood fill: use list_constraints to find all relationships
    // Simpler approach: iterate all constraint vectors and propagate
    loop {
        let before = line_set.len() + arc_set.len();

        // All line-line constraints
        macro_rules! link_ll {
            ($vec:expr) => {
                for c in &$vec {
                    if line_set.contains(&c.a) { line_set.insert(c.b); }
                    if line_set.contains(&c.b) { line_set.insert(c.a); }
                }
            };
        }
        link_ll!(ctx.sketch.parallel);
        link_ll!(ctx.sketch.perpendicular);
        link_ll!(ctx.sketch.equal_length);
        link_ll!(ctx.sketch.collinear);
        link_ll!(ctx.sketch.coincident_ll11);
        link_ll!(ctx.sketch.coincident_ll12);
        link_ll!(ctx.sketch.coincident_ll21);
        link_ll!(ctx.sketch.coincident_ll22);

        // All arc-arc constraints
        macro_rules! link_aa {
            ($vec:expr) => {
                for c in &$vec {
                    if arc_set.contains(&c.a) { arc_set.insert(c.b); }
                    if arc_set.contains(&c.b) { arc_set.insert(c.a); }
                }
            };
        }
        link_aa!(ctx.sketch.equal_radius);
        link_aa!(ctx.sketch.tangent_aa);
        link_aa!(ctx.sketch.concentric);
        link_aa!(ctx.sketch.coincident_arc_center_start);
        link_aa!(ctx.sketch.coincident_arc_center_end);
        link_aa!(ctx.sketch.coincident_arc_start_center);
        link_aa!(ctx.sketch.coincident_arc_end_center);
        link_aa!(ctx.sketch.coincident_arc_start_start);
        link_aa!(ctx.sketch.coincident_arc_start_end);
        link_aa!(ctx.sketch.coincident_arc_end_start);
        link_aa!(ctx.sketch.coincident_arc_end_end);

        // Line-Arc constraints
        for c in &ctx.sketch.tangent_la {
            if line_set.contains(&c.line) { arc_set.insert(c.arc); }
            if arc_set.contains(&c.arc) { line_set.insert(c.line); }
        }
        macro_rules! link_la {
            ($vec:expr, $l:ident, $a:ident) => {
                for c in &$vec {
                    if line_set.contains(&c.$l) { arc_set.insert(c.$a); }
                    if arc_set.contains(&c.$a) { line_set.insert(c.$l); }
                }
            };
        }
        link_la!(ctx.sketch.coincident_lp1_arc_center, line, arc);
        link_la!(ctx.sketch.coincident_lp2_arc_center, line, arc);
        link_la!(ctx.sketch.coincident_lp1_arc_start, line, arc);
        link_la!(ctx.sketch.coincident_lp2_arc_start, line, arc);
        link_la!(ctx.sketch.coincident_lp1_arc_end, line, arc);
        link_la!(ctx.sketch.coincident_lp2_arc_end, line, arc);

        // Symmetry
        for c in &ctx.sketch.symmetry_ll {
            let has = line_set.contains(&c.a) || line_set.contains(&c.b) || line_set.contains(&c.c);
            if has { line_set.insert(c.a); line_set.insert(c.b); line_set.insert(c.c); }
        }

        // Angle constraints
        for c in &ctx.sketch.angle {
            if line_set.contains(&c.a) { line_set.insert(c.b); }
            if line_set.contains(&c.b) { line_set.insert(c.a); }
        }

        if line_set.len() + arc_set.len() == before { break; }
    }

    ctx.selection.clear();
    let mut names = Vec::new();
    for r in &line_set {
        let r = *r;
        ctx.selection.push(Selection::Line(r));
        names.push(ctx.sketch.lines[r].name.clone());
    }
    for r in &arc_set {
        let r = *r;
        ctx.selection.push(Selection::Arc(r));
        names.push(ctx.sketch.arcs[r].name.clone());
    }
    names.sort();
    ok(format!("Linked: {}", names.join(", ")))
}

fn cmd_deselect(ctx: &mut CommandContext, args: &str) -> CommandResult {
    if args.trim().is_empty() {
        ctx.selection.clear();
        return ok("Selection cleared");
    }
    // Deselect specific entities. An already-deselected entity is a
    // no-op; a name that resolves to nothing is an error.
    for name in args.split_whitespace() {
        if name.starts_with('L') {
            match resolve_line(&ctx.sketch, name) {
                Ok(r) => ctx.selection.retain(|s| !matches!(s, Selection::Line(l) if *l == r)),
                Err(e) => return err(e),
            }
        } else if is_arc_name(name) {
            match resolve_arc(&ctx.sketch, name) {
                Ok(r) => ctx.selection.retain(|s| !matches!(s, Selection::Arc(a) if *a == r)),
                Err(e) => return err(e),
            }
        } else if name.starts_with('P') {
            match resolve_point(&ctx.sketch, name) {
                Ok(r) => ctx.selection.retain(|s| !matches!(s, Selection::Point(p) if *p == r)),
                Err(e) => return err(e),
            }
        } else {
            return err(format!("Unknown entity: {}", name));
        }
    }
    ok(format!("Selection: {} entities", ctx.selection.len()))
}

// ---------------------------------------------------------------------------
// Print / Info / List
// ---------------------------------------------------------------------------

fn cmd_print(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    // Try geometric coordinate function first (prints x,y)
    if let Some(result) = eval_geo_coord(&ctx.sketch, args) {
        return match result {
            Ok(v) => ok(format!("({:.6}, {:.6})", v.x, v.y)),
            Err(e) => err(e),
        };
    }
    // Try geometric scalar function
    if let Some(result) = eval_geo_scalar(&ctx.sketch, args) {
        return match result {
            Ok(v) => ok(format!("{:.6}", v)),
            Err(e) => err(e),
        };
    }
    // Try session variable
    if let Some(v) = ctx.session_vars.get(args) {
        return ok(format!("{:.6}", v));
    }
    if let Some(v) = ctx.session_vecs.get(args) {
        return ok(format!("({:.6}, {:.6})", v.x, v.y));
    }
    // Eval as expression
    match eval_expr_with(&ctx.sketch, args, &ctx.session_vars) {
        Ok(v) => ok(format!("{:.6}", v)),
        Err(e) => err(format!("Eval error: {}", e)),
    }
}

/// Get all constraints that mention a given entity name.
fn constraints_for(sketch: &Sketch, name: &str) -> Vec<String> {
    sketch.list_constraints().into_iter()
        .filter(|c| c.split_whitespace().any(|w| w == name || w.starts_with(&format!("{}.", name))))
        .collect()
}

/// Format a `RangeBound` for script / info output using the same
/// surface syntax that the user typed: `>= v`, `<= v`, `lo to hi`.
fn format_range_bound(rb: &RangeBound) -> String {
    match rb {
        RangeBound::Min(v) => format!(">= {}", v),
        RangeBound::Max(v) => format!("<= {}", v),
        RangeBound::Between(lo, hi) => format!("{} to {}", lo, hi),
    }
}

fn cmd_info(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    // Constraint name lookup: C<nid> (every numeric constraint name,
    // including dimension-managed distance constraints) and
    // synthetic flag names like CL0H / CL2V.
    if name.starts_with('C') && !name.contains('.')
        && let Some(desc) = ctx.sketch.find_constraint_description(name) {
        return ok(format!("{}: {}", name, desc));
    }
    // Endpoint info: L0.p1, L0.p2, A0.center, etc.
    if name.contains('.')
        && let Ok(pos) = resolve_endpoint_pos(&ctx.sketch, name) {
            let mut s = format!("{}: ({:.4}, {:.4})", name, pos.x, pos.y);
            // Check lock status
            if let Some((entity, field)) = name.split_once('.')
                && entity.starts_with('L')
                    && let Ok(r) = resolve_line(&ctx.sketch, entity) {
                        let l = &ctx.sketch.lines[r];
                        if field == "p1" && !l.p1.optimize { s += " [locked]"; }
                        if field == "p2" && !l.p2.optimize { s += " [locked]"; }
                    }
            let cstrs = constraints_for(&ctx.sketch, name);
            if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
            return ok(s);
        }
    if name.starts_with('L') && !name.contains('.') {
        let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        let l = &ctx.sketch.lines[r];
        let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
        let mut s = format!("{}: ({:.4},{:.4})-({:.4},{:.4}) len={:.4} style={}",
            l.name, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y, len, l.style.name());
        if l.construction { s += " [constr]"; }
        if l.quiet { s += " [quiet]"; }
        if !l.p1.optimize { s += " [p1 locked]"; }
        if !l.p2.optimize { s += " [p2 locked]"; }
        let cstrs = constraints_for(&ctx.sketch, name);
        if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
        ok(s)
    } else if name.starts_with('P') && !name.contains('.') {
        let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        let p = &ctx.sketch.points[r];
        let locked = p.constraints.has_fix_x || p.constraints.has_fix_y;
        let mut s = format!("{}: ({:.4},{:.4}){}{}", p.name, p.pos.value.x, p.pos.value.y,
            if locked { " [locked]" } else { "" },
            if p.quiet { " [quiet]" } else { "" });
        let cstrs = constraints_for(&ctx.sketch, name);
        if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
        ok(s)
    } else if is_arc_name(name) && !name.contains('.') {
        let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        let a = &ctx.sketch.arcs[r];
        let sp = crate::geometry::arc_start_pos(a);
        let ep = crate::geometry::arc_end_pos(a);
        let shape_label = if a.is_ellipse {
            format!("[ellipse] ry={:.4} rot={:.1}deg", a.radius_b.value, a.rotation.value.to_degrees())
        } else if a.closed {
            "[circle]".to_string()
        } else {
            String::new()
        };
        let mut s = format!("{}: center=({:.4},{:.4}) r={:.4} angles={:.1}..{:.1} start=({:.4},{:.4}) end=({:.4},{:.4}) {}",
            a.name, a.center.value.x, a.center.value.y, a.radius.value,
            a.start_angle.value.to_degrees(), a.end_angle.value.to_degrees(),
            sp.x, sp.y, ep.x, ep.y, shape_label);
        if a.construction { s += " [constr]"; }
        if a.quiet { s += " [quiet]"; }
        let cstrs = constraints_for(&ctx.sketch, name);
        if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
        ok(s)
    } else if name.starts_with('d')
        && let Some(d) = ctx.sketch.dimensions.iter().find(|d| d.name == name) {
        let source = if let Some(rb) = &d.range {
            format!("range={}", format_range_bound(rb))
        } else if let Some(es) = &d.expr_str {
            format!("expr={}", es)
        } else {
            "expr=(numeric)".to_string()
        };
        let flags = match (d.derived, d.broken) {
            (true, true) => " derived broken",
            (true, false) => " derived",
            (false, true) => " broken",
            (false, false) => "",
        };
        ok(format!("{}: value={:.4} {} offset={:.2} along={:.2}{}",
            d.name, d.value, source, d.offset.y, d.text_along, flags))
    } else if let Some(p) = ctx.sketch.user_params.iter().find(|p| p.name == name) {
        // Checked after dimensions: a user param may start with 'd'.
        ok(format!("{}: value={:.4} expr={}{}", p.name, p.value, p.expr_str,
            if p.broken { " broken" } else { "" }))
    } else if name.starts_with('d') && name[1..].bytes().all(|b| b.is_ascii_digit()) {
        err(format!("Unknown dimension: {}", name))
    } else {
        err(format!("Unknown entity: {}", name))
    }
}

fn cmd_measure(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: measure L0 | measure L0 L1 | measure P0 P1"); }

    enum Entity { Line(Ref<Line>), Arc(Ref<Arc>), Point(vect2d, String) }
    let resolve = |token: &str| -> Result<Entity, String> {
        if let Ok(r) = resolve_line(&ctx.sketch, token) { return Ok(Entity::Line(r)); }
        if let Ok(r) = resolve_arc(&ctx.sketch, token) { return Ok(Entity::Arc(r)); }
        if let Ok(ep) = resolve_endpoint_ref(&ctx.sketch, token) {
            let pos = resolve_endpoint_pos_from_ref(&ctx.sketch, &ep);
            return Ok(Entity::Point(pos, token.to_string()));
        }
        if let Ok(r) = resolve_point(&ctx.sketch, token) {
            return Ok(Entity::Point(ctx.sketch.points[r].pos.value, token.to_string()));
        }
        Err(format!("Unknown entity: {}", token))
    };

    if tokens.len() == 1 {
        let e = match resolve(tokens[0]) { Ok(e) => e, Err(e) => return err(e) };
        match e {
            Entity::Line(r) => {
                let l = &ctx.sketch.lines[r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx).to_degrees();
                ok(format!("{}: length={:.4}, angle={:.4} deg\n  p1=({:.4},{:.4}) p2=({:.4},{:.4})",
                    l.name, len, angle, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y))
            }
            Entity::Arc(r) => {
                let a = &ctx.sketch.arcs[r];
                let sweep_deg = (a.end_angle.value - a.start_angle.value).abs().to_degrees();
                let arc_len = a.radius.value * (a.end_angle.value - a.start_angle.value).abs();
                let sp = crate::geometry::arc_start_pos(a);
                let ep = crate::geometry::arc_end_pos(a);
                let s = if a.is_ellipse {
                    format!("{}: rx={:.4}, ry={:.4}, rotation={:.4} deg, sweep={:.4} deg\n  center=({:.4},{:.4}) start=({:.4},{:.4}) end=({:.4},{:.4})",
                        a.name, a.radius.value, a.radius_b.value, a.rotation.value.to_degrees(),
                        sweep_deg, a.center.value.x, a.center.value.y, sp.x, sp.y, ep.x, ep.y)
                } else {
                    format!("{}: radius={:.4}, sweep={:.4} deg, arc_length={:.4}\n  center=({:.4},{:.4}) start=({:.4},{:.4}) end=({:.4},{:.4})",
                        a.name, a.radius.value, sweep_deg, arc_len,
                        a.center.value.x, a.center.value.y, sp.x, sp.y, ep.x, ep.y)
                };
                ok(s)
            }
            Entity::Point(pos, name) => {
                ok(format!("{}: ({:.4},{:.4})", name, pos.x, pos.y))
            }
        }
    } else if tokens.len() == 2 {
        let e1 = match resolve(tokens[0]) { Ok(e) => e, Err(e) => return err(e) };
        let e2 = match resolve(tokens[1]) { Ok(e) => e, Err(e) => return err(e) };
        match (e1, e2) {
            (Entity::Point(a, _), Entity::Point(b, _)) => {
                let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
                ok(format!("distance: {:.4}", d))
            }
            (Entity::Line(a), Entity::Line(b)) => {
                let la = &ctx.sketch.lines[a];
                let lb = &ctx.sketch.lines[b];
                let dx1 = la.p2.value.x - la.p1.value.x;
                let dy1 = la.p2.value.y - la.p1.value.y;
                let dx2 = lb.p2.value.x - lb.p1.value.x;
                let dy2 = lb.p2.value.y - lb.p1.value.y;
                let cross = dx1 * dy2 - dy1 * dx2;
                let dot = dx1 * dx2 + dy1 * dy2;
                let angle = cross.atan2(dot).to_degrees().abs();
                let supplement = 180.0 - angle;
                let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
                // Perpendicular distance from lb.p1 to line la
                let perp_dist = if len1 > 1e-12 {
                    ((lb.p1.value.x - la.p1.value.x) * dy1 - (lb.p1.value.y - la.p1.value.y) * dx1).abs() / len1
                } else { 0.0 };
                let mut lines = Vec::new();
                lines.push(format!("angle: {:.4} deg (supplement: {:.4} deg)", angle, supplement));
                if angle < 0.1 || supplement < 0.1 {
                    lines.push(format!("parallel, distance: {:.4}", perp_dist));
                }
                if (angle - 90.0).abs() < 0.1 || (supplement - 90.0).abs() < 0.1 {
                    lines.push("perpendicular".to_string());
                }
                ok(lines.join("\n"))
            }
            (Entity::Point(p, _), Entity::Line(r)) | (Entity::Line(r), Entity::Point(p, _)) => {
                let l = &ctx.sketch.lines[r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let perp_dist = if len > 1e-12 {
                    ((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx).abs() / len
                } else { 0.0 };
                ok(format!("perpendicular distance: {:.4}", perp_dist))
            }
            (Entity::Point(p, _), Entity::Arc(r)) | (Entity::Arc(r), Entity::Point(p, _)) => {
                let a = &ctx.sketch.arcs[r];
                let dc = ((p.x - a.center.value.x).powi(2) + (p.y - a.center.value.y).powi(2)).sqrt();
                let dist_to_arc = (dc - a.radius.value).abs();
                ok(format!("distance to center: {:.4}, distance to arc: {:.4}", dc, dist_to_arc))
            }
            (Entity::Line(lr), Entity::Arc(ar)) | (Entity::Arc(ar), Entity::Line(lr)) => {
                let l = &ctx.sketch.lines[lr];
                let a = &ctx.sketch.arcs[ar];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let perp_dist = if len > 1e-12 {
                    ((a.center.value.x - l.p1.value.x) * dy - (a.center.value.y - l.p1.value.y) * dx).abs() / len
                } else { 0.0 };
                let gap = perp_dist - a.radius.value;
                let mut lines = Vec::new();
                lines.push(format!("center-to-line distance: {:.4}, gap: {:.4}", perp_dist, gap));
                if gap.abs() < 0.01 { lines.push("tangent".to_string()); }
                ok(lines.join("\n"))
            }
            (Entity::Arc(a), Entity::Arc(b)) => {
                let aa = &ctx.sketch.arcs[a];
                let ab = &ctx.sketch.arcs[b];
                let dc = ((aa.center.value.x - ab.center.value.x).powi(2) +
                          (aa.center.value.y - ab.center.value.y).powi(2)).sqrt();
                ok(format!("center-to-center: {:.4}, radii: {:.4} + {:.4} = {:.4}, gap: {:.4}",
                    dc, aa.radius.value, ab.radius.value,
                    aa.radius.value + ab.radius.value,
                    dc - aa.radius.value - ab.radius.value))
            }
        }
    } else {
        err("Usage: measure L0 | measure L0 L1 | measure P0 P1")
    }
}

/// Resolve endpoint ref to position.
fn resolve_endpoint_pos_from_ref(sketch: &Sketch, ep: &EndpointRef) -> vect2d {
    match ep {
        EndpointRef::Point(r) => sketch.points[*r].pos.value,
        EndpointRef::LineP1(r) => sketch.lines[*r].p1.value,
        EndpointRef::LineP2(r) => sketch.lines[*r].p2.value,
        EndpointRef::ArcCenter(r) => sketch.arcs[*r].center.value,
        EndpointRef::ArcStart(r) => crate::geometry::arc_start_pos(&sketch.arcs[*r]),
        EndpointRef::ArcEnd(r) => crate::geometry::arc_end_pos(&sketch.arcs[*r]),
    }
}

fn cmd_list(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let filter = args.trim();
    if filter == "selection" {
        if ctx.selection.is_empty() { return ok("(no selection)"); }
        let names: Vec<String> = ctx.selection.iter().map(|s| match s {
            Selection::Point(r) => ctx.sketch.points[*r].name.clone(),
            Selection::Line(r) => ctx.sketch.lines[*r].name.clone(),
            Selection::Arc(r) => ctx.sketch.arcs[*r].name.clone(),
            Selection::LineP1(r) => format!("{}.p1", ctx.sketch.lines[*r].name),
            Selection::LineP2(r) => format!("{}.p2", ctx.sketch.lines[*r].name),
            Selection::ArcCenter(r) => format!("{}.center", ctx.sketch.arcs[*r].name),
            Selection::ArcStart(r) => format!("{}.start", ctx.sketch.arcs[*r].name),
            Selection::ArcEnd(r) => format!("{}.end", ctx.sketch.arcs[*r].name),
            Selection::Constraint(_) => "constraint".into(),
            Selection::Dimension(did) => ctx.sketch.dimension_index_by_did(*did).and_then(|i| ctx.sketch.dimensions.get(i)).map(|d| d.name.clone()).unwrap_or("dim?".into()),
        }).collect();
        return ok(names.join(", "));
    }
    // Constraint/dimension type filters — show only matching entries
    const CONSTRAINT_FILTERS: &[&str] = &[
        "horizontal", "vertical", "parallel", "perpendicular", "equal", "collinear",
        "tangent", "coincident", "concentric", "midpoint", "symmetry", "point_on", "lock",
    ];
    const DIMENSION_FILTERS: &[&str] = &["angle", "length", "radius", "sweep", "distance", "hdistance", "vdistance", "xangle"];
    // Match against the line body — `list` output prefixes every constraint
    // with its C<name>: tag.
    fn body_matches(s: &str, needle: &str) -> bool {
        let body = s.split_once(": ").map(|(_, rest)| rest).unwrap_or(s);
        body.starts_with(needle)
    }
    if CONSTRAINT_FILTERS.contains(&filter) {
        let all = ctx.sketch.list_constraints();
        let filtered: Vec<String> = all.into_iter().filter(|s| body_matches(s, filter)).collect();
        return if filtered.is_empty() { ok("(empty)") } else { ok(filtered.join("\n")) };
    }
    if DIMENSION_FILTERS.contains(&filter) {
        let all = ctx.sketch.list_constraints();
        let filtered: Vec<String> = all.into_iter().filter(|s| body_matches(s, filter)).collect();
        return if filtered.is_empty() { ok("(empty)") } else { ok(filtered.join("\n")) };
    }
    if filter == "constr" {
        let mut items = Vec::new();
        for r in ctx.sketch.lines.refs() {
            let l = &ctx.sketch.lines[r];
            if !l.construction { continue; }
            let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
            items.push(format!("{}: ({:.2},{:.2})-({:.2},{:.2}) len={:.2} [constr]",
                l.name, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y, len));
        }
        for r in ctx.sketch.arcs.refs() {
            let a = &ctx.sketch.arcs[r];
            if !a.construction { continue; }
            items.push(format!("{}: center=({:.2},{:.2}) r={:.2} [constr]",
                a.name, a.center.value.x, a.center.value.y, a.radius.value));
        }
        return if items.is_empty() { ok("(no construction entities)") } else { ok(items.join("\n")) };
    }
    if !filter.is_empty() && !matches!(filter, "all" | "lines" | "points" | "arcs" | "dims" | "params" | "constraints") {
        return err(format!("Unknown filter: {}. Use: all, lines, points, arcs, dims, params, constraints, constr, selection, or a constraint type (horizontal, parallel, ...)", filter));
    }
    let mut lines = Vec::new();
    let show_all = filter.is_empty() || filter == "all";

    if show_all || filter == "lines" {
        for r in ctx.sketch.lines.refs() {
            let l = &ctx.sketch.lines[r];
            let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
            let c = if l.construction { " [constr]" } else { "" };
            let q = if l.quiet { " [quiet]" } else { "" };
            lines.push(format!("{}: ({:.2},{:.2})-({:.2},{:.2}) len={:.2}{c}{q}",
                l.name, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y, len));
        }
    }
    if show_all || filter == "points" {
        for r in ctx.sketch.points.refs() {
            let p = &ctx.sketch.points[r];
            if p.helper { continue; }
            let q = if p.quiet { " [quiet]" } else { "" };
            lines.push(format!("{}: ({:.2},{:.2}){q}", p.name, p.pos.value.x, p.pos.value.y));
        }
    }
    if show_all || filter == "arcs" {
        for r in ctx.sketch.arcs.refs() {
            let a = &ctx.sketch.arcs[r];
            let c = if a.construction { " [constr]" } else { "" };
            let q = if a.quiet { " [quiet]" } else { "" };
            if a.is_ellipse {
                if a.closed {
                    lines.push(format!("{}: center=({:.2},{:.2}) rx={:.2} ry={:.2} rot={:.1}deg [ellipse]{c}{q}",
                        a.name, a.center.value.x, a.center.value.y,
                        a.radius.value, a.radius_b.value, a.rotation.value.to_degrees()));
                } else {
                    let sp = crate::geometry::arc_start_pos(a);
                    let ep = crate::geometry::arc_end_pos(a);
                    lines.push(format!("{}: center=({:.2},{:.2}) rx={:.2} ry={:.2} rot={:.1}deg start=({:.2},{:.2}) end=({:.2},{:.2}) [elliptic arc]{c}{q}",
                        a.name, a.center.value.x, a.center.value.y,
                        a.radius.value, a.radius_b.value, a.rotation.value.to_degrees(),
                        sp.x, sp.y, ep.x, ep.y));
                }
            } else if a.closed {
                lines.push(format!("{}: center=({:.2},{:.2}) r={:.2} [circle]{c}{q}",
                    a.name, a.center.value.x, a.center.value.y, a.radius.value));
            } else {
                let sp = crate::geometry::arc_start_pos(a);
                let ep = crate::geometry::arc_end_pos(a);
                lines.push(format!("{}: center=({:.2},{:.2}) r={:.2} start=({:.2},{:.2}) end=({:.2},{:.2}){c}{q}",
                    a.name, a.center.value.x, a.center.value.y, a.radius.value,
                    sp.x, sp.y, ep.x, ep.y));
            }
        }
    }
    if show_all || filter == "dims" {
        for d in &ctx.sketch.dimensions {
            let source = if let Some(rb) = &d.range {
                format_range_bound(rb)
            } else {
                d.expr_str.clone().unwrap_or_default()
            };
            let tag = if d.derived { " derived" } else { "" };
            lines.push(format!("{}: {:.4} {}{}", d.name, d.value, source, tag));
        }
    }
    if show_all || filter == "params" {
        for p in &ctx.sketch.user_params {
            lines.push(format!("{} = {} ({:.4}){}", p.name, p.expr_str, p.value,
                if p.broken { " broken" } else { "" }));
        }
    }
    if show_all || filter == "constraints" {
        lines.extend(ctx.sketch.list_constraints());
    }
    if lines.is_empty() {
        ok("(empty)")
    } else {
        ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Undo/Redo/History
// ---------------------------------------------------------------------------

fn cmd_undo(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let n: usize = args.trim().parse().unwrap_or(1);
    for _ in 0..n {
        if let Some((s, c)) = ctx.history.undo() {
            ctx.sketch = s.into();
            ctx.cursor = c.pos;
            ctx.cursor_tangent = c.tangent;
        } else {
            return ok("Nothing to undo");
        }
    }
    // History::undo returns a solved sketch; no second solve.
    ok(format!("Undone {} step(s)", n))
}

fn cmd_redo(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let n: usize = args.trim().parse().unwrap_or(1);
    for _ in 0..n {
        if let Some((s, c)) = ctx.history.redo() {
            ctx.sketch = s.into();
            ctx.cursor = c.pos;
            ctx.cursor_tangent = c.tangent;
        } else {
            return ok("Nothing to redo");
        }
    }
    // History::redo returns a solved sketch; no second solve.
    ok(format!("Redone {} step(s)", n))
}

fn cmd_history(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let n: usize = args.trim().parse().unwrap_or(usize::MAX);
    let groups = ctx.history.group_list();
    let total = groups.len();
    let start = total.saturating_sub(n);
    let cursor = ctx.history.cursor;
    let mut lines = Vec::new();
    for (i, (_, end_pos, desc)) in groups.iter().enumerate().skip(start) {
        let marker = if *end_pos == cursor { " <--" } else { "" };
        lines.push(format!("[{}] {}{}", i + 1, desc, marker));
    }
    if cursor == 0 {
        lines.insert(0, "[0] (initial state) <--".into());
    } else {
        lines.insert(0, "[0] (initial state)".into());
    }
    ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

fn cmd_center(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    if args.is_empty() {
        // Fit all -- delegate to pending_fit
        ctx.pending_fit = true;
        return ok("Fitting all");
    }
    let pos = match parse_coord(ctx, args, None) {
        Ok(p) => p,
        Err(_) => {
            // Try as entity name
            if let Ok(p) = resolve_endpoint_pos(&ctx.sketch, args) { p }
            else if args.starts_with('L') {
                if let Ok(r) = resolve_line(&ctx.sketch, args) {
                    let l = &ctx.sketch.lines[r];
                    vect2d::new((l.p1.value.x + l.p2.value.x) / 2.0, (l.p1.value.y + l.p2.value.y) / 2.0)
                } else { return err(format!("Unknown: {}", args)); }
            } else { return err(format!("Cannot resolve: {}", args)); }
        }
    };
    // Center the view on pos
    ctx.offset_x = 400.0 - pos.x as f32 * ctx.scale;
    ctx.offset_y = 300.0 + pos.y as f32 * ctx.scale;
    ok(format!("Centered on ({:.2},{:.2})", pos.x, pos.y))
}

fn cmd_zoom(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    match args {
        "+" => { ctx.scale *= 1.5; ok(format!("Zoom: {:.1}", ctx.scale)) }
        "-" => { ctx.scale /= 1.5; ok(format!("Zoom: {:.1}", ctx.scale)) }
        _ => {
            if let Ok(v) = args.parse::<f32>() {
                ctx.scale = v.clamp(1e-4, 1e7);
                ok(format!("Zoom: {:.1}", ctx.scale))
            } else {
                err("Usage: zoom +  or  zoom -  or  zoom 2.0")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Additional geometry
// ---------------------------------------------------------------------------

fn cmd_add_arc(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven"]);
    if tokens.len() != 3 { return err("Usage: add_arc x1,y1 x2,y2 xm,ym [noconnect] [notangent] [nocursor] [driven]"); }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let pm = match parse_coord(ctx, tokens[2], None) { Ok(p) => p, Err(e) => return err(e) };
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddArc { start: p1, end: p2, mid: pm }).arc() else {
        return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    if quiet { ctx.exec(Action::SetQuietArc { arc: arc_ref, on: true }); }
    if constr { ctx.exec(Action::SetConstructionArc { arc: arc_ref, on: true }); }
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor {
        ctx.cursor = Some(p2);
        set_cursor_tangent_from_arc(ctx, arc_ref);
    }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}", name);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, false);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
        if !notangent {
            let tangents = auto_tangent_arc(ctx, arc_ref);
            if !tangents.is_empty() {
                msg += &format!(" [tangent: {}]", tangents.join(", "));
            }
        }
    }
    if driven {
        let radius = ctx.sketch.arcs[arc_ref].radius.value;
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: radius, expr: None, derived: false, range: None,
        }, "radius", radius);
        let a = &ctx.sketch.arcs[arc_ref];
        let sweep = (a.end_angle.value - a.start_angle.value).abs().to_degrees();
        msg += &driven_dim_fragment(ctx, Action::AddDimension {
            kind: DimensionKind::ArcSweep(arc_ref),
            value: sweep, expr: None, derived: false, range: None,
        }, "sweep", sweep);
    }
    if quiet { msg += " [quiet]"; }
    ok(msg)
}

/// Scan the four line-line coincident collections for a constraint
/// involving `(line, is_p1)`. Returns the partner line, which of its
/// endpoints is the shared corner, and the ConstraintId so the caller
/// can delete the coincidence during a fillet or similar topology
/// edit. Returns None if the endpoint isn't coincident with any other
/// line's endpoint directly -- shared-via-helper-point corners are
/// rejected on purpose, keeping the fillet scope tractable.
fn find_ll_coincident_partner(
    sketch: &Sketch,
    line: Ref<Line>,
    is_p1: bool,
) -> Option<(Ref<Line>, bool, crate::ids::ConstraintId)> {
    use crate::ids::ConstraintId;
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

/// Refs resolved from a corner argument token (or pair of tokens):
/// the two lines that meet at the corner, which endpoint of each is
/// at the corner, and the coincident constraint that ties them there.
struct CornerRefs {
    line_a: Ref<Line>,
    is_p1_a: bool,
    line_b: Ref<Line>,
    is_p1_b: bool,
    coincident_id: crate::ids::ConstraintId,
}

/// Resolve a single corner spec (list of 1 or 2 tokens) against the
/// sketch. Used by fillet and chamfer to accept any mix of
/// `L1 L2` and `L1.pN` args on one command line.
fn resolve_corner_tokens(sketch: &Sketch, tokens: &[String]) -> Result<CornerRefs, String> {
    match tokens.len() {
        1 => {
            let ep = resolve_endpoint_ref(sketch, &tokens[0])?;
            let (la, is_p1_a) = match ep {
                EndpointRef::LineP1(l) => (l, true),
                EndpointRef::LineP2(l) => (l, false),
                _ => return Err(format!("endpoint must be a line end: {}", tokens[0])),
            };
            match find_ll_coincident_partner(sketch, la, is_p1_a) {
                Some((lb, is_p1_b, cid)) => Ok(CornerRefs {
                    line_a: la, is_p1_a, line_b: lb, is_p1_b, coincident_id: cid }),
                None => Err(format!(
                    "{} isn't coincident with another line endpoint", tokens[0])),
            }
        }
        2 => {
            let la = resolve_line(sketch, &tokens[0])?;
            let lb = resolve_line(sketch, &tokens[1])?;
            if la == lb { return Err("two-line corner needs two different lines".into()); }
            let mut found = None;
            for probe in [true, false] {
                if let Some((p, is_p1_p, id)) = find_ll_coincident_partner(sketch, la, probe)
                    && p == lb
                {
                    found = Some((probe, is_p1_p, id));
                    break;
                }
            }
            match found {
                Some((is_p1_a, is_p1_b, cid)) => Ok(CornerRefs {
                    line_a: la, is_p1_a, line_b: lb, is_p1_b, coincident_id: cid }),
                None => Err(format!(
                    "{} and {} are not connected at an endpoint",
                    tokens[0], tokens[1])),
            }
        }
        _ => Err("corner expects 1 token (Ln.pN) or 2 tokens (Ln Ln)".into()),
    }
}

/// Split the full corner-spec list (everything before the trailing
/// radius/distance token) into individual corner specs. An `Lx.pN`
/// token stands alone; a bare `Lx` consumes the following `Lx` too
/// to form a two-line corner.
fn parse_corner_list(tokens: &[&str]) -> Result<Vec<Vec<String>>, String> {
    let mut corners = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        if t.contains('.') {
            corners.push(vec![t.to_string()]);
            i += 1;
        } else if t.starts_with('L') {
            if i + 1 >= tokens.len() {
                return Err(format!("expected a second line after '{}'", t));
            }
            let next = tokens[i + 1];
            if !next.starts_with('L') || next.contains('.') {
                return Err(format!(
                    "expected a bare line name after '{}', got '{}'", t, next));
            }
            corners.push(vec![t.to_string(), next.to_string()]);
            i += 2;
        } else {
            return Err(format!("unexpected corner token '{}'", t));
        }
    }
    Ok(corners)
}

/// Parse a radius/distance token as either a numeric literal or a
/// live expression. Live exprs are stored on the primary dim so the
/// whole operation tracks its source. For chained corners we also
/// need an evaluated numeric to drive geometry right now.
fn parse_radius_token(sketch: &Sketch, tok: &str) -> Result<(f64, Option<String>), String> {
    match parse_dim_value(sketch, tok)? {
        (v, None) => Ok((v, None)),
        (_, Some(expr)) => {
            let v = eval_expr(sketch, &expr)
                .map_err(|e| format!("Cannot evaluate '{}': {}", expr, e))?;
            Ok((v, Some(expr)))
        }
    }
}

fn cmd_fillet(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [notangent, noradius] = peel_keywords(&mut tokens, ["notangent", "noradius"]);
    if tokens.len() < 2 {
        return err("Usage: fillet <corner>... r [notangent] [noradius]  where each corner is Lx.pN or Lx Ly");
    }
    let radius_tok = tokens.pop().unwrap();
    let (radius, radius_expr) = match parse_radius_token(&ctx.sketch, radius_tok) {
        Ok(v) => v, Err(e) => return err(e),
    };
    if radius <= 1e-9 { return err("fillet radius must be positive"); }

    let corner_specs = match parse_corner_list(&tokens) { Ok(v) => v, Err(e) => return err(format!("fillet: {}", e)) };
    if corner_specs.is_empty() {
        return err("fillet: need at least one corner");
    }

    // Validate once up front so clearly-bad commands fail before any
    // mutation. The result is discarded; the actual fillet loop
    // re-resolves each spec after the previous fillet because
    // coincident-collection indices shift when an LL coincident is
    // removed.
    for spec in &corner_specs {
        if let Err(e) = resolve_corner_tokens(&ctx.sketch, spec) {
            return err(format!("fillet {}: {}", spec.join(" "), e));
        }
    }

    let mut outs: Vec<FilletOut> = Vec::new();
    ctx.begin_group();

    // First corner takes the user's radius expression (or literal);
    // subsequent corners reference the first corner's dim name so
    // they all track a single source.
    let mut primary_dim_name: Option<String> = None;
    for (idx, spec) in corner_specs.iter().enumerate() {
        // Re-resolve against current state -- every previous fillet
        // deleted one LL coincident, shifting later indices.
        let refs = match resolve_corner_tokens(&ctx.sketch, spec) {
            Ok(r) => r,
            Err(e) => {
                outs.push(FilletOut {
                    spec: spec.clone(),
                    arc_name: String::new(),
                    removed: None,
                    dim_name: None,
                    added: vec![format!("FAILED: {}", e)],
                });
                continue;
            }
        };
        let (radius_for_this, expr_for_this) = if idx == 0 {
            (radius, radius_expr.clone())
        } else if let Some(name) = &primary_dim_name {
            let v = eval_expr(&ctx.sketch, name).unwrap_or(radius);
            (v, Some(name.clone()))
        } else {
            (radius, None)
        };
        let out = match apply_one_fillet(
            ctx, spec.clone(), &refs, radius_for_this, expr_for_this, notangent, noradius,
        ) {
            Ok(o) => o,
            Err(e) => {
                outs.push(FilletOut {
                    spec: spec.clone(),
                    arc_name: String::new(),
                    removed: None,
                    dim_name: None,
                    added: vec![format!("FAILED: {}", e)],
                });
                continue;
            }
        };
        if idx == 0 && out.dim_name.is_some() {
            primary_dim_name = out.dim_name.clone();
        }
        outs.push(out);
    }

    // Format one line per corner: the spec, the new arc, the dim,
    // the deleted coincident and the set of added constraint ids.
    let mut lines: Vec<String> = Vec::with_capacity(outs.len());
    for out in &outs {
        if out.arc_name.is_empty() {
            lines.push(format!("  {}: {}", out.spec.join(" "), out.added.join(" ")));
            continue;
        }
        let mut parts = Vec::new();
        parts.push(out.arc_name.clone());
        if let Some(d) = &out.dim_name { parts.push(d.clone()); }
        let mut tail = Vec::new();
        if let Some(r) = &out.removed { tail.push(format!("removed {}", r)); }
        if !out.added.is_empty() { tail.push(format!("added {}", out.added.join(" "))); }
        let tail_str = if tail.is_empty() { String::new() } else { format!(" [{}]", tail.join(", ")) };
        lines.push(format!("  {} -> {}{}", out.spec.join(" "), parts.join(" "), tail_str));
    }
    let succeeded = outs.iter().filter(|o| !o.arc_name.is_empty()).count();
    let header = if corner_specs.len() == 1 {
        format!("Filleted (r={:.4}):", radius)
    } else {
        format!("Filleted {} of {} corners (r={:.4}):",
            succeeded, corner_specs.len(), radius)
    };
    if let Some(last) = outs.iter().rev().find(|o| !o.arc_name.is_empty()) {
        ctx.session_names.insert("_".into(), last.arc_name.clone());
    }
    let msg = format!("{}\n{}", header, lines.join("\n"));
    if succeeded == 0 { err(msg) } else { ok(msg) }
}

/// Apply one fillet at the resolved corner. Returns the ids of every
/// new constraint and the deleted coincident so the caller can
/// surface them in the command output. Returns Err with a
/// human-readable string for geometry errors (too short, zero-angle,
/// etc.) without mutating the sketch.
fn apply_one_fillet(
    ctx: &mut CommandContext,
    spec: Vec<String>,
    refs: &CornerRefs,
    radius: f64,
    radius_expr: Option<String>,
    notangent: bool,
    noradius: bool,
) -> Result<FilletOut, String> {
    let CornerRefs { line_a, is_p1_a, line_b, is_p1_b, coincident_id } = *refs;

    // Geometry: unit direction vectors from the shared corner toward
    // the far endpoint of each line.
    let la_ref = &ctx.sketch.lines[line_a];
    let lb_ref = &ctx.sketch.lines[line_b];
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
    let removed = crate::ids::constraint_id_name(&ctx.sketch, coincident_id);
    ctx.exec(Action::DeleteConstraint { id: coincident_id });

    if is_p1_a { ctx.sketch.get_mut().lines[line_a].p1.value = t_a; }
    else { ctx.sketch.get_mut().lines[line_a].p2.value = t_a; }
    if is_p1_b { ctx.sketch.get_mut().lines[line_b].p1.value = t_b; }
    else { ctx.sketch.get_mut().lines[line_b].p2.value = t_b; }

    let Some(arc_ref) = ctx.exec(Action::AddArc { start: t_a, end: t_b, mid }).arc() else {
        return Err("Cannot fillet: degenerate corner geometry".into());
    };
    let arc_name = ctx.sketch.arcs[arc_ref].name.clone();

    let arc_start = arc_start_pos(&ctx.sketch.arcs[arc_ref]);
    let arc_end = arc_end_pos(&ctx.sketch.arcs[arc_ref]);
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
    // Helper: look up the most recently added constraint in the
    // collection whose kind matches `action` and format its id.
    let bridge_name = |ctx: &CommandContext, action: &Action| -> Option<String> {
        match action {
            Action::ApplyCoincidentLP1ArcStart { .. } => ctx.sketch.coincident_lp1_arc_start.last().map(|c| format!("C{}", c.nid)),
            Action::ApplyCoincidentLP1ArcEnd { .. } => ctx.sketch.coincident_lp1_arc_end.last().map(|c| format!("C{}", c.nid)),
            Action::ApplyCoincidentLP2ArcStart { .. } => ctx.sketch.coincident_lp2_arc_start.last().map(|c| format!("C{}", c.nid)),
            Action::ApplyCoincidentLP2ArcEnd { .. } => ctx.sketch.coincident_lp2_arc_end.last().map(|c| format!("C{}", c.nid)),
            _ => None,
        }
    };
    let saved_skip = ctx.skip_dof_check;
    ctx.skip_dof_check = true;
    let ca = coincide_a.clone();
    ctx.exec(coincide_a);
    if let Some(n) = bridge_name(ctx, &ca) { added.push(n); }
    let cb = coincide_b.clone();
    ctx.exec(coincide_b);
    if let Some(n) = bridge_name(ctx, &cb) { added.push(n); }
    ctx.skip_dof_check = saved_skip;

    if !notangent {
        // A rejected tangent is reported in the corner's output, not
        // silently dropped.
        for line in [line_a, line_b] {
            ctx.exec(Action::ApplyTangentLA { line, arc: arc_ref });
            match ctx.status_error.take() {
                None => {
                    if let Some(c) = ctx.sketch.tangent_la.last() {
                        added.push(format!("C{}", c.nid));
                    }
                }
                Some(e) => added.push(format!(
                    "tangent {} skipped ({})", ctx.sketch.lines[line].name, e)),
            }
        }
    }

    let mut dim_name: Option<String> = None;
    if !noradius {
        ctx.exec(Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: radius, expr: radius_expr, derived: false, range: None,
        });
        match ctx.status_error.take() {
            None => dim_name = Some(last_dim_name(ctx)),
            Some(e) => added.push(format!("radius dim skipped ({})", e)),
        }
    }

    Ok(FilletOut { spec, arc_name, removed, dim_name, added })
}

/// Outcome of one fillet corner, bubbled up to cmd_fillet so every
/// success message lists the deleted coincident and every added
/// constraint / dim id.
struct FilletOut {
    spec: Vec<String>,
    arc_name: String,
    removed: Option<String>,
    dim_name: Option<String>,
    added: Vec<String>,
}

fn cmd_chamfer(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() < 2 {
        return err("Usage: chamfer <corner>... d  where each corner is Lx.pN or Lx Ly");
    }
    let dist_tok = tokens.pop().unwrap();
    let (distance, dist_expr) = match parse_radius_token(&ctx.sketch, dist_tok) {
        Ok(v) => v, Err(e) => return err(e),
    };
    if distance <= 1e-9 { return err("chamfer distance must be positive"); }

    let corner_specs = match parse_corner_list(&tokens) { Ok(v) => v, Err(e) => return err(format!("chamfer: {}", e)) };
    if corner_specs.is_empty() { return err("chamfer: need at least one corner"); }

    // Validate once up front; re-resolve inside the loop because
    // coincident-collection indices shift when an LL coincident is
    // removed by a prior chamfer.
    for spec in &corner_specs {
        if let Err(e) = resolve_corner_tokens(&ctx.sketch, spec) {
            return err(format!("chamfer {}: {}", spec.join(" "), e));
        }
    }

    let mut outs: Vec<ChamferOut> = Vec::new();
    ctx.begin_group();

    let mut primary_dim_name: Option<String> = None;
    for (idx, spec) in corner_specs.iter().enumerate() {
        let refs = match resolve_corner_tokens(&ctx.sketch, spec) {
            Ok(r) => r,
            Err(e) => {
                outs.push(ChamferOut {
                    spec: spec.clone(),
                    new_line_name: String::new(),
                    point_name: String::new(),
                    removed: None,
                    primary_dim: None,
                    secondary_dim: None,
                    added: vec![format!("FAILED: {}", e)],
                });
                continue;
            }
        };
        let (distance_for_this, expr_for_this) = if idx == 0 {
            (distance, dist_expr.clone())
        } else if let Some(name) = &primary_dim_name {
            let v = eval_expr(&ctx.sketch, name).unwrap_or(distance);
            (v, Some(name.clone()))
        } else {
            (distance, None)
        };
        let out = match apply_one_chamfer(ctx, spec.clone(), &refs, distance_for_this, expr_for_this) {
            Ok(o) => o,
            Err(e) => {
                outs.push(ChamferOut {
                    spec: spec.clone(),
                    new_line_name: String::new(),
                    point_name: String::new(),
                    removed: None,
                    primary_dim: None,
                    secondary_dim: None,
                    added: vec![format!("FAILED: {}", e)],
                });
                continue;
            }
        };
        if idx == 0 && out.primary_dim.is_some() {
            primary_dim_name = out.primary_dim.clone();
        }
        outs.push(out);
    }

    let mut lines: Vec<String> = Vec::with_capacity(outs.len());
    for out in &outs {
        if out.new_line_name.is_empty() {
            lines.push(format!("  {}: {}", out.spec.join(" "), out.added.join(" ")));
            continue;
        }
        let mut parts = Vec::new();
        parts.push(out.new_line_name.clone());
        parts.push(out.point_name.clone());
        if let Some(d) = &out.primary_dim { parts.push(d.clone()); }
        if let Some(d) = &out.secondary_dim { parts.push(d.clone()); }
        let mut tail = Vec::new();
        if let Some(r) = &out.removed { tail.push(format!("removed {}", r)); }
        if !out.added.is_empty() { tail.push(format!("added {}", out.added.join(" "))); }
        let tail_str = if tail.is_empty() { String::new() } else { format!(" [{}]", tail.join(", ")) };
        lines.push(format!("  {} -> {}{}", out.spec.join(" "), parts.join(" "), tail_str));
    }
    let succeeded = outs.iter().filter(|o| !o.new_line_name.is_empty()).count();
    let header = if corner_specs.len() == 1 {
        format!("Chamfered (d={:.4}):", distance)
    } else {
        format!("Chamfered {} of {} corners (d={:.4}):",
            succeeded, corner_specs.len(), distance)
    };
    if let Some(last) = outs.iter().rev().find(|o| !o.new_line_name.is_empty()) {
        ctx.session_names.insert("_".into(), last.new_line_name.clone());
    }
    let msg = format!("{}\n{}", header, lines.join("\n"));
    if succeeded == 0 { err(msg) } else { ok(msg) }
}

struct ChamferOut {
    spec: Vec<String>,
    new_line_name: String,
    point_name: String,
    removed: Option<String>,
    primary_dim: Option<String>,
    secondary_dim: Option<String>,
    added: Vec<String>,
}

fn apply_one_chamfer(
    ctx: &mut CommandContext,
    spec: Vec<String>,
    refs: &CornerRefs,
    distance: f64,
    dist_expr: Option<String>,
) -> Result<ChamferOut, String> {
    let CornerRefs { line_a, is_p1_a, line_b, is_p1_b, coincident_id } = *refs;
    let la_ref = &ctx.sketch.lines[line_a];
    let lb_ref = &ctx.sketch.lines[line_b];
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

    let removed = crate::ids::constraint_id_name(&ctx.sketch, coincident_id);
    ctx.exec(Action::DeleteConstraint { id: coincident_id });

    if is_p1_a { ctx.sketch.get_mut().lines[line_a].p1.value = t_a; }
    else { ctx.sketch.get_mut().lines[line_a].p2.value = t_a; }
    if is_p1_b { ctx.sketch.get_mut().lines[line_b].p1.value = t_b; }
    else { ctx.sketch.get_mut().lines[line_b].p2.value = t_b; }

    let Some(point_ref) = ctx.exec(Action::AddPoint { pos: corner }).point() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    let point_name = ctx.sketch.points[point_ref].name.clone();

    let Some(new_line_ref) = ctx.exec(Action::AddLine { p1: t_a, p2: t_b }).line() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    let new_line_name = ctx.sketch.lines[new_line_ref].name.clone();

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
    let saved_skip = ctx.skip_dof_check;
    ctx.skip_dof_check = true;
    let ca = coincide_a.clone();
    ctx.exec(coincide_a);
    let ca_id = match ca {
        Action::ApplyCoincidentLL11 { .. } => ctx.sketch.coincident_ll11.last().map(|c| format!("C{}", c.nid)),
        Action::ApplyCoincidentLL21 { .. } => ctx.sketch.coincident_ll21.last().map(|c| format!("C{}", c.nid)),
        _ => None,
    };
    if let Some(n) = ca_id { added.push(n); }
    let cb = coincide_b.clone();
    ctx.exec(coincide_b);
    let cb_id = match cb {
        Action::ApplyCoincidentLL12 { .. } => ctx.sketch.coincident_ll12.last().map(|c| format!("C{}", c.nid)),
        Action::ApplyCoincidentLL22 { .. } => ctx.sketch.coincident_ll22.last().map(|c| format!("C{}", c.nid)),
        _ => None,
    };
    if let Some(n) = cb_id { added.push(n); }
    ctx.exec(Action::ApplyPointOnLine { point: point_ref, line: line_a });
    if let Some(c) = ctx.sketch.point_on_line.last() { added.push(format!("C{}", c.nid)); }
    ctx.exec(Action::ApplyPointOnLine { point: point_ref, line: line_b });
    if let Some(c) = ctx.sketch.point_on_line.last() { added.push(format!("C{}", c.nid)); }
    ctx.skip_dof_check = saved_skip;

    let ep_a = if is_p1_a { DimensionEndpoint::LineP1(line_a) } else { DimensionEndpoint::LineP2(line_a) };
    let ep_b = if is_p1_b { DimensionEndpoint::LineP1(line_b) } else { DimensionEndpoint::LineP2(line_b) };
    ctx.exec(Action::AddDimension {
        kind: DimensionKind::PointPointDistance(DimensionEndpoint::Point(point_ref), ep_a),
        value: distance, expr: dist_expr, derived: false, range: None,
    });
    let primary_dim = Some(last_dim_name(ctx));

    ctx.exec(Action::AddDimension {
        kind: DimensionKind::PointPointDistance(DimensionEndpoint::Point(point_ref), ep_b),
        value: distance, expr: primary_dim.clone(), derived: false, range: None,
    });
    let secondary_dim = Some(last_dim_name(ctx));

    Ok(ChamferOut {
        spec, new_line_name, point_name, removed, primary_dim, secondary_dim, added,
    })
}

fn cmd_offset_line(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: offset_line L0 distance"); }
    let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let d = match eval_expr(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let l = &ctx.sketch.lines[line];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 { return err("Zero-length line"); }
    let nx = -dy / len * d;
    let ny = dx / len * d;
    let p1 = vect2d::new(l.p1.value.x + nx, l.p1.value.y + ny);
    let p2 = vect2d::new(l.p2.value.x + nx, l.p2.value.y + ny);
    ctx.begin_group();
    ctx.exec(Action::AddLine { p1, p2 });
    let name = ctx.sketch.lines.refs().last().map(|r| ctx.sketch.lines[r].name.clone()).unwrap_or_default();
    ctx.cursor = Some(p2);
    ctx.session_names.insert("_".into(), name.clone());
    ok(format!("Added {} (offset of {} by {})", name, tokens[0], d))
}

// ---------------------------------------------------------------------------
// Additional constraints
// ---------------------------------------------------------------------------

fn cmd_midpoint(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: midpoint P0 L0 | midpoint L0.p1 A0"); }
    let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let target = tokens[1];
    // Target is a line or arc
    if let Ok(line) = resolve_line(&ctx.sketch, target) {
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.midpoint.iter().any(|c| c.point == p && c.line == line),
            EndpointRef::LineP1(l) => s.midpoint_lp1.iter().any(|c| c.line == l && c.target == line),
            EndpointRef::LineP2(l) => s.midpoint_lp2.iter().any(|c| c.line == l && c.target == line),
            EndpointRef::ArcStart(a) => s.midpoint_arc_start.iter().any(|c| c.arc == a && c.line == line),
            EndpointRef::ArcEnd(a) => s.midpoint_arc_end.iter().any(|c| c.arc == a && c.line == line),
            _ => false,
        };
        if exists { return err("Midpoint constraint already exists"); }
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyMidpoint { point: p, line },
            EndpointRef::LineP1(l) => Action::ApplyMidpointLP1 { line: l, target: line },
            EndpointRef::LineP2(l) => Action::ApplyMidpointLP2 { line: l, target: line },
            EndpointRef::ArcStart(a) => Action::ApplyMidpointArcStart { arc: a, line },
            EndpointRef::ArcEnd(a) => Action::ApplyMidpointArcEnd { arc: a, line },
            _ => return err("First arg must be a point or endpoint"),
        };
        ctx.begin_group();
        ctx.exec(action);
        let nid = ctx.sketch.next_constraint_id.saturating_sub(1);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied midpoint");
        ok_or_status(ctx, msg)
    } else if let Ok(arc) = resolve_arc(&ctx.sketch, target) {
        if ctx.sketch.arcs[arc].closed { return err("Cannot use midpoint on a full circle"); }
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.midpoint_arc_point.iter().any(|c| c.point == p && c.arc == arc),
            EndpointRef::LineP1(l) => s.midpoint_lp1_arc.iter().any(|c| c.line == l && c.arc == arc),
            EndpointRef::LineP2(l) => s.midpoint_lp2_arc.iter().any(|c| c.line == l && c.arc == arc),
            EndpointRef::ArcStart(a) => s.midpoint_arc_start_arc.iter().any(|c| c.a == a && c.b == arc),
            EndpointRef::ArcEnd(a) => s.midpoint_arc_end_arc.iter().any(|c| c.a == a && c.b == arc),
            _ => false,
        };
        if exists { return err("Midpoint constraint already exists"); }
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyMidpointArcPoint { point: p, arc },
            EndpointRef::LineP1(l) => Action::ApplyMidpointLP1Arc { line: l, arc },
            EndpointRef::LineP2(l) => Action::ApplyMidpointLP2Arc { line: l, arc },
            EndpointRef::ArcStart(a) => Action::ApplyMidpointArcStartArc { a, b: arc },
            EndpointRef::ArcEnd(a) => Action::ApplyMidpointArcEndArc { a, b: arc },
            _ => return err("First arg must be a point or endpoint"),
        };
        ctx.begin_group();
        ctx.exec(action);
        let nid = ctx.sketch.next_constraint_id.saturating_sub(1);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied midpoint");
        ok_or_status(ctx, msg)
    } else {
        err("Second arg must be a line (L0) or arc (A0)")
    }
}

/// EndpointRef -> DimensionEndpoint, the actions' endpoint currency.
fn to_dim_endpoint(ep: EndpointRef) -> DimensionEndpoint {
    match ep {
        EndpointRef::Point(p) => DimensionEndpoint::Point(p),
        EndpointRef::LineP1(l) => DimensionEndpoint::LineP1(l),
        EndpointRef::LineP2(l) => DimensionEndpoint::LineP2(l),
        EndpointRef::ArcCenter(a) => DimensionEndpoint::ArcCenter(a),
        EndpointRef::ArcStart(a) => DimensionEndpoint::ArcStart(a),
        EndpointRef::ArcEnd(a) => DimensionEndpoint::ArcEnd(a),
    }
}


fn cmd_symmetry(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 3 { return err("Usage: symmetry L0 L1 L2 | symmetry P0 L0 P1 | symmetry A0 L0 A1"); }
    // Try arc + line + arc symmetry
    if is_arc_name(tokens[0]) && is_arc_name(tokens[2])
        && let (Ok(a), Ok(line), Ok(c)) = (resolve_arc(&ctx.sketch, tokens[0]),
            resolve_line(&ctx.sketch, tokens[1]),
            resolve_arc(&ctx.sketch, tokens[2]))
        {
            if a == c { return err("Cannot constrain an arc symmetric with itself"); }
            if ctx.sketch.symmetry_aa.iter().any(|s|
                s.line == line && ((s.a == a && s.c == c) || (s.a == c && s.c == a))) {
                return err("Symmetry constraint already exists");
            }
            ctx.begin_group();
            ctx.exec(Action::ApplySymmetryAA { a, line, c });
            let nid = ctx.sketch.symmetry_aa.last().map(|c| c.nid).unwrap_or(0);
            let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied arc symmetry");
            return ok_or_status(ctx, msg);
        }
    // Try point/endpoint + line + point/endpoint symmetry
    let mid_is_line = resolve_line(&ctx.sketch, tokens[1]).is_ok();
    let first_is_pointlike = resolve_point(&ctx.sketch, tokens[0]).is_ok()
        || resolve_endpoint_ref(&ctx.sketch, tokens[0]).is_ok();
    let third_is_pointlike = resolve_point(&ctx.sketch, tokens[2]).is_ok()
        || resolve_endpoint_ref(&ctx.sketch, tokens[2]).is_ok();
    if mid_is_line && first_is_pointlike && third_is_pointlike {
        // Duplicate check is skipped for symmetry_pp -- the solver
        // handles redundancy gracefully.
        ctx.begin_group();
        let a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(e) => to_dim_endpoint(e), Err(e) => return err(e) };
        let line = resolve_line(&ctx.sketch, tokens[1]).unwrap();
        let c = match resolve_endpoint_ref(&ctx.sketch, tokens[2]) { Ok(e) => to_dim_endpoint(e), Err(e) => return err(e) };
        ctx.exec(Action::ApplySymmetryPP { a, line, c });
        let nid = ctx.sketch.symmetry_pp.last().map(|c| c.nid).unwrap_or(0);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied point symmetry");
        return ok_or_status(ctx, msg);
    }
    // Fall back to line-line-line symmetry
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let c = match resolve_line(&ctx.sketch, tokens[2]) { Ok(r) => r, Err(e) => return err(e) };
    if ctx.sketch.symmetry_ll.iter().any(|s|
        s.b == b && ((s.a == a && s.c == c) || (s.a == c && s.c == a))) {
        return err("Symmetry constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplySymmetryLL { a, b, c });
    let nid = ctx.sketch.symmetry_ll.last().map(|c| c.nid).unwrap_or(0);
    let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied symmetry");
    ok_or_status(ctx, msg)
}

/// Reflect a point across a line defined by two points.
fn mirror_point_across(pt: vect2d, lp1: vect2d, lp2: vect2d) -> vect2d {
    let dx = lp2.x - lp1.x;
    let dy = lp2.y - lp1.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-24 { return pt; }
    let t = ((pt.x - lp1.x) * dx + (pt.y - lp1.y) * dy) / len2;
    let proj = vect2d::new(lp1.x + t * dx, lp1.y + t * dy);
    vect2d::new(2.0 * proj.x - pt.x, 2.0 * proj.y - pt.y)
}

/// Source entity to be mirrored.
enum MirrorSource {
    Line(Ref<Line>),
    Point(Ref<Point>),
    Arc(Ref<Arc>),
}

fn cmd_mirror(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    // Find "about" keyword
    let about_pos = match tokens.iter().position(|&t| t == "about") {
        Some(p) => p,
        None => return err("Usage: mirror L0 L1 ... about L_axis [noconstraint] [strict]"),
    };
    if about_pos == 0 {
        return err("No entities to mirror");
    }

    // Parse trailing keywords after the mirror line
    let mut after_about: Vec<&str> = tokens[about_pos + 1..].to_vec();
    let [noconstraint, strict] = peel_keywords(&mut after_about, ["noconstraint", "strict"]);
    if noconstraint && strict {
        return err("noconstraint conflicts with strict");
    }
    if after_about.len() != 1 {
        return err("Expected exactly one mirror line after 'about'");
    }
    let mirror_line = match resolve_line(&ctx.sketch, after_about[0]) { Ok(r) => r, Err(e) => return err(e) };

    // Collect source entities
    let source_tokens = &tokens[..about_pos];
    let mut sources: Vec<MirrorSource> = Vec::new();
    if source_tokens.len() == 1 && source_tokens[0] == "selection" {
        for sel in &ctx.selection {
            match sel {
                Selection::Line(r) => {
                    if !sources.iter().any(|s| matches!(s, MirrorSource::Line(l) if *l == *r)) {
                        sources.push(MirrorSource::Line(*r));
                    }
                }
                Selection::Arc(r) => {
                    if !sources.iter().any(|s| matches!(s, MirrorSource::Arc(a) if *a == *r)) {
                        sources.push(MirrorSource::Arc(*r));
                    }
                }
                Selection::Point(r) => {
                    if !sources.iter().any(|s| matches!(s, MirrorSource::Point(p) if *p == *r)) {
                        sources.push(MirrorSource::Point(*r));
                    }
                }
                _ => {} // skip endpoint selections, constraints, dimensions
            }
        }
        if sources.is_empty() {
            return err("No lines, arcs, or points in selection");
        }
    } else {
        for &name in source_tokens {
            if name.starts_with('L') {
                let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
                if r == mirror_line { return err(format!("Cannot mirror {} about itself", name)); }
                sources.push(MirrorSource::Line(r));
            } else if is_arc_name(name) {
                let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
                sources.push(MirrorSource::Arc(r));
            } else if name.starts_with('P') {
                let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
                sources.push(MirrorSource::Point(r));
            } else {
                return err(format!("Unknown entity: {}", name));
            }
        }
    }

    let ml = &ctx.sketch.lines[mirror_line];
    let mlp1 = ml.p1.value;
    let mlp2 = ml.p2.value;

    ctx.begin_group();
    let mut warnings = Vec::new();
    let mut applied = Vec::new();
    let mut msgs = Vec::new();

    // Maps from source to mirrored refs
    let mut line_map: Vec<(Ref<Line>, Ref<Line>)> = Vec::new();
    let mut point_map: Vec<(Ref<Point>, Ref<Point>)> = Vec::new();
    let mut arc_map: Vec<(Ref<Arc>, Ref<Arc>)> = Vec::new();
    let mut mirrored_names: Vec<String> = Vec::new();

    for source in &sources {
        match source {
            MirrorSource::Line(src_ref) => {
                let l = &ctx.sketch.lines[*src_ref];
                let src_name = l.name.clone();
                let mp1 = mirror_point_across(l.p1.value, mlp1, mlp2);
                let mp2 = mirror_point_across(l.p2.value, mlp1, mlp2);
                let Some(new_ref) = ctx.exec(Action::AddLine { p1: mp1, p2: mp2 }).line() else {
                    return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
                };
                let new_name = ctx.sketch.lines[new_ref].name.clone();
                line_map.push((*src_ref, new_ref));
                msgs.push(format!("Mirrored {} -> {}", src_name, new_name));
                mirrored_names.push(new_name);
            }
            MirrorSource::Point(src_ref) => {
                let p = &ctx.sketch.points[*src_ref];
                let src_name = p.name.clone();
                let mp = mirror_point_across(p.pos.value, mlp1, mlp2);
                let Some(new_ref) = ctx.exec(Action::AddPoint { pos: mp }).point() else {
                    return err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()));
                };
                let new_name = ctx.sketch.points[new_ref].name.clone();
                point_map.push((*src_ref, new_ref));
                msgs.push(format!("Mirrored {} -> {}", src_name, new_name));
                mirrored_names.push(new_name);
            }
            MirrorSource::Arc(src_ref) => {
                let a = &ctx.sketch.arcs[*src_ref];
                let src_name = a.name.clone();
                let mc = mirror_point_across(a.center.value, mlp1, mlp2);
                let r = a.radius.value;
                let created = if a.closed {
                    let edge = vect2d::new(mc.x + r, mc.y);
                    ctx.exec(Action::AddCircle { center: mc, edge })
                } else {
                    let ms = mirror_point_across(arc_start_pos(a), mlp1, mlp2);
                    let me = mirror_point_across(arc_end_pos(a), mlp1, mlp2);
                    // Compute a mid-point on the arc and mirror it
                    let mid_angle = (a.start_angle.value + a.end_angle.value) / 2.0;
                    let mid_pt = vect2d::new(
                        a.center.value.x + r * mid_angle.cos(),
                        a.center.value.y + r * mid_angle.sin(),
                    );
                    let mm = mirror_point_across(mid_pt, mlp1, mlp2);
                    ctx.exec(Action::AddArc { start: ms, end: me, mid: mm })
                };
                let Some(new_ref) = created.arc() else {
                    return err("Cannot mirror arc: degenerate mirrored geometry");
                };
                let new_name = ctx.sketch.arcs[new_ref].name.clone();
                arc_map.push((*src_ref, new_ref));
                msgs.push(format!("Mirrored {} -> {}", src_name, new_name));
                mirrored_names.push(new_name);
            }
        }
    }

    if !noconstraint {
        // Phase 2: Recreate coincident constraints among mirrored copies
        // Snapshot all relevant coincident pairs, then apply
        let mut coinc_actions: Vec<(Action, String)> = Vec::new();
        // Helper to look up mirrored line ref
        let find_ml = |src: Ref<Line>| line_map.iter().find(|(s, _)| *s == src).map(|(_, m)| *m);
        let find_mp = |src: Ref<Point>| point_map.iter().find(|(s, _)| *s == src).map(|(_, m)| *m);
        // Line-line coincidents
        macro_rules! scan_ll {
            ($field:ident, $action:ident, $ep_a:expr, $ep_b:expr) => {
                for c in &ctx.sketch.$field {
                    if let (Some(ma), Some(mb)) = (find_ml(c.a), find_ml(c.b)) {
                        let desc = format!("coincident {}.{}={}.{}",
                            ctx.sketch.lines[ma].name, $ep_a, ctx.sketch.lines[mb].name, $ep_b);
                        coinc_actions.push((Action::$action { a: ma, b: mb }, desc));
                    }
                }
            };
        }
        scan_ll!(coincident_ll11, ApplyCoincidentLL11, "p1", "p1");
        scan_ll!(coincident_ll12, ApplyCoincidentLL12, "p1", "p2");
        scan_ll!(coincident_ll21, ApplyCoincidentLL21, "p2", "p1");
        scan_ll!(coincident_ll22, ApplyCoincidentLL22, "p2", "p2");
        // Point-point coincidents
        for c in &ctx.sketch.coincident_pp {
            if let (Some(ma), Some(mb)) = (find_mp(c.a), find_mp(c.b)) {
                let desc = format!("coincident {} {}",
                    ctx.sketch.points[ma].name, ctx.sketch.points[mb].name);
                coinc_actions.push((Action::ApplyCoincidentPP { a: ma, b: mb }, desc));
            }
        }
        // Line-point coincidents (LP1, LP2)
        for c in &ctx.sketch.coincident_lp1 {
            if let (Some(ml), Some(mp)) = (find_ml(c.line), find_mp(c.point)) {
                let desc = format!("coincident {}.p1 {}",
                    ctx.sketch.lines[ml].name, ctx.sketch.points[mp].name);
                coinc_actions.push((Action::ApplyCoincidentLP1 { line: ml, point: mp }, desc));
            }
        }
        for c in &ctx.sketch.coincident_lp2 {
            if let (Some(ml), Some(mp)) = (find_ml(c.line), find_mp(c.point)) {
                let desc = format!("coincident {}.p2 {}",
                    ctx.sketch.lines[ml].name, ctx.sketch.points[mp].name);
                coinc_actions.push((Action::ApplyCoincidentLP2 { line: ml, point: mp }, desc));
            }
        }
        // Apply collected coincident actions
        for (action, desc) in coinc_actions {
            if let Err(e) = rect_exec(ctx, action, strict, &desc, &mut applied, &mut warnings) {
                return err(e);
            }
        }

        // Phase 3: Collect symmetry constraint info (snapshot positions before mutating)
        struct SymEntry { src_ep: String, dst_ep: String, pos: vect2d }
        let mut sym_entries: Vec<SymEntry> = Vec::new();
        for &(src, dst) in &line_map {
            let sp1 = ctx.sketch.lines[src].p1.value;
            let sp2 = ctx.sketch.lines[src].p2.value;
            let src_name = ctx.sketch.lines[src].name.clone();
            let dst_name = ctx.sketch.lines[dst].name.clone();
            sym_entries.push(SymEntry { src_ep: format!("{}.p1", src_name), dst_ep: format!("{}.p1", dst_name), pos: sp1 });
            sym_entries.push(SymEntry { src_ep: format!("{}.p2", src_name), dst_ep: format!("{}.p2", dst_name), pos: sp2 });
        }
        for &(src, dst) in &point_map {
            let sp = ctx.sketch.points[src].pos.value;
            let src_name = ctx.sketch.points[src].name.clone();
            let dst_name = ctx.sketch.points[dst].name.clone();
            sym_entries.push(SymEntry { src_ep: src_name, dst_ep: dst_name, pos: sp });
        }
        for &(src, dst) in &arc_map {
            let sc = ctx.sketch.arcs[src].center.value;
            let src_name = ctx.sketch.arcs[src].name.clone();
            let dst_name = ctx.sketch.arcs[dst].name.clone();
            sym_entries.push(SymEntry { src_ep: format!("{}.center", src_name), dst_ep: format!("{}.center", dst_name), pos: sc });
            if !ctx.sketch.arcs[src].closed {
                let ss = arc_start_pos(&ctx.sketch.arcs[src]);
                let se = arc_end_pos(&ctx.sketch.arcs[src]);
                sym_entries.push(SymEntry { src_ep: format!("{}.start", src_name), dst_ep: format!("{}.start", dst_name), pos: ss });
                sym_entries.push(SymEntry { src_ep: format!("{}.end", src_name), dst_ep: format!("{}.end", dst_name), pos: se });
            }
        }
        // Apply symmetry with dedup
        let mut constrained_positions: Vec<vect2d> = Vec::new();
        for entry in &sym_entries {
            if constrained_positions.iter().any(|p| (p.x - entry.pos.x).abs() < 1e-6 && (p.y - entry.pos.y).abs() < 1e-6) {
                continue;
            }
            constrained_positions.push(entry.pos);
            let a = match resolve_endpoint_ref(&ctx.sketch, &entry.src_ep) {
                Ok(e) => to_dim_endpoint(e),
                Err(e) => { warnings.push(format!("symmetry {}: {}", entry.src_ep, e)); continue }
            };
            let c = match resolve_endpoint_ref(&ctx.sketch, &entry.dst_ep) {
                Ok(e) => to_dim_endpoint(e),
                Err(e) => { warnings.push(format!("symmetry {}: {}", entry.dst_ep, e)); continue }
            };
            let desc = format!("symmetry {} {} {}", entry.src_ep, after_about[0], entry.dst_ep);
            if let Err(e) = rect_exec(ctx, Action::ApplySymmetryPP { a, line: mirror_line, c }, strict, &desc, &mut applied, &mut warnings)
                && strict { return err(e); }
        }
    }

    // Set session names
    if let Some(first) = mirrored_names.first() {
        ctx.session_names.insert("_".into(), first.clone());
    }
    for (i, name) in mirrored_names.iter().enumerate() {
        ctx.session_names.insert(format!("_{}", i), name.clone());
    }

    let mut msg = msgs.join("\n");
    for a in &applied {
        msg += &format!("\n  {}", a);
    }
    for w in &warnings {
        msg += &format!("\n  warning: {}", w);
    }
    ok(msg)
}

/// Which arc endpoint a helper point bridges to.
enum ArcEp { Center, Start, End }

/// Check if an arc endpoint already has a point_on_line constraint via a helper point.
fn has_arc_endpoint_on_line(s: &Sketch, arc: Ref<Arc>, ep: ArcEp, line: Ref<Line>) -> bool {
    // Find helper points bridged to this arc endpoint
    let bridged_points: Vec<Ref<Point>> = match ep {
        ArcEp::Center => s.coincident_arc_center.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::Start => s.coincident_arc_start.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::End => s.coincident_arc_end.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
    };
    // Check if any of those points are on the target line
    bridged_points.iter().any(|p| s.point_on_line.iter().any(|c| c.point == *p && c.line == line))
}

/// Check if an arc endpoint already has a point_on_arc constraint via a helper point.
fn has_arc_endpoint_on_arc(s: &Sketch, src: Ref<Arc>, ep: ArcEp, target: Ref<Arc>) -> bool {
    let bridged_points: Vec<Ref<Point>> = match ep {
        ArcEp::Center => s.coincident_arc_center.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::Start => s.coincident_arc_start.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::End => s.coincident_arc_end.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
    };
    bridged_points.iter().any(|p| s.point_on_arc.iter().any(|c| c.point == *p && c.arc == target))
}

fn cmd_point_on(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: point_on P0 L0  or  point_on L0.p1 A0"); }
    let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let target = tokens[1];
    if target.starts_with('L') {
        let line = match resolve_line(&ctx.sketch, target) { Ok(r) => r, Err(e) => return err(e) };
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.point_on_line.iter().any(|c| c.point == p && c.line == line),
            EndpointRef::LineP1(l) => s.line_p1_on_line.iter().any(|c| c.a == l && c.b == line),
            EndpointRef::LineP2(l) => s.line_p2_on_line.iter().any(|c| c.a == l && c.b == line),
            EndpointRef::ArcCenter(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::Center, line),
            EndpointRef::ArcStart(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::Start, line),
            EndpointRef::ArcEnd(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::End, line),
        };
        if exists { return err("Point-on-line constraint already exists"); }
        ctx.begin_group();
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyPointOnLine { point: p, line },
            EndpointRef::LineP1(l) => Action::ApplyLineP1OnLine { a: l, b: line },
            EndpointRef::LineP2(l) => Action::ApplyLineP2OnLine { a: l, b: line },
            EndpointRef::ArcCenter(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcCenter(a), line },
            EndpointRef::ArcStart(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcStart(a), line },
            EndpointRef::ArcEnd(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcEnd(a), line },
        };
        ctx.exec(action);
        let nid = ctx.sketch.next_constraint_id.saturating_sub(1);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied point-on-line");
        ok_or_status(ctx, msg)
    } else if is_arc_name(target) {
        let arc = match resolve_arc(&ctx.sketch, target) { Ok(r) => r, Err(e) => return err(e) };
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.point_on_arc.iter().any(|c| c.point == p && c.arc == arc),
            EndpointRef::LineP1(l) => s.line_p1_on_arc.iter().any(|c| c.line == l && c.arc == arc),
            EndpointRef::LineP2(l) => s.line_p2_on_arc.iter().any(|c| c.line == l && c.arc == arc),
            EndpointRef::ArcCenter(src) => has_arc_endpoint_on_arc(s, src, ArcEp::Center, arc),
            EndpointRef::ArcStart(src) => has_arc_endpoint_on_arc(s, src, ArcEp::Start, arc),
            EndpointRef::ArcEnd(src) => has_arc_endpoint_on_arc(s, src, ArcEp::End, arc),
        };
        if exists { return err("Point-on-arc constraint already exists"); }
        ctx.begin_group();
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyPointOnArc { point: p, arc },
            EndpointRef::LineP1(l) => Action::ApplyLineP1OnArc { line: l, arc },
            EndpointRef::LineP2(l) => Action::ApplyLineP2OnArc { line: l, arc },
            EndpointRef::ArcCenter(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcCenter(a), arc },
            EndpointRef::ArcStart(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcStart(a), arc },
            EndpointRef::ArcEnd(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcEnd(a), arc },
        };
        ctx.exec(action);
        let nid = ctx.sketch.next_constraint_id.saturating_sub(1);
        let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), "Applied point-on-arc");
        ok_or_status(ctx, msg)
    } else {
        err("Second arg must be a line (L0) or arc (A0)")
    }
}

// ---------------------------------------------------------------------------
// Additional dimensions
// ---------------------------------------------------------------------------

fn cmd_angle(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    // Peel optional trailing keywords in any order (driven/derived + sector)
    #[derive(Clone, Copy)]
    enum SectorMode { Default, Supplement, Closest, Acute, Obtuse }
    let [is_derived, is_driven, supplement, closest, acute, obtuse] = peel_keywords(&mut tokens,
        ["derived", "driven", "supplement", "closest", "acute", "obtuse"]);
    let sector_mode = if supplement { SectorMode::Supplement }
        else if closest { SectorMode::Closest }
        else if acute { SectorMode::Acute }
        else if obtuse { SectorMode::Obtuse }
        else { SectorMode::Default };

    // Compute current angle between direction vectors (p1->p2)
    let compute_angle = |ctx: &CommandContext, a_ref, b_ref| -> (f64, f64) {
        let la = &ctx.sketch.lines[a_ref];
        let lb = &ctx.sketch.lines[b_ref];
        let dx1 = la.p2.value.x - la.p1.value.x;
        let dy1 = la.p2.value.y - la.p1.value.y;
        let dx2 = lb.p2.value.x - lb.p1.value.x;
        let dy2 = lb.p2.value.y - lb.p1.value.y;
        let cross = dx1 * dy2 - dy1 * dx2;
        let dot = dx1 * dx2 + dy1 * dy2;
        let current_deg = cross.atan2(dot).to_degrees().abs();
        (current_deg, 180.0 - current_deg)
    };

    if tokens.len() == 2 && (is_derived || is_driven) {
        let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let (current_deg, supplement_deg) = compute_angle(ctx, a, b);
        let supplement = match sector_mode {
            SectorMode::Supplement => true,
            SectorMode::Acute => current_deg > supplement_deg,
            SectorMode::Obtuse => current_deg <= supplement_deg,
            _ => false,
        };
        let val = if supplement { supplement_deg } else { current_deg };
        let kind = DimensionKind::Angle(a, b, supplement);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr: None, range: None,  });
            let label = if is_derived { "derived" } else { "driven" };
            return ok_or_status(ctx, format!("Updated {} {} angle = ({:.4})", label, name, val));
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} angle {} {} = ({:.4})", label, dim_name, tokens[0], tokens[1], val));
    }

    // Range form: `angle L0 L1 >= V | <= V | LO to HI [supplement]`.
    // `closest` / `acute` / `obtuse` are value-selection heuristics
    // and are rejected with a range (no single target value).
    let range_opt = if tokens.len() >= 3 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[2..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        if matches!(sector_mode, SectorMode::Closest | SectorMode::Acute | SectorMode::Obtuse) {
            return err("closest / acute / obtuse require a specific target angle; use a bare value or `supplement`");
        }
        let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let (current_deg, supplement_deg) = compute_angle(ctx, a, b);
        let supplement = matches!(sector_mode, SectorMode::Supplement);
        let measured = if supplement { supplement_deg } else { current_deg };
        let kind = DimensionKind::Angle(a, b, supplement);
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        let sector = if supplement { " (supplement)" } else { "" };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} angle {}{} (current {:.4})", name, bound_desc, sector, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} angle {} {} {}{} (current {:.4})",
            dim_name, tokens[0], tokens[1], bound_desc, sector, measured));
    }

    if tokens.len() != 3 { return err("Usage: angle L0 L1 45 [supplement|closest|acute|obtuse] [derived|driven]  or  angle L0 L1 >=V | <=V | LO to HI [supplement]"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let (value, expr) = match parse_dim_value(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };
    // Accept negative values (e.g. from angle() function) by taking absolute value
    let value = value.abs();
    let display = if expr.is_some() { tokens[2].to_string() } else { format!("{}", value) };
    let (current_deg, supplement_deg) = compute_angle(ctx, a, b);
    let check_val = if expr.is_some() { eval_expr(&ctx.sketch, expr.as_ref().unwrap()).unwrap_or(value).abs() } else { value };
    let supplement = match sector_mode {
        SectorMode::Default => false,
        SectorMode::Supplement => true,
        SectorMode::Closest => (check_val - supplement_deg).abs() < (check_val - current_deg).abs(),
        SectorMode::Acute => current_deg > supplement_deg,
        SectorMode::Obtuse => current_deg <= supplement_deg,
    };
    let kind = DimensionKind::Angle(a, b, supplement);
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value, expr, range: None,  });
        let sector = if supplement { "supplement" } else { "" };
        return ok_or_status(ctx, format!("Updated {} angle = {} {}", name, display, sector).trim_end().to_string());
    }
    ctx.exec(Action::AddDimension { kind, value, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let sector = if supplement { " (supplement)" } else { "" };
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} angle {} {} = {}{}", prefix, dim_name, tokens[0], tokens[1], display, sector))
}

/// Parse a single range-value token: numeric literal, `=expr` /
/// `{expr}` live expression (validated), or a bare expression that
/// evaluates now and is stored as a literal. Mirrors
/// `parse_dim_value` with the result typed as `RangeValue` rather
/// than `(f64, Option<String>)`.
pub(crate) fn parse_range_value(sketch: &Sketch, token: &str) -> Result<RangeValue, String> {
    let token = token.trim();
    // Snapshot form: `=expr` evaluates now and becomes a literal.
    if let Some(expr) = token.strip_prefix('=') {
        return match eval_expr(sketch, expr.trim()) {
            Ok(v) => Ok(RangeValue::Literal(v)),
            Err(e) => Err(format!("Cannot evaluate snapshot '{}': {}", expr, e)),
        };
    }
    if let Ok(v) = token.parse::<f64>() {
        return Ok(RangeValue::Literal(v));
    }
    // Anything else: live expression.
    arael_sym::parse(token).map_err(|e|
        format!("Cannot parse range value '{}': {}", token, e))?;
    Ok(RangeValue::Live(token.to_string()))
}

/// Parse `>= V`, `<= V`, `>=V`, `<=V`, or `LO to HI` shapes from the
/// trailing tokens of a dimension command. Each side is a single
/// value token per the existing "Expression syntax" grammar
/// (numeric, evaluate-once expression, or live via `=` / `{}`).
/// Returns None if the tokens don't match any recognised range
/// syntax; returns Err if they do but one of the values fails to
/// parse.
pub(crate) fn parse_range_tokens(sketch: &Sketch, tokens: &[&str]) -> Result<Option<RangeBound>, String> {
    if tokens.is_empty() { return Ok(None); }
    // >=V or >= V
    if tokens[0].starts_with(">=") {
        let rest = &tokens[0][2..];
        let v_tok = if rest.is_empty() {
            if tokens.len() != 2 { return Ok(None); }
            tokens[1]
        } else {
            if tokens.len() != 1 { return Ok(None); }
            rest
        };
        return Ok(Some(RangeBound::Min(parse_range_value(sketch, v_tok)?)));
    }
    // <=V or <= V
    if tokens[0].starts_with("<=") {
        let rest = &tokens[0][2..];
        let v_tok = if rest.is_empty() {
            if tokens.len() != 2 { return Ok(None); }
            tokens[1]
        } else {
            if tokens.len() != 1 { return Ok(None); }
            rest
        };
        return Ok(Some(RangeBound::Max(parse_range_value(sketch, v_tok)?)));
    }
    // `LO to HI`
    if tokens.len() == 3 && tokens[1] == "to" {
        let lo = parse_range_value(sketch, tokens[0])?;
        let hi = parse_range_value(sketch, tokens[2])?;
        return Ok(Some(RangeBound::Between(lo, hi)));
    }
    Ok(None)
}

/// GUI wrapper: accept the raw text the user typed into the
/// dimension-value input and try to parse it as a range bound.
/// Returns `Ok(None)` if the input isn't a range (caller should
/// fall through to numeric / live-expression parsing), `Ok(Some)`
/// on a successful range match, `Err` on a match with a malformed
/// value.
pub fn parse_range_input(sketch: &Sketch, input: &str) -> Result<Option<RangeBound>, String> {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    parse_range_tokens(sketch, &tokens)
}

fn cmd_distance(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }


    // "distance L0.p1 L1.p2 derived/driven" or "distance P0 L0 derived/driven" — measure-only, no value
    if tokens.len() == 2 && (is_derived || is_driven) {
        let label = if is_derived { "Derived" } else { "Driven" };
        // Try point-line distance first
        if (tokens[0].starts_with('P') || tokens[0].contains('.')) && tokens[1].starts_with('L') && !tokens[1].contains('.') {
            let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let line = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let p = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
            let l = &ctx.sketch.lines[line];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            let dist = if len < 1e-12 { 0.0 } else { ((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx).abs() / len };
            let kind = DimensionKind::PointLineDistance(to_dim_endpoint(ep), line);
            ctx.begin_group();
            ctx.exec(Action::AddDimension { kind, value: dist, expr: None, derived: is_derived, range: None,  });
            let dim_name = last_dim_name(ctx);
            return ok_or_status(ctx, format!("{} {} distance {} {} = ({:.4})", label, dim_name, tokens[0], tokens[1], dist));
        }
        // Point-point distance
        let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let pa = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
        let pb = resolve_endpoint_pos(&ctx.sketch, tokens[1]).unwrap();
        let dx = pa.x - pb.x; let dy = pa.y - pb.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let kind = DimensionKind::PointPointDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b));
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: dist, expr: None, range: None,  });
            let ulabel = if is_derived { "derived" } else { "driven" };
            return ok_or_status(ctx, format!("Updated {} {} distance = ({:.4})", name, ulabel, dist));
        }
        ctx.exec(Action::AddDimension { kind, value: dist, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("{} {} distance {} {} = ({:.4})", label, dim_name, tokens[0], tokens[1], dist));
    }

    // Range-dimension form: two entities + one of `>=V`, `<=V`,
    // `>= V`, `<= V`, or `LO to HI`. Each value is a numeric
    // literal, evaluate-once expression, or live expression
    // (`=expr` / `{expr}`). Incompatible with derived/driven (a
    // range isn't a measure-only or value-capture dimension).
    let range_opt = if tokens.len() >= 3 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[2..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        // Resolve entities and pick a kind. For two-line / two-arc
        // shapes, also capture an optional pairing constraint to emit
        // up front (Parallel for LineLineDistance, Concentric for
        // ConcentricDistance).
        let (kind, measured, parallel_emit, concentric_emit): (
            DimensionKind, f64, Option<(Ref<Line>, Ref<Line>)>, Option<(Ref<Arc>, Ref<Arc>)>,
        ) = {
            // Two bare lines -> LineLineDistance
            if !tokens[0].contains('.') && !tokens[1].contains('.')
                && tokens[0].starts_with('L') && tokens[1].starts_with('L')
            {
                let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let la = &ctx.sketch.lines[a];
                let lb = &ctx.sketch.lines[b];
                let dx = la.p2.value.x - la.p1.value.x;
                let dy = la.p2.value.y - la.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let measured = if len < 1e-12 { 0.0 } else {
                    ((lb.p1.value.x - la.p1.value.x) * dy
                   - (lb.p1.value.y - la.p1.value.y) * dx).abs() / len
                };
                let already_parallel = ctx.sketch.parallel.iter().any(|p|
                    (p.a == a && p.b == b) || (p.a == b && p.b == a));
                let emit = if already_parallel { None } else { Some((a, b)) };
                (DimensionKind::LineLineDistance(a, b), measured, emit, None)
            }
            // Point + Line -> PointLineDistance
            else if (tokens[0].starts_with('P') || tokens[0].contains('.'))
                && tokens[1].starts_with('L') && !tokens[1].contains('.')
            {
                let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let line = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let p = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
                let l = &ctx.sketch.lines[line];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let measured = if len < 1e-12 { 0.0 } else { ((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx).abs() / len };
                (DimensionKind::PointLineDistance(to_dim_endpoint(ep), line), measured, None, None)
            }
            // Geometrically-concentric arcs -> ConcentricDistance. The
            // dimension enforces its own center-coincidence, so an
            // explicit `Concentric` isn't required to place it -- we
            // still emit one up front for visibility in `list`. The
            // caller-emit pattern mirrors LineLineDistance + Parallel.
            else if !tokens[0].contains('.') && !tokens[1].contains('.')
                && tokens[0].starts_with('A') && tokens[1].starts_with('A')
                && let Ok(arc_a) = resolve_arc(&ctx.sketch, tokens[0])
                && let Ok(arc_b) = resolve_arc(&ctx.sketch, tokens[1])
                && arc_a != arc_b
                && !ctx.sketch.arcs[arc_a].is_ellipse
                && !ctx.sketch.arcs[arc_b].is_ellipse
                && arcs_are_concentric(&ctx.sketch, arc_a, arc_b)
            {
                let ra = ctx.sketch.arcs[arc_a].radius.value;
                let rb2 = ctx.sketch.arcs[arc_b].radius.value;
                let already_concentric = ctx.sketch.concentric.iter().any(|c|
                    (c.a == arc_a && c.b == arc_b) || (c.a == arc_b && c.b == arc_a));
                let emit = if already_concentric { None } else { Some((arc_a, arc_b)) };
                (DimensionKind::ConcentricDistance(arc_a, arc_b), (rb2 - ra).abs(), None, emit)
            }
            // Default: PointPointDistance
            else {
                let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let pa = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
                let pb = resolve_endpoint_pos(&ctx.sketch, tokens[1]).unwrap();
                let dx = pa.x - pb.x; let dy = pa.y - pb.y;
                (DimensionKind::PointPointDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)),
                 (dx * dx + dy * dy).sqrt(), None, None)
            }
        };
        ctx.begin_group();
        if let Some((a, b)) = parallel_emit {
            ctx.exec(Action::ApplyParallel { a, b });
        }
        if let Some((a, b)) = concentric_emit {
            ctx.exec(Action::ApplyConcentric { a, b });
        }
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!(
            "Set {} distance {} {} {} (current {:.4})",
            dim_name, tokens[0], tokens[1], bound_desc, measured));
    }

    if tokens.len() != 3 { return err("Usage: distance L0.p1 L1.p2 5.0 [derived|driven]  or  distance P0 L0 3.0 [derived|driven]  or  distance A0 A1 5.0 (concentric circles)  or  distance L0 L1 5.0 (two lines; also applies a Parallel constraint)  or  distance <entities> >=V | <=V | LO to HI (range bound)"); }
    let (val, expr) = match parse_dim_value(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };

    // Two-line perpendicular distance: `distance L0 L1 v` for two bare
    // line names. Also applies a Parallel constraint between them -- the
    // gap between non-parallel lines is ill-defined, so the CLI makes
    // the pairing explicit. Idempotent when Parallel is already there.
    if !tokens[0].contains('.') && !tokens[1].contains('.')
        && tokens[0].starts_with('L') && tokens[1].starts_with('L')
    {
        let line_a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let line_b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let kind = DimensionKind::LineLineDistance(line_a, line_b);
        ctx.begin_group();
        // "Already parallel" means the two lines are currently
        // geometrically parallel -- whether via an explicit Parallel,
        // matching H/V flags, or just happening to align. In all
        // three cases a fresh ApplyParallel would be DOF-rejected or
        // redundant, so skip it; the dim is also placeable without
        // a paired Parallel since the action-side guard is purely
        // geometric. If the lines aren't parallel yet, add Parallel
        // first so the solver brings them into alignment before
        // trying to satisfy the gap.
        let la_line = &ctx.sketch.lines[line_a];
        let lb_line = &ctx.sketch.lines[line_b];
        let ax = la_line.p2.value.x - la_line.p1.value.x;
        let ay = la_line.p2.value.y - la_line.p1.value.y;
        let bx = lb_line.p2.value.x - lb_line.p1.value.x;
        let by = lb_line.p2.value.y - lb_line.p1.value.y;
        let alen = (ax * ax + ay * ay).sqrt();
        let blen = (bx * bx + by * by).sqrt();
        let already_parallel = alen > 1e-12 && blen > 1e-12
            && (ax * by - ay * bx).abs() / (alen * blen) < 1e-6;
        if !already_parallel {
            ctx.exec(Action::ApplyParallel { a: line_a, b: line_b });
        }
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
            return ok_or_status(ctx, format!("Updated {} line-line distance = {}", name, tokens[2]));
        }
        // When the pair is already parallel, the dim is unambiguously
        // a new gap constraint. At axis-aligned configs the Jacobian
        // row of a new LineLineDistance can be instantaneously
        // tangent-aligned with an existing H/Tangent row, so the
        // rank check misses the genuine DOF reduction. Skip the DOF
        // check here so the user isn't asked to 'force' a constraint
        // we know is real.
        let saved_skip = ctx.skip_dof_check;
        if already_parallel { ctx.skip_dof_check = true; }
        ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
        ctx.skip_dof_check = saved_skip;
        let dim_name = last_dim_name(ctx);
        let prefix = if is_derived { "Derived" } else { "Set" };
        return ok_or_status(ctx, format!("{} {} line-line distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2]));
    }

    // Concentric-arcs radial distance: `distance A0 A1 v` for two
    // distinct non-ellipse circles whose centers coincide (within
    // epsilon). The dim is self-contained: it enforces both
    // center-coincidence and radial gap, so it can be placed on any
    // geometrically-concentric pair without a pre-existing
    // `Concentric` constraint. For visibility in `list`, also emit
    // `ApplyConcentric` up front when one isn't already there, same
    // pattern as LineLineDistance + Parallel.
    if !tokens[0].contains('.') && !tokens[1].contains('.')
        && tokens[0].starts_with('A') && tokens[1].starts_with('A')
        && let Ok(arc_a) = resolve_arc(&ctx.sketch, tokens[0])
        && let Ok(arc_b) = resolve_arc(&ctx.sketch, tokens[1])
        && arc_a != arc_b
        && !ctx.sketch.arcs[arc_a].is_ellipse
        && !ctx.sketch.arcs[arc_b].is_ellipse
        && arcs_are_concentric(&ctx.sketch, arc_a, arc_b)
    {
        let kind = DimensionKind::ConcentricDistance(arc_a, arc_b);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
            return ok_or_status(ctx, format!("Updated {} concentric distance = {}", name, tokens[2]));
        }
        let already_concentric = ctx.sketch.concentric.iter().any(|c|
            (c.a == arc_a && c.b == arc_b) || (c.a == arc_b && c.b == arc_a));
        if !already_concentric {
            ctx.exec(Action::ApplyConcentric { a: arc_a, b: arc_b });
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let prefix = if is_derived { "Derived" } else { "Set" };
        return ok_or_status(ctx, format!("{} {} concentric distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2]));
    }

    // Try point-line distance
    if (tokens[0].starts_with('P') || tokens[0].contains('.')) && tokens[1].starts_with('L') && !tokens[1].contains('.') {
        let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let line = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let kind = DimensionKind::PointLineDistance(to_dim_endpoint(ep), line);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
            return ok_or_status(ctx, format!("Updated {} distance = {}", name, tokens[2]));
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let prefix = if is_derived { "Derived" } else { "Set" };
        return ok_or_status(ctx, format!("{} {} distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2]));
    }

    // Point-point distance
    let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = DimensionKind::PointPointDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b));
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} distance = {}", name, tokens[2]));
    }
    ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2]))
}

fn cmd_hdistance(ctx: &mut CommandContext, args: &str) -> CommandResult {
    cmd_axis_distance(ctx, args, true)
}

fn cmd_vdistance(ctx: &mut CommandContext, args: &str) -> CommandResult {
    cmd_axis_distance(ctx, args, false)
}

fn cmd_axis_distance(ctx: &mut CommandContext, args: &str, horizontal: bool) -> CommandResult {
    let label = if horizontal { "hdistance" } else { "vdistance" };
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }


    // "hdistance L0.p1 L1.p2 derived" — measure-only
    if tokens.len() == 2 && (is_derived || is_driven) {
        let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let pa = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
        let pb = resolve_endpoint_pos(&ctx.sketch, tokens[1]).unwrap();
        let measured = if horizontal { (pa.x - pb.x).abs() } else { (pa.y - pb.y).abs() };
        let kind = if horizontal { DimensionKind::HDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) }
                   else { DimensionKind::VDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) };
        ctx.begin_group();
        ctx.exec(Action::AddDimension { kind, value: measured, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let prefix = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} {} {} {} = ({:.4})", prefix, dim_name, label, tokens[0], tokens[1], measured));
    }

    // Range form: `hdistance <a> <b> >= V | <= V | LO to HI`
    let range_opt = if tokens.len() >= 3 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[2..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let pa = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
        let pb = resolve_endpoint_pos(&ctx.sketch, tokens[1]).unwrap();
        let measured = if horizontal { (pa.x - pb.x).abs() } else { (pa.y - pb.y).abs() };
        let kind = if horizontal { DimensionKind::HDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) }
                   else { DimensionKind::VDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) };
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} {} {} (current {:.4})", name, label, bound_desc, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} {} {} {} {} (current {:.4})",
            dim_name, label, tokens[0], tokens[1], bound_desc, measured));
    }

    if tokens.len() != 3 {
        return err(format!("Usage: {} L0.p1 L1.p2 5 [derived|driven]  or  {} <a> <b> >=V | <=V | LO to HI", label, label));
    }
    let (val, expr) = match parse_dim_value(&ctx.sketch, tokens[2]) { Ok(v) => v, Err(e) => return err(e) };
    let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = if horizontal { DimensionKind::HDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) }
               else { DimensionKind::VDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} {} = {}", name, label, tokens[2]));
    }
    ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} {} {} {} = {}", prefix, dim_name, label, tokens[0], tokens[1], tokens[2]))
}

/// Resolve the entity argument of `xangle` to either a line (angle
/// of the line from the x-axis) or an ellipse (rotation of the
/// major axis). Rejects circular arcs, whose rotation is a fixed
/// Param::fixed(0.0).
fn resolve_xangle_target(
    sketch: &Sketch,
    name: &str,
) -> Result<(DimensionKind, f64), String> {
    if let Ok(arc) = resolve_arc(sketch, name) {
        let a = &sketch.arcs[arc];
        if !a.is_ellipse {
            return Err(format!("xangle on {name}: only ellipses have an optimisable rotation; circular arcs have rotation = 0"));
        }
        return Ok((DimensionKind::ArcRotation(arc),
                   arael::utils::rad2deg(a.rotation.value)));
    }
    let line = resolve_line(sketch, name)?;
    let l = &sketch.lines[line];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    Ok((DimensionKind::LineAngle(line),
        arael::utils::rad2deg(dy.atan2(dx))))
}

fn cmd_xangle(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }

    // "xangle L0 derived" / "xangle A0 derived" -- measure-only
    if tokens.len() == 1 && (is_derived || is_driven) {
        let (kind, measured) = match resolve_xangle_target(&ctx.sketch, tokens[0]) {
            Ok(v) => v, Err(e) => return err(e),
        };
        ctx.begin_group();
        ctx.exec(Action::AddDimension { kind, value: measured, expr: None, derived: is_derived, range: None,  });
        let dim_name = ctx.sketch.dimensions.last().map(|d| d.name.clone()).unwrap_or_default();
        let label = if is_derived { "Derived" } else { "Driven" };
        return ok_or_status(ctx, format!("{} {} xangle {} = ({:.4})", label, dim_name, tokens[0], measured));
    }

    // Range form: `xangle L0 >= V | <= V | LO to HI`, likewise for arcs.
    let range_opt = if tokens.len() >= 2 && !is_derived && !is_driven {
        match parse_range_tokens(&ctx.sketch, &tokens[1..]) {
            Ok(rb) => rb,
            Err(e) => return err(e),
        }
    } else { None };
    if let Some(rb) = range_opt {
        let (kind, measured) = match resolve_xangle_target(&ctx.sketch, tokens[0]) {
            Ok(v) => v, Err(e) => return err(e),
        };
        let bound_desc = match &rb {
            RangeBound::Min(v) => format!(">= {}", v),
            RangeBound::Max(v) => format!("<= {}", v),
            RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return ok_or_status(ctx, format!("Updated {} xangle {} (current {:.4})", name, bound_desc, measured));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return ok_or_status(ctx, format!("Set {} xangle {} {} (current {:.4})", dim_name, tokens[0], bound_desc, measured));
    }

    if tokens.len() != 2 {
        return err("Usage: xangle L0 45 [derived|driven]  or  xangle A0 45 [derived|driven]  or  xangle L0 >=V | <=V | LO to HI");
    }
    let (kind, _measured) = match resolve_xangle_target(&ctx.sketch, tokens[0]) {
        Ok(v) => v, Err(e) => return err(e),
    };
    let (val, expr) = match parse_dim_value(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
        return ok_or_status(ctx, format!("Updated {} xangle = {}", name, tokens[1]));
    }
    ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    ok_or_status(ctx, format!("{} {} xangle {} = {}", prefix, dim_name, tokens[0], tokens[1]))
}

fn cmd_freeze(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();

    // Collect entities to freeze
    let mut line_refs: Vec<Ref<Line>> = Vec::new();
    let mut arc_refs: Vec<Ref<Arc>> = Vec::new();

    if tokens.is_empty() {
        // Freeze all
        line_refs.extend(ctx.sketch.lines.refs());
        arc_refs.extend(ctx.sketch.arcs.refs());
    } else {
        for name in &tokens {
            if name.starts_with('L') {
                match resolve_line(&ctx.sketch, name) {
                    Ok(r) => line_refs.push(r),
                    Err(e) => return err(e),
                }
            } else if is_arc_name(name) {
                match resolve_arc(&ctx.sketch, name) {
                    Ok(r) => arc_refs.push(r),
                    Err(e) => return err(e),
                }
            } else {
                return err(format!("freeze applies to lines and arcs: {}", name));
            }
        }
    }

    ctx.begin_group();
    let saved_skip = ctx.skip_dof_check;
    let mut frozen = Vec::new();
    let mut skipped = Vec::new();

    for r in &line_refs {
        let (name, len) = {
            let l = &ctx.sketch.lines[*r];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            (l.name.clone(), (dx * dx + dy * dy).sqrt())
        };
        let kind = DimensionKind::LineLength(*r);
        if find_existing_dimension(&ctx.sketch, &kind).is_some() {
            skipped.push(format!("{} length (exists)", name));
            continue;
        }
        ctx.skip_dof_check = true;
        ctx.exec(Action::AddDimension { kind, value: len, expr: None, derived: false, range: None,  });
        match ctx.status_error.take() {
            Some(e) => skipped.push(format!("{} length (rejected: {})", name, e)),
            None => frozen.push(format!("{} {} length={:.4}", last_dim_name(ctx), name, len)),
        }
    }

    for r in &arc_refs {
        let (name, radius, closed, sweep_deg) = {
            let a = &ctx.sketch.arcs[*r];
            (a.name.clone(), a.radius.value, a.closed,
             arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs()))
        };

        // Radius
        let kind = DimensionKind::ArcRadius(*r);
        if find_existing_dimension(&ctx.sketch, &kind).is_none() {
            ctx.skip_dof_check = true;
            ctx.exec(Action::AddDimension { kind, value: radius, expr: None, derived: false, range: None,  });
            match ctx.status_error.take() {
                Some(e) => skipped.push(format!("{} radius (rejected: {})", name, e)),
                None => frozen.push(format!("{} {} radius={:.4}", last_dim_name(ctx), name, radius)),
            }
        } else {
            skipped.push(format!("{} radius (exists)", name));
        }

        // Sweep (only for non-closed arcs)
        if !closed {
            let kind = DimensionKind::ArcSweep(*r);
            if find_existing_dimension(&ctx.sketch, &kind).is_none() {
                ctx.skip_dof_check = true;
                ctx.exec(Action::AddDimension { kind, value: sweep_deg, expr: None, derived: false, range: None,  });
                match ctx.status_error.take() {
                    Some(e) => skipped.push(format!("{} sweep (rejected: {})", name, e)),
                    None => frozen.push(format!("{} {} sweep={:.4}", last_dim_name(ctx), name, sweep_deg)),
                }
            } else {
                skipped.push(format!("{} sweep (exists)", name));
            }
        }
    }

    ctx.skip_dof_check = saved_skip;

    let mut lines = Vec::new();
    if !frozen.is_empty() {
        lines.push(format!("Frozen: {}", frozen.join(", ")));
    }
    if !skipped.is_empty() {
        lines.push(format!("Skipped: {}", skipped.join(", ")));
    }
    if lines.is_empty() {
        ok("Nothing to freeze")
    } else {
        ok(lines.join("\n"))
    }
}

/// Find the helper point associated with an arc endpoint ref (center/start/end).
/// Returns None for non-arc endpoints or if no helper point exists.
fn resolve_endpoint_as_point(sketch: &Sketch, ep: EndpointRef) -> Option<Ref<Point>> {
    match ep {
        EndpointRef::Point(p) => Some(p),
        EndpointRef::ArcCenter(arc) => sketch.coincident_arc_center.iter().find(|c| c.arc == arc).map(|c| c.point),
        EndpointRef::ArcStart(arc) => sketch.coincident_arc_start.iter().find(|c| c.arc == arc).map(|c| c.point),
        EndpointRef::ArcEnd(arc) => sketch.coincident_arc_end.iter().find(|c| c.arc == arc).map(|c| c.point),
        _ => None,
    }
}

fn find_coincident_id(sketch: &Sketch, a: EndpointRef, b: EndpointRef) -> Option<crate::ids::ConstraintId> {
    use crate::ids::ConstraintId;
    use EndpointRef::*;
    // The middle macro argument is the legacy kind tag; identity is
    // the nid now, the tag documents which collection is scanned.
    macro_rules! find_in {
        ($coll:expr, $kind:expr, $pred:expr) => {
            $coll.iter().find($pred).map(|c| ConstraintId::Numbered(c.nid))
        }
    }
    match (a, b) {
        (Point(a), Point(b)) => find_in!(sketch.coincident_pp, CoincidentKind::PP, |c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => find_in!(sketch.coincident_lp1, CoincidentKind::LP1, |c| c.line == l && c.point == p),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => find_in!(sketch.coincident_lp2, CoincidentKind::LP2, |c| c.line == l && c.point == p),
        (LineP1(a), LineP1(b)) => find_in!(sketch.coincident_ll11, CoincidentKind::LL11, |c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (LineP1(a), LineP2(b)) => find_in!(sketch.coincident_ll12, CoincidentKind::LL12, |c| c.a == a && c.b == b)
            .or_else(|| find_in!(sketch.coincident_ll21, CoincidentKind::LL21, |c| c.a == b && c.b == a)),
        (LineP2(a), LineP1(b)) => find_in!(sketch.coincident_ll21, CoincidentKind::LL21, |c| c.a == a && c.b == b)
            .or_else(|| find_in!(sketch.coincident_ll12, CoincidentKind::LL12, |c| c.a == b && c.b == a)),
        (LineP2(a), LineP2(b)) => find_in!(sketch.coincident_ll22, CoincidentKind::LL22, |c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => find_in!(sketch.coincident_arc_center, CoincidentKind::ArcCenter, |c| c.point == p && c.arc == arc),
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => find_in!(sketch.coincident_arc_start, CoincidentKind::ArcStart, |c| c.point == p && c.arc == arc),
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => find_in!(sketch.coincident_arc_end, CoincidentKind::ArcEnd, |c| c.point == p && c.arc == arc),
        (LineP1(l), ArcCenter(arc)) | (ArcCenter(arc), LineP1(l)) => find_in!(sketch.coincident_lp1_arc_center, CoincidentKind::LP1ArcCenter, |c| c.line == l && c.arc == arc),
        (LineP2(l), ArcCenter(arc)) | (ArcCenter(arc), LineP2(l)) => find_in!(sketch.coincident_lp2_arc_center, CoincidentKind::LP2ArcCenter, |c| c.line == l && c.arc == arc),
        (LineP1(l), ArcStart(arc)) | (ArcStart(arc), LineP1(l)) => find_in!(sketch.coincident_lp1_arc_start, CoincidentKind::LP1ArcStart, |c| c.line == l && c.arc == arc),
        (LineP2(l), ArcStart(arc)) | (ArcStart(arc), LineP2(l)) => find_in!(sketch.coincident_lp2_arc_start, CoincidentKind::LP2ArcStart, |c| c.line == l && c.arc == arc),
        (LineP1(l), ArcEnd(arc)) | (ArcEnd(arc), LineP1(l)) => find_in!(sketch.coincident_lp1_arc_end, CoincidentKind::LP1ArcEnd, |c| c.line == l && c.arc == arc),
        (LineP2(l), ArcEnd(arc)) | (ArcEnd(arc), LineP2(l)) => find_in!(sketch.coincident_lp2_arc_end, CoincidentKind::LP2ArcEnd, |c| c.line == l && c.arc == arc),
        (ArcCenter(a), ArcStart(b)) | (ArcStart(b), ArcCenter(a)) => find_in!(sketch.coincident_arc_center_start, CoincidentKind::ArcCenterStart, |c| c.a == a && c.b == b),
        (ArcCenter(a), ArcEnd(b)) | (ArcEnd(b), ArcCenter(a)) => find_in!(sketch.coincident_arc_center_end, CoincidentKind::ArcCenterEnd, |c| c.a == a && c.b == b),
        (ArcStart(a), ArcStart(b)) => find_in!(sketch.coincident_arc_start_start, CoincidentKind::ArcStartStart, |c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (ArcStart(a), ArcEnd(b)) => find_in!(sketch.coincident_arc_start_end, CoincidentKind::ArcStartEnd, |c| c.a == a && c.b == b)
            .or_else(|| find_in!(sketch.coincident_arc_end_start, CoincidentKind::ArcEndStart, |c| c.a == b && c.b == a)),
        (ArcEnd(a), ArcStart(b)) => find_in!(sketch.coincident_arc_end_start, CoincidentKind::ArcEndStart, |c| c.a == a && c.b == b)
            .or_else(|| find_in!(sketch.coincident_arc_start_end, CoincidentKind::ArcStartEnd, |c| c.a == b && c.b == a)),
        (ArcEnd(a), ArcEnd(b)) => find_in!(sketch.coincident_arc_end_end, CoincidentKind::ArcEndEnd, |c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        _ => None,
    }
}

fn find_point_on_line_id(sketch: &Sketch, ep: EndpointRef, line: Ref<Line>) -> Option<crate::ids::ConstraintId> {
    use crate::ids::ConstraintId;
    match ep {
        EndpointRef::Point(p) => sketch.point_on_line.iter().find(|c| c.point == p && c.line == line)
            .map(|c| ConstraintId::Numbered(c.nid)),
        EndpointRef::LineP1(l) => sketch.line_p1_on_line.iter().find(|c| c.a == l && c.b == line)
            .map(|c| ConstraintId::Numbered(c.nid)),
        EndpointRef::LineP2(l) => sketch.line_p2_on_line.iter().find(|c| c.a == l && c.b == line)
            .map(|c| ConstraintId::Numbered(c.nid)),
        _ => None,
    }
}

fn find_point_on_arc_id(sketch: &Sketch, ep: EndpointRef, arc: Ref<Arc>) -> Option<crate::ids::ConstraintId> {
    use crate::ids::ConstraintId;
    match ep {
        EndpointRef::Point(p) => sketch.point_on_arc.iter().find(|c| c.point == p && c.arc == arc)
            .map(|c| ConstraintId::Numbered(c.nid)),
        EndpointRef::LineP1(l) => sketch.line_p1_on_arc.iter().find(|c| c.line == l && c.arc == arc)
            .map(|c| ConstraintId::Numbered(c.nid)),
        EndpointRef::LineP2(l) => sketch.line_p2_on_arc.iter().find(|c| c.line == l && c.arc == arc)
            .map(|c| ConstraintId::Numbered(c.nid)),
        _ => None,
    }
}

fn find_midpoint_id(sketch: &Sketch, ep: EndpointRef, target_name: &str) -> Option<crate::ids::ConstraintId> {
    use crate::ids::ConstraintId;
    if let Ok(line) = resolve_line(sketch, target_name) {
        match ep {
            EndpointRef::Point(p) => sketch.midpoint.iter().find(|c| c.point == p && c.line == line).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::LineP1(l) => sketch.midpoint_lp1.iter().find(|c| c.line == l && c.target == line).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::LineP2(l) => sketch.midpoint_lp2.iter().find(|c| c.line == l && c.target == line).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::ArcStart(a) => sketch.midpoint_arc_start.iter().find(|c| c.arc == a && c.line == line).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::ArcEnd(a) => sketch.midpoint_arc_end.iter().find(|c| c.arc == a && c.line == line).map(|c| ConstraintId::Numbered(c.nid)),
            _ => None,
        }
    } else if let Ok(arc) = resolve_arc(sketch, target_name) {
        match ep {
            EndpointRef::Point(p) => sketch.midpoint_arc_point.iter().find(|c| c.point == p && c.arc == arc).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::LineP1(l) => sketch.midpoint_lp1_arc.iter().find(|c| c.line == l && c.arc == arc).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::LineP2(l) => sketch.midpoint_lp2_arc.iter().find(|c| c.line == l && c.arc == arc).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::ArcStart(a) => sketch.midpoint_arc_start_arc.iter().find(|c| c.a == a && c.b == arc).map(|c| ConstraintId::Numbered(c.nid)),
            EndpointRef::ArcEnd(a) => sketch.midpoint_arc_end_arc.iter().find(|c| c.a == a && c.b == arc).map(|c| ConstraintId::Numbered(c.nid)),
            _ => None,
        }
    } else { None }
}

/// Resolve a multi-token relational constraint form
/// (`L0 L1 parallel`, `L0 horizontal`, `A0 L0 A1 symmetry`, etc.) to
/// a `ConstraintId` and delete it. Called from `cmd_delete` when the
/// argument list has more than one token. Not exposed as its own
/// top-level command.
fn delete_relational(ctx: &mut CommandContext, args: &str) -> CommandResult {
    use crate::ids::ConstraintId;
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() < 2 {
        return err("Usage: delete L0 horizontal | delete L0 L1 parallel");
    }

    let ctype = tokens.last().unwrap();
    let sketch = &ctx.sketch;

    macro_rules! find_ab {
        ($coll:expr, $a:expr, $b:expr) => {
            $coll.iter().position(|c| (c.a == $a && c.b == $b) || (c.a == $b && c.b == $a))
        }
    }

    let id: Option<ConstraintId> = match *ctype {
        "horizontal" => {
            let r = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            if sketch.lines[r].constraints.horizontal { Some(ConstraintId::Horizontal(r)) } else { None }
        }
        "vertical" => {
            let r = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            if sketch.lines[r].constraints.vertical { Some(ConstraintId::Vertical(r)) } else { None }
        }
        "parallel" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_ab!(sketch.parallel, a, b).map(|i| ConstraintId::Numbered(sketch.parallel[i].nid))
        }
        "perpendicular" | "perp" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_ab!(sketch.perpendicular, a, b).map(|i| ConstraintId::Numbered(sketch.perpendicular[i].nid))
        }
        "equal" | "equal_length" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_ab!(sketch.equal_length, a, b).map(|i| ConstraintId::Numbered(sketch.equal_length[i].nid))
        }
        "collinear" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_ab!(sketch.collinear, a, b).map(|i| ConstraintId::Numbered(sketch.collinear[i].nid))
        }
        "tangent" if tokens.len() >= 3 => {
            if tokens[0].starts_with('L') && is_arc_name(tokens[1]) {
                let line = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let arc = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                sketch.tangent_la.iter().find(|c| c.line == line && c.arc == arc).map(|c| ConstraintId::Numbered(c.nid))
            } else if is_arc_name(tokens[0]) && is_arc_name(tokens[1]) {
                let a = match resolve_arc(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let b = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                find_ab!(sketch.tangent_aa, a, b).map(|i| ConstraintId::Numbered(sketch.tangent_aa[i].nid))
            } else { None }
        }
        "concentric" if tokens.len() >= 3 => {
            let a = match resolve_arc(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_ab!(sketch.concentric, a, b).map(|i| ConstraintId::Numbered(sketch.concentric[i].nid))
        }
        "lock" => {
            // Locks are entity flags, not a ConstraintId: route through
            // the unlock actions so undo restores them.
            let ep = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let action = match ep {
                EndpointRef::Point(p) => Some(Action::UnlockPoint { point: p }),
                EndpointRef::LineP1(l) => Some(Action::UnlockLineP1 { line: l }),
                EndpointRef::LineP2(l) => Some(Action::UnlockLineP2 { line: l }),
                EndpointRef::ArcCenter(a) => Some(Action::UnlockArcCenter { arc: a }),
                _ => None,
            };
            let Some(action) = action else { return err("Constraint not found".to_string()); };
            ctx.begin_group();
            ctx.exec(action);
            return ok(format!("Removed lock on {}", tokens[0]));
        }
        "equal_radius" if tokens.len() >= 3 => {
            let a = match resolve_arc(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_ab!(sketch.equal_radius, a, b).map(|i| ConstraintId::Numbered(sketch.equal_radius[i].nid))
        }
        "coincident" if tokens.len() >= 3 => {
            let a = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_endpoint_ref(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            find_coincident_id(sketch, a, b)
        }
        "point_on" if tokens.len() >= 3 => {
            let ep = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let target = tokens[1];
            let found = if target.starts_with('L') || target.starts_with('l') {
                let line = match resolve_line(sketch, target) { Ok(r) => r, Err(e) => return err(e) };
                find_point_on_line_id(sketch, ep, line)
            } else if is_arc_name(target) || target.starts_with('a') {
                let arc = match resolve_arc(sketch, target) { Ok(r) => r, Err(e) => return err(e) };
                find_point_on_arc_id(sketch, ep, arc)
            } else { None };
            // Arc endpoints use helper points -- fall back to the
            // helper's own point_on entry if the direct lookup missed.
            if found.is_none()
                && let Some(p) = resolve_endpoint_as_point(&ctx.sketch, ep) {
                    if target.starts_with('L') || target.starts_with('l') {
                        let line = match resolve_line(&ctx.sketch, target) { Ok(r) => r, Err(e) => return err(e) };
                        ctx.sketch.point_on_line.iter()
                            .position(|c| c.point == p && c.line == line)
                            .map(|i| ConstraintId::Numbered(ctx.sketch.point_on_line[i].nid))
                    } else if is_arc_name(target) || target.starts_with('a') {
                        let arc = match resolve_arc(&ctx.sketch, target) { Ok(r) => r, Err(e) => return err(e) };
                        ctx.sketch.point_on_arc.iter()
                            .position(|c| c.point == p && c.arc == arc)
                            .map(|i| ConstraintId::Numbered(ctx.sketch.point_on_arc[i].nid))
                    } else {
                        None
                    }
                } else {
                    found
                }
        }
        "symmetry" if tokens.len() >= 4 => {
            // Try arc-arc symmetry first
            if let (Ok(a), Ok(line), Ok(c)) = (resolve_arc(sketch, tokens[0]),
                resolve_line(sketch, tokens[1]),
                resolve_arc(sketch, tokens[2]))
            {
                sketch.symmetry_aa.iter().find(|s| s.line == line && ((s.a == a && s.c == c) || (s.a == c && s.c == a)))
                    .map(|s| ConstraintId::Numbered(s.nid))
            } else {
                let ep_a = resolve_endpoint_ref(sketch, tokens[0]);
                let ep_c = resolve_endpoint_ref(sketch, tokens[2]);
                if let (Ok(EndpointRef::Point(a)), Ok(EndpointRef::Point(c))) = (ep_a, ep_c) {
                    let line = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                    sketch.symmetry_pp.iter().find(|s| (s.a == a && s.c == c && s.line == line) || (s.a == c && s.c == a && s.line == line))
                        .map(|s| ConstraintId::Numbered(s.nid))
                } else {
                    let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                    let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                    let c = match resolve_line(sketch, tokens[2]) { Ok(r) => r, Err(e) => return err(e) };
                    sketch.symmetry_ll.iter().find(|s| s.b == b && ((s.a == a && s.c == c) || (s.a == c && s.c == a)))
                        .map(|s| ConstraintId::Numbered(s.nid))
                }
            }
        }
        "midpoint" if tokens.len() >= 3 => {
            let ep = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            find_midpoint_id(sketch, ep, tokens[1])
        }
        _ => { return err(format!("Unknown constraint type: {}. Use: horizontal, vertical, parallel, perpendicular, equal, equal_radius, collinear, tangent, concentric, coincident, point_on, symmetry, midpoint, lock", ctype)); }
    };

    if let Some(id) = id {
        // Name it before the delete removes it.
        let id_name = crate::ids::constraint_id_name(&ctx.sketch, id);
        ctx.begin_group();
        ctx.exec(Action::DeleteConstraint { id });
        match id_name {
            Some(n) => ok(format!("Removed {} constraint {}", ctype, n)),
            None => ok(format!("Removed {} constraint", ctype)),
        }
    } else {
        err("Constraint not found".to_string())
    }
}

// ---------------------------------------------------------------------------
// Rename param
// ---------------------------------------------------------------------------

fn cmd_rename_param(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: rename_param old_name new_name"); }
    let old = tokens[0];
    let new = tokens[1];
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == old) {
        if let Err(e) = ctx.sketch.validate_param_name(new, Some(idx)) {
            return err(e);
        }
        let expr = ctx.sketch.user_params[idx].expr_str.clone();
        ctx.begin_group();
        ctx.exec(Action::UpdateUserParam { index: idx, name: new.to_string(), expr_str: expr });
        ok(format!("Renamed {} -> {}", old, new))
    } else {
        err(format!("Unknown parameter: {}", old))
    }
}

// ---------------------------------------------------------------------------
// Find
// ---------------------------------------------------------------------------

fn cmd_find(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: find x,y [radius]"); }
    let pos = match parse_coord(ctx, tokens[0], None) { Ok(p) => p, Err(e) => return err(e) };
    let radius = if tokens.len() > 1 {
        tokens[1].parse::<f64>().unwrap_or(1.0)
    } else { 1.0 };
    let r2 = radius * radius;
    let mut found = Vec::new();
    for r in ctx.sketch.points.refs() {
        let p = &ctx.sketch.points[r];
        if p.helper { continue; }
        let d2 = (p.pos.value.x - pos.x).powi(2) + (p.pos.value.y - pos.y).powi(2);
        if d2 <= r2 { found.push(format!("{} ({:.2},{:.2})", p.name, p.pos.value.x, p.pos.value.y)); }
    }
    for r in ctx.sketch.lines.refs() {
        let l = &ctx.sketch.lines[r];
        // Point-to-segment distance
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let len2 = dx * dx + dy * dy;
        let dist = if len2 < 1e-24 {
            ((l.p1.value.x - pos.x).powi(2) + (l.p1.value.y - pos.y).powi(2)).sqrt()
        } else {
            let t = (((pos.x - l.p1.value.x) * dx + (pos.y - l.p1.value.y) * dy) / len2).clamp(0.0, 1.0);
            let cx = l.p1.value.x + t * dx - pos.x;
            let cy = l.p1.value.y + t * dy - pos.y;
            (cx * cx + cy * cy).sqrt()
        };
        if dist <= radius { found.push(format!("{} (d={:.2})", l.name, dist)); }
    }
    for r in ctx.sketch.arcs.refs() {
        let a = &ctx.sketch.arcs[r];
        let dc = ((a.center.value.x - pos.x).powi(2) + (a.center.value.y - pos.y).powi(2)).sqrt();
        let dist_to_curve = (dc - a.radius.value).abs();
        if dist_to_curve <= radius || dc <= radius {
            found.push(format!("{} (d={:.2})", a.name, dist_to_curve.min(dc)));
        }
    }
    if found.is_empty() {
        ok(format!("Nothing found within {:.1} of ({:.2},{:.2})", radius, pos.x, pos.y))
    } else {
        ok(found.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Goto history position
// ---------------------------------------------------------------------------

fn cmd_goto(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let target_group: usize = match args.trim().parse() {
        Ok(v) => v,
        Err(_) => return err("Usage: goto <group_number> (see 'history')"),
    };
    let groups = ctx.history.group_list();
    let target_pos = if target_group == 0 {
        0
    } else if target_group <= groups.len() {
        groups[target_group - 1].1
    } else {
        return err(format!("Group {} does not exist (max {})", target_group, groups.len()));
    };
    if let Some((s, c)) = ctx.history.goto(target_pos) {
        ctx.sketch = s.into();
        ctx.cursor = c.pos;
        ctx.cursor_tangent = c.tangent;
    }
    ok(format!("Moved to group {} (position {})", target_group, ctx.history.cursor))
}

// ---------------------------------------------------------------------------
// Session variables
// ---------------------------------------------------------------------------

fn cmd_let(ctx: &mut CommandContext, args: &str) -> CommandResult {
    // let name = expr
    let args = args.trim();
    let (name, expr) = match args.split_once('=') {
        Some((n, e)) => (n.trim(), e.trim()),
        None => return err("Usage: let name = expression"),
    };
    if name.is_empty() { return err("Variable name cannot be empty"); }
    // Try as coordinate (geo function or endpoint ref)
    if let Some(result) = eval_geo_coord(&ctx.sketch, expr) {
        match result {
            Ok(v) => {
                ctx.session_vecs.insert(name.to_string(), v);
                ctx.session_vars.insert(format!("{}.x", name), v.x);
                ctx.session_vars.insert(format!("{}.y", name), v.y);
                return ok(format!("{} = ({:.4}, {:.4})", name, v.x, v.y));
            }
            Err(e) => return err(e),
        }
    }
    if let Ok(pos) = resolve_endpoint_pos(&ctx.sketch, expr) {
        ctx.session_vecs.insert(name.to_string(), pos);
        ctx.session_vars.insert(format!("{}.x", name), pos.x);
        ctx.session_vars.insert(format!("{}.y", name), pos.y);
        return ok(format!("{} = ({:.4}, {:.4})", name, pos.x, pos.y));
    }
    // Try as scalar
    if let Some(result) = eval_geo_scalar(&ctx.sketch, expr) {
        match result {
            Ok(v) => { ctx.session_vars.insert(name.to_string(), v); return ok(format!("{} = {:.6}", name, v)); }
            Err(e) => return err(e),
        }
    }
    match eval_expr_with(&ctx.sketch, expr, &ctx.session_vars) {
        Ok(v) => { ctx.session_vars.insert(name.to_string(), v); ok(format!("{} = {:.6}", name, v)) }
        Err(e) => err(format!("Eval error: {}", e)),
    }
}

// ---------------------------------------------------------------------------
// File operations
// ---------------------------------------------------------------------------

fn cmd_save(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let path = args.trim();
    if path.is_empty() { return err("Usage: save path.json"); }
    match serde_json::to_string_pretty(&ctx.sketch) {
        Ok(json) => match std::fs::write(path, &json) {
            Ok(_) => ok(format!("Saved to {}", path)),
            Err(e) => err(format!("Write error: {}", e)),
        },
        Err(e) => err(format!("Serialize error: {}", e)),
    }
}

fn cmd_load(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let path = args.trim();
    if path.is_empty() { return err("Usage: load path.json"); }
    match std::fs::read_to_string(path) {
        Ok(json) => match serde_json::from_str::<Sketch>(&json) {
            Ok(mut sketch) => {
                sketch.assign_constraint_names();
                sketch.solve();
                ctx.history = crate::history::History::new(&sketch);
                ctx.sketch = sketch.into();
                ok(format!("Loaded {}", path))
            }
            Err(e) => err(format!("Parse error: {}", e)),
        },
        Err(e) => err(format!("Read error: {}", e)),
    }
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

fn cmd_cursor(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    if args.is_empty() || args == "info" {
        // Show cursor position and tangent
        let pos_str = match ctx.cursor {
            Some(p) => format!("({:.4}, {:.4})", p.x, p.y),
            None => "off".into(),
        };
        let tan_str = match ctx.cursor_tangent {
            Some(t) => format!("({:.4}, {:.4})", t.x, t.y),
            None => "none".into(),
        };
        return ok(format!("Cursor: {} tangent: {}", pos_str, tan_str));
    }
    match args {
        "off" | "hide" => { ctx.cursor = None; return ok("Cursor hidden"); }
        "on" | "show" => {
            if ctx.cursor.is_none() { ctx.cursor = Some(vect2d::new(0.0, 0.0)); }
            let p = ctx.cursor.unwrap();
            return ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y));
        }
        _ => {}
    }
    // Set cursor to coordinate (absolute, relative, endpoint ref, etc.)
    match parse_coord(ctx, args, ctx.cursor) {
        Ok(p) => { ctx.cursor = Some(p); ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y)) }
        Err(e) => err(e),
    }
}

// ---------------------------------------------------------------------------
// Dimension text position
// ---------------------------------------------------------------------------

fn cmd_dim_pos(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 3 { return err("Usage: dim_pos d0 offset 1.5  or  dim_pos d0 along 0.3"); }
    let dim_name = tokens[0];
    let field = tokens[1];
    let val_str = tokens[2];
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == dim_name) {
        Some(i) => i,
        None => return err(format!("Unknown dimension: {}", dim_name)),
    };
    let is_relative = val_str.starts_with('@');
    let val_str = val_str.strip_prefix('@').unwrap_or(val_str);
    let val = match eval_expr(&ctx.sketch, val_str) { Ok(v) => v, Err(e) => return err(e) };
    let mut offset = ctx.sketch.dimensions[idx].offset;
    let mut text_along = ctx.sketch.dimensions[idx].text_along;
    match field {
        "offset" => {
            offset.y = if is_relative { offset.y + val } else { val };
            ctx.begin_group();
            ctx.exec(Action::MoveDimension { did: ctx.sketch.dimensions[idx].did, offset, text_along });
            ok(format!("{} offset = {:.4}", dim_name, ctx.sketch.dimensions[idx].offset.y))
        }
        "along" => {
            text_along = if is_relative { text_along + val } else { val };
            ctx.begin_group();
            ctx.exec(Action::MoveDimension { did: ctx.sketch.dimensions[idx].did, offset, text_along });
            ok(format!("{} along = {:.4}", dim_name, ctx.sketch.dimensions[idx].text_along))
        }
        _ => err(format!("Unknown field '{}'. Use: offset, along", field)),
    }
}

// ---------------------------------------------------------------------------
// Derived dimensions
// ---------------------------------------------------------------------------

fn cmd_set_derived(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        Some(i) => i,
        None => return err(format!("Unknown dimension: {}", name)),
    };
    if ctx.sketch.dimensions[idx].derived {
        return ok(format!("{} is already derived", name));
    }
    let did = ctx.sketch.dimensions[idx].did;
    ctx.begin_group();
    ctx.exec(Action::ConvertDimension { did, derived: true, value: None });
    ok_or_status(ctx, format!("{} is now derived (reference only)", name))
}

fn cmd_set_driven(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
    if tokens.is_empty() { return err("Usage: set_driven d0 [value]"); }
    let name = tokens[0].trim();
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        Some(i) => i,
        None => return err(format!("Unknown dimension: {}", name)),
    };
    if !ctx.sketch.dimensions[idx].derived {
        return ok(format!("{} is already driven (constraining)", name));
    }
    let (new_value, new_expr) = if tokens.len() > 1 {
        let val_str = tokens[1].trim().trim_matches('"');
        if let Ok(v) = val_str.parse::<f64>() {
            (v, None)
        } else {
            // Expression: evaluate for initial value, store as expr
            let v = match eval_expr(&ctx.sketch, val_str) { Ok(v) => v, Err(e) => return err(e) };
            (v, Some(val_str.to_string()))
        }
    } else {
        (ctx.sketch.dimensions[idx].value, None)
    };
    let did = ctx.sketch.dimensions[idx].did;
    ctx.begin_group();
    ctx.exec(Action::ConvertDimension { did, derived: false, value: Some(new_value) });
    if ctx.status_error.is_none()
        && let Some(expr) = &new_expr {
            ctx.exec(Action::UpdateDimension {
                did, value: new_value, expr: Some(expr.clone()), range: None,
            });
    }
    let msg = match &new_expr {
        Some(expr) => format!("{} is now driven (constraining) = {}", name, expr),
        None => format!("{} is now driven (constraining) = {:.4}", name, new_value),
    };
    ok_or_status(ctx, msg)
}

// ---------------------------------------------------------------------------
// DOF analysis
// ---------------------------------------------------------------------------

/// Classify free directions from DofResult eigenvectors.
fn classify_dof_directions(result: &arael_sketch_solver::DofResult) -> Vec<String> {
    let threshold = 1e-6;
    let n = result.eigenvalues.len();
    let mut free_dirs = Vec::new();
    for col in 0..n {
        if result.eigenvalues[col].abs() > threshold { continue; }
        let ev = &result.eigenvectors[col];

        let max_comp = ev.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        if max_comp < 1e-10 { continue; }
        let comp_threshold = max_comp * 0.1;

        let mut parts: Vec<(String, f64)> = Vec::new();
        for i in 0..n {
            if ev[i].abs() > comp_threshold {
                let name = if result.param_names[i].is_empty() {
                    format!("param[{}]", i)
                } else {
                    result.param_names[i].clone()
                };
                parts.push((name, ev[i]));
            }
        }
        if parts.is_empty() { continue; }
        free_dirs.push(classify_free_direction(&parts));
    }
    free_dirs
}

fn cmd_dof_eigenvalues(ctx: &mut CommandContext, raw: bool) -> CommandResult {
    let t0 = web_time::Instant::now();
    // Hessian-based diagnostic. Default is the symmetrically Jacobi-
    // preconditioned Hessian `D^{-1} H D^{-1}` with `D =
    // diag(sqrt(diag(H)))` -- the null-space is preserved (rank
    // unchanged) but the per-parameter scale differences that
    // otherwise make eigenvalue rank-detection break at high sketch
    // scales are folded into `D`. Eigenvectors back-transformed to
    // raw parameter space. `raw` shows the un-preconditioned Hessian
    // for residual-design debugging.
    let result = match ctx.sketch.get_mut().compute_dof_eigenvalues_opt(true, !raw) {
        Ok(r) => r,
        Err(e) => return err(e),
    };
    let t_total = t0.elapsed();
    let n = result.eigenvalues.len();
    if n == 0 {
        return ok("Hessian: 0x0 (empty)".to_string());
    }
    let header = if raw { "Hessian (raw)" } else { "Hessian (preconditioned)" };
    let mut lines = vec![format!("{}: {}x{}, DOF: {}, time: {:.2}ms",
        header, n, n, result.dof, t_total.as_secs_f64() * 1000.0)];
    let mut evs: Vec<(f64, usize)> = result.eigenvalues.iter().cloned().enumerate().map(|(i,v)| (v, i)).collect();
    evs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (val, col) in &evs {
        let ev = &result.eigenvectors[*col];
        let max_comp = ev.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let comp_threshold = max_comp * 0.3;
        // Render as linear combination: "+0.707 A0.start_angle -0.707 A0.end_angle"
        let parts: Vec<String> = (0..n).filter(|&i| ev[i].abs() > comp_threshold)
            .enumerate()
            .map(|(k, i)| {
                let name = if result.param_names[i].is_empty() { format!("[{}]", i) } else { result.param_names[i].clone() };
                let v = ev[i];
                if k == 0 {
                    if v < 0.0 { format!("-{:.3} {}", -v, name) } else { format!("{:.3} {}", v, name) }
                } else {
                    if v < 0.0 { format!("- {:.3} {}", -v, name) } else { format!("+ {:.3} {}", v, name) }
                }
            })
            .collect();
        lines.push(format!("  {:.6e}  {}", val, parts.join(" ")));
    }
    ok(lines.join("\n"))
}

fn cmd_dof_singular(ctx: &mut CommandContext, raw: bool) -> CommandResult {
    use arael_sketch_solver::SymbolBag;
    ctx.sketch.get_mut().prepare_expr_constraints();
    let saved_drift = ctx.sketch.drift_isigma;
    ctx.sketch.get_mut().drift_isigma = 0.0;
    let mut params = Vec::new();
    ctx.sketch.mutate_values(|s| s.serialize(&mut params));
    let n = params.len();
    let bag = SymbolBag::build(&ctx.sketch);
    let mut idx_to_name: Vec<String> = vec![String::new(); n];
    for (name, &idx) in &bag.param_indices {
        let i = idx as usize;
        if i < n && idx_to_name[i].is_empty() { idx_to_name[i] = name.clone(); }
    }
    let t0 = web_time::Instant::now();
    let jacobian = ctx.sketch.get_mut().calc_jacobian(&params);
    let t_build = t0.elapsed();
    ctx.sketch.get_mut().drift_isigma = saved_drift;
    let m = jacobian.num_residuals();
    if m == 0 || n == 0 {
        return ok(format!("Jacobian: {} residuals x {} params (empty)", m, n));
    }
    // Degenerate geometry yields NaN; an SVD iterating on it never
    // converges. Same guard as compute_dof.
    if jacobian.rows.iter().any(|r| r.entries.iter().any(|&(_, v)| !v.is_finite())) {
        return err("Jacobian contains non-finite values (degenerate geometry)");
    }
    // Raw SVD: un-normalised Jacobian; sigmas carry the real
    // per-parameter scale, useful for spotting residual-design issues.
    // Normalised SVD (the default): each column of J scaled by
    // 1 / col_L2_norm, so the spectrum reflects only row-space linear
    // dependence. Right singular vectors get back-transformed to raw
    // parameter space (v_raw[i] = v_norm[i] / col_norms[i], renormed
    // to unit length) so the printed eigenvector still describes a
    // direction in the user's parameter coordinates.
    let t1 = web_time::Instant::now();
    let (svd, col_norms) = if raw {
        (jacobian.svd(), Vec::new())
    } else {
        jacobian.svd_column_normalised()
    };
    let t_svd = t1.elapsed();
    let svs_vec = &svd.singular_values;
    let k_dim = svs_vec.len();
    // Re-pack V (n x k row-major) and U (m x k row-major) into accessors
    // that mirror the legacy nalgebra/faer API (sv at singular-value index
    // -> direction in parameter space for V, contribution per row for U).
    let v_row = |idx: usize| -> Vec<f64> {
        let mut row = vec![0.0f64; n];
        for i in 0..n { row[i] = svd.v[i * k_dim + idx]; }
        row
    };
    let u_col = |idx: usize| -> Vec<f64> {
        let mut col = vec![0.0f64; m];
        for i in 0..m { col[i] = svd.u[i * k_dim + idx]; }
        col
    };

    let labels = ctx.sketch.constraint_labels();

    let header = if raw { "Jacobian (raw)" } else { "Jacobian (column-normalised)" };
    let mut lines = vec![format!("{}: {} residuals x {} params", header, m, n)];
    lines.push(format!("  build: {:.2}ms, svd: {:.2}ms",
        t_build.as_secs_f64() * 1000.0,
        t_svd.as_secs_f64() * 1000.0));
    let mut svs: Vec<(f64, usize)> = svs_vec.iter().cloned().enumerate().map(|(i,v)| (v, i)).collect();
    svs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (val, idx) in &svs {
        // Right singular vector: direction in parameter space.
        // For the normalised SVD we back-transform by dividing each
        // component by the corresponding column norm and then rescaling
        // to unit length, so the printed eigenvector still describes a
        // physical direction the user can act on. Raw SVD needs no
        // transform.
        let sv_raw = v_row(*idx);
        let sv: Vec<f64> = if raw {
            sv_raw
        } else {
            let mut adjusted: Vec<f64> = sv_raw.iter().enumerate()
                .map(|(i, &v)| v / col_norms[i].max(1e-15))
                .collect();
            let norm: f64 = adjusted.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-20 {
                for v in adjusted.iter_mut() { *v /= norm; }
            }
            adjusted
        };
        let max_comp = sv.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        let comp_threshold = max_comp * 0.3;
        let parts: Vec<String> = (0..n).filter(|&i| sv[i].abs() > comp_threshold)
            .enumerate()
            .map(|(k, i)| {
                let name = if idx_to_name[i].is_empty() { format!("[{}]", i) } else { idx_to_name[i].clone() };
                let v = sv[i];
                if k == 0 {
                    if v < 0.0 { format!("-{:.3} {}", -v, name) } else { format!("{:.3} {}", v, name) }
                } else {
                    if v < 0.0 { format!("- {:.3} {}", -v, name) } else { format!("+ {:.3} {}", v, name) }
                }
            })
            .collect();
        lines.push(format!("  {:.6e}  {}", val, parts.join(" ")));

        // Left singular vector: which constraints project onto this direction.
        // Aggregate u[row]^2 by (cid, label) — the label gives attribute-level
        // granularity (e.g. Arc:0 vs Arc:4) while cid gives the instance.
        let uv = u_col(*idx);
        let mut weight: std::collections::HashMap<(u32, &'static str), f64> = std::collections::HashMap::new();
        for (row_idx, row) in jacobian.rows.iter().enumerate() {
            if row_idx >= uv.len() { break; }
            let w = uv[row_idx];
            *weight.entry((row.constraint, row.label)).or_insert(0.0) += w * w;
        }
        // Sum over all constraints = 1 (u is a unit vector). Report as percentages.
        let mut contribs: Vec<((u32, &'static str), f64)> = weight.into_iter().collect();
        contribs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top_max = contribs.first().map(|(_, w)| *w).unwrap_or(0.0);
        let top_threshold = top_max * 0.1; // show anything >= 10% of dominant
        contribs.retain(|(_, w)| *w > top_threshold);
        contribs.truncate(6);
        if !contribs.is_empty() {
            let parts: Vec<String> = contribs.iter().map(|((cid, label), w)| {
                // Instance label like "arc:A0" or "parallel:L3,L0"; take the part after ":"
                // for compactness since it already identifies the entity.
                let instance = labels.get(cid).cloned().unwrap_or_else(|| format!("cid={}", cid));
                let instance_short = instance.split_once(':').map(|x| x.1).unwrap_or(&instance).to_string();
                format!("{:.0}% {}/{}", w * 100.0, instance_short, label)
            }).collect();
            lines.push(format!("           {}", parts.join(", ")));
        }
    }
    ok(lines.join("\n"))
}

fn cmd_dof_jacobian(ctx: &mut CommandContext) -> CommandResult {
    use arael_sketch_solver::SymbolBag;
    ctx.sketch.get_mut().prepare_expr_constraints();
    let saved_drift = ctx.sketch.drift_isigma;
    ctx.sketch.get_mut().drift_isigma = 0.0;
    let mut params = Vec::new();
    ctx.sketch.mutate_values(|s| s.serialize(&mut params));
    let n = params.len();
    if n == 0 {
        ctx.sketch.get_mut().drift_isigma = saved_drift;
        return ok("No params".to_string());
    }
    let bag = SymbolBag::build(&ctx.sketch);
    let mut idx_to_name: Vec<String> = vec![String::new(); n];
    for (name, &idx) in &bag.param_indices {
        let i = idx as usize;
        if i < n && idx_to_name[i].is_empty() { idx_to_name[i] = name.clone(); }
    }
    let jacobian = ctx.sketch.get_mut().calc_jacobian(&params);
    let labels = ctx.sketch.constraint_labels();
    ctx.sketch.get_mut().drift_isigma = saved_drift;
    let mut lines = vec![format!("Jacobian: {} rows x {} cols", jacobian.num_residuals(), n)];
    for (i, row) in jacobian.rows.iter().enumerate() {
        let entries: Vec<String> = row.entries.iter()
            .map(|&(idx, val)| {
                let name = if idx_to_name[idx as usize].is_empty() {
                    format!("[{}]", idx)
                } else {
                    idx_to_name[idx as usize].clone()
                };
                // Collapse exact zeros to "<name>=0" to cut noise while keeping
                // the parameter visible as part of the constraint.
                if val == 0.0 {
                    format!("{}=0", name)
                } else {
                    format!("{}={:+.6}", name, val)
                }
            })
            .collect();
        let norm: f64 = row.entries.iter().map(|&(_, v)| v * v).sum::<f64>().sqrt();
        // Combine instance (from cid->label map) with attribute label (from row.label).
        let instance = labels.get(&row.constraint).cloned().unwrap_or_else(|| format!("cid={}", row.constraint));
        let instance_short = instance.split_once(':').map(|x| x.1).unwrap_or(&instance).to_string();
        let combined = format!("{}/{}", instance_short, row.label);
        let r_str = if row.residual == 0.0 { "r=0".to_string() } else { format!("r={:+.6e}", row.residual) };
        let norm_str = if norm == 0.0 { "|dr|=0".to_string() } else { format!("|dr|={:.6e}", norm) };
        lines.push(format!("  row {:3} cid={:3} {:30} {:16} {:16} dr/d[{}]",
            i, row.constraint, combined, r_str, norm_str, entries.join(", ")));
    }
    ok(lines.join("\n"))
}

fn cmd_dof(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let arg = args.trim();
    if arg == "eigenvalues" {
        return cmd_dof_eigenvalues(ctx, false);
    }
    if arg == "eigenvalues raw" {
        return cmd_dof_eigenvalues(ctx, true);
    }
    // `dof singular` uses the column-normalised Jacobian (each column
    // scaled by 1 / L2_norm) so the sigma spectrum reflects row-space
    // linear dependence only, not per-parameter scale conditioning.
    // This matches what the internal rank detector uses. `dof singular
    // raw` falls back to the un-normalised Jacobian, useful when
    // debugging residual scaling choices where the raw sigmas carry
    // meaningful physical magnitudes.
    if arg == "singular" {
        return cmd_dof_singular(ctx, false);
    }
    if arg == "singular raw" {
        return cmd_dof_singular(ctx, true);
    }
    if arg == "jacobian" {
        return cmd_dof_jacobian(ctx);
    }
    if !arg.is_empty() && arg != "analyze" {
        return err("Usage: dof | dof analyze | dof eigenvalues [raw] | dof singular [raw] | dof jacobian");
    }

    if arg == "analyze" {
        // The eigenvector analysis has no cached form; it is a
        // diagnostic and may bump the generation.
        let result = match ctx.sketch.get_mut().compute_dof(true) {
            Ok(r) => r,
            Err(e) => return err(e),
        };
        let free_dirs = classify_dof_directions(&result);
        let mut lines = vec![format!("DOF: {}", result.dof)];
        for (i, desc) in free_dirs.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, desc));
        }
        ok(lines.join("\n"))
    } else {
        // Plain count: the read door, so the cache and the warm
        // session survive the query.
        match ctx.sketch.dof() {
            Ok(d) => ok(format!("DOF: {}", d)),
            Err(e) => err(e),
        }
    }
}

/// Classify a free direction from its eigenvector components.
fn classify_free_direction(parts: &[(String, f64)]) -> String {
    // Single parameter free
    if parts.len() == 1 {
        return format!("{} is free", parts[0].0);
    }

    // Collect entity names and check motion patterns
    let mut entities = std::collections::BTreeSet::new();
    let mut all_x = true;
    let mut all_y = true;
    let mut has_non_xy = false;

    for (name, _val) in parts {
        // Extract entity name (e.g., "L0" from "L0.p1.x")
        let entity = name.split('.').next().unwrap_or(name);
        entities.insert(entity.to_string());

        if name.ends_with(".x") {
            all_y = false;
        } else if name.ends_with(".y") {
            all_x = false;
        } else {
            // radius, angle, etc.
            all_x = false;
            all_y = false;
            has_non_xy = true;
        }
    }

    let entity_list: Vec<&str> = entities.iter().map(|s| s.as_str()).collect();
    let entity_str = if entity_list.len() <= 9 {
        entity_list.join(", ")
    } else {
        format!("{} entities", entity_list.len())
    };

    // Check for pure translation
    if all_x && !has_non_xy {
        return format!("translate X: {}", entity_str);
    }
    if all_y && !has_non_xy {
        return format!("translate Y: {}", entity_str);
    }

    // Check for uniform translation (all x components equal AND all y components equal)
    let x_vals: Vec<f64> = parts.iter()
        .filter(|(n, _)| n.ends_with(".x"))
        .map(|(_, v)| *v).collect();
    let y_vals: Vec<f64> = parts.iter()
        .filter(|(n, _)| n.ends_with(".y"))
        .map(|(_, v)| *v).collect();

    if !has_non_xy && !x_vals.is_empty() && !y_vals.is_empty()
        && y_vals.iter().all(|v| v.abs() < 1e-6) {
        // All Y near zero, only X moves
        return format!("translate X: {}", entity_str);
    }
    if !has_non_xy && !x_vals.is_empty() && !y_vals.is_empty()
        && x_vals.iter().all(|v| v.abs() < 1e-6) {
        return format!("translate Y: {}", entity_str);
    }

    // Check for rotation: x and y components should follow tangent pattern
    // For rotation around centroid, dx_i ~ -(y_i - cy), dy_i ~ (x_i - cx)
    // Simplified: if x and y components are mixed and entities share the motion, call it rotation
    if !x_vals.is_empty() && !y_vals.is_empty() && x_vals.len() == y_vals.len() && !has_non_xy {
        // Check if all translation components are equal (pure translation)
        let all_x_equal = x_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        let all_y_equal = y_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        if all_x_equal && all_y_equal {
            return format!("translate: {}", entity_str);
        }
        return format!("rotate: {}", entity_str);
    }

    // Check for single-entity multi-param freedom
    if entities.len() == 1 {
        let param_list: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
        return format!("{} free: {}", entity_list[0], param_list.join(", "));
    }

    // Fallback: list participating entities
    format!("coupled motion: {}", entity_str)
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

fn cmd_help(args: &str) -> CommandResult {
    if args.trim() == "full" {
        return CommandResult {
            output: include_str!("../docs/COMMANDS.md").to_string(),
            is_error: false, no_echo: false, markdown: true,
        };
    }
    if args.is_empty() {
        ok("Commands: add_line add_point add_circle add_arc offset_line delete horizontal vertical \
            parallel perpendicular equal collinear tangent coincident concentric midpoint \
            symmetry point_on length radius sweep angle distance hdistance vdistance xangle freeze set_derived set_driven \
            lock unlock param del_param rename_param style select deselect print info list \
            find dof cost undo redo history goto center zoom cursor dim_pos clear let save load \
            exit help\n\
            Type 'help <command>' for details. 'help full' for complete reference.")
    } else {
        let msg = match args.trim() {
            "add_line" => "add_line x1,y1 x2,y2 [x3,y3 ...] [noconnect] [nocursor] [driven]",
            "add_rect" => "add_rect x1,y1 x2,y2 [hv] [noconnect] [noconstraint] [driven] [strict]",
            "add_rect3" => "add_rect3 p1 p2 p3 [noconnect] [noconstraint] [driven] [strict]",
            "add_rectcenter" => "add_rectcenter cx,cy px,py [hv] [noconnect] [noconstraint] [driven] [strict]",
            "add_point" => "add_point x,y [nocursor]",
            "add_circle" => "add_circle cx,cy radius [noconnect] [nocursor] [driven]",
            "add_circle2" => "add_circle2 p1 p2 [noconnect] [nocursor] [driven] — circle from 2 diametrically opposite points",
            "add_circle3" => "add_circle3 p1 p2 p3 [noconnect] [nocursor] [driven] — circle from 3 points on circumference",
            "add_circle2t" => "add_circle2t L0 L1 radius [noconnect] [noconstraint] [driven] [strict] — circle tangent to 2 lines",
            "add_circle3t" => "add_circle3t L0 L1 L2 [noconnect] [noconstraint] [driven] [strict] — circle tangent to 3 lines",
            "add_ellipse" => "add_ellipse cx,cy rx ry rotation_deg [noconnect] [nocursor] [driven]",
            "delete" => "delete <L0|P0|A0|EA0|C3|CL0H|d0> | delete L0 L1 parallel",
            "horizontal" => "horizontal L0 [L1 ...]",
            "vertical" => "vertical L0 [L1 ...]",
            "parallel" => "parallel L0 L1",
            "perpendicular" => "perpendicular L0 L1",
            "equal" => "equal L0 L1 (length) | equal A0 A1 (radius)",
            "collinear" => "collinear L0 L1",
            "tangent" => "tangent L0 A0 | tangent A0 A1",
            "coincident" => "coincident L0.p2 L1.p1 (any endpoint pair: P0, L0.p1/p2, A0.center/start/end)",
            "concentric" => "concentric A0 A1",
            "midpoint" => "midpoint P0 L0 | midpoint L0.p1 L1 | midpoint P0 A0 (arc angular midpoint)",
            "symmetry" => "symmetry L0 L1 L2 | symmetry P0 L0 P1 | symmetry A0 L0 A1",
            "mirror" => "mirror L0 L1 ... about L_axis [noconstraint] [strict] | mirror selection about L_axis",
            "point_on" => "point_on P0 L0 | point_on L0.p1 A0",
            "length" => "length L0 5 | length L0 L0.length | length L0 =2*scale | length L0 {expr} [derived|driven]",
            "radius" => "radius A0 1.5 | radius A0 =5*scale | radius A0 {expr} [derived|driven]",
            "radius_b" => "radius_b A0 1.5 [derived|driven] -- ellipse semi-minor axis",
            "sweep" => "sweep A0 180 | sweep A0 =90*n | sweep A0 {expr} [derived|driven]",
            "angle" => "angle L0 L1 45 [supplement|closest|acute|obtuse] [derived|driven]",
            "distance" => "distance L0.p1 L1.p2 5 | distance P0 L0 3 | distance L0.p1 L1.p2 =expr [derived|driven]",
            "hdistance" => "hdistance L0.p1 L1.p2 5 [derived|driven] — horizontal (x-axis) distance",
            "vdistance" => "vdistance L0.p1 L1.p2 3 [derived|driven] — vertical (y-axis) distance",
            "xangle" => "xangle L0 45 [derived|driven] — line angle from x-axis in degrees",
            "freeze" => "freeze [L0 L1 A0 ...] — add numeric dimensions at current values (all if no args)",
            "set_derived" => "set_derived d0 (make dimension display-only)",
            "set_driven" => "set_driven d0 [value|\"expr\"] (make dimension constraining)",
            "lock" => "lock P0 | lock L0.p1 | lock L0.p1 x,y",
            "unlock" => "unlock P0 | unlock L0.p1",
            "param" => "param name value | param name \"expr\" (creates or updates)",
            "del_param" => "del_param name",
            "rename_param" => "rename_param old_name new_name",
            "style" => "style L0 [solid|dashed|dashdot]",
            "quiet" => "quiet L0 [on|off] — toggle/set quiet mode (hides dimensions and center)",
            "constr" => "constr L0 [on|off] — toggle/set construction line (dashdot, different color)",
            "drag" => "drag L0.p1 x,y | drag L0 @dx,dy — drag entity/endpoint to position",
            "select" => "select L0 [L1 ...] | select all | select L0 chain | select L0 linked",
            "deselect" => "deselect [L0 L1 ...] (clears all or specific)",
            "print" => "print <expression> (evaluate and display)",
            "info" => "info L0 | info P0 | info A0 | info d0 | info paramname",
            "measure" => "measure L0 | measure L0 L1 | measure P0 P1 | measure L0 A0",
            "list" => "list [all|lines|points|arcs|dims|params|constraints|constr|selection]",
            "find" => "find x,y [radius] (list nearby entities)",
            "undo" => "undo [n]",
            "redo" => "redo [n]",
            "history" => "history [n] (show last n entries)",
            "goto" => "goto <position> (jump to history position)",
            "center" => "center L0 | center x,y | center (fit all)",
            "zoom" => "zoom + | zoom - | zoom 2.0",
            "msg" => "msg text — print message to history (supports markdown, \\n for newlines)",
            "cursor" => "cursor [x,y | @dx,dy | on | off] — show/set/hide command cursor",
            "dim_pos" => "dim_pos d0 offset 1.5 | dim_pos d0 along 0.3 (@ for relative)",
            "clear" => "clear (new empty sketch)",
            "add_arc" => "add_arc x1,y1 x2,y2 xm,ym (start, end, midpoint)",
            "add_earc" => "add_earc p1 p2 rx ry rot_deg [large] [cw]",
            "add_earc3" => "add_earc3 p1 p2 pmid rx ry",
            "add_earc_center" => "add_earc_center cx,cy rx ry rot_deg start_deg end_deg [cw]",
            "add_earc_tangent" => "add_earc_tangent p1 t1 p2 t2 [bulge] (tangent-defined, bulge=perp_dist/half_chord)",
            "add_earc_rtangent" => "add_earc_rtangent p2 t2 [bulge] (chain from cursor+tangent)",
            "offset_line" | "offset" => "offset_line L0 distance (create parallel line offset by distance)",
            "fillet" => "fillet L1 L2 r [notangent] [noradius]  or  fillet L1.pN r [notangent] [noradius] (round a corner with a tangent arc of radius r; breaks the shared LL coincident, trims both lines, adds arc + tangent + radius dim)",
            "chamfer" => "chamfer L1 L2 d  or  chamfer L1.pN d (bevel a corner at distance d from the corner; breaks the shared LL coincident, trims both lines by d, adds a bevel line + corner anchor point + two equal distance dims)",
            "let" => "let name = expression (session variable, scalar or coordinate)",
            "save" => "save path.json",
            "load" => "load path.json",
            "exit" | "quit" => "exit — close the application (blocked for MCP clients)",
            "dof" => "dof | dof analyze | dof eigenvalues [raw] | dof singular [raw] | dof jacobian",
            "perp" => "alias for perpendicular",
            other => return err(format!("help: unknown command: {}. Usage: help | help <command> | help full", other)),
        };
        ok(msg.to_string())
    }
}

// ---------------------------------------------------------------------------
// Autocomplete
// ---------------------------------------------------------------------------

const COMMAND_NAMES: &[&str] = &[
    "add_line", "add_rect", "add_rect3", "add_rectcenter",
    "add_point", "add_circle", "add_circle2", "add_circle3", "add_circle2t", "add_circle3t", "add_ellipse",
    "add_arc", "add_earc", "add_earc3", "add_earc_center", "add_earc_tangent", "add_earc_rtangent", "offset_line", "offset", "fillet", "chamfer",
    "delete", "horizontal", "vertical", "parallel", "perpendicular", "perp",
    "equal", "collinear", "tangent", "coincident", "concentric", "midpoint",
    "symmetry", "mirror", "point_on", "length", "radius", "radius_b", "sweep", "angle", "distance", "hdistance", "vdistance", "xangle",
    "set_derived", "set_driven",
    "lock", "unlock", "param", "del_param", "rename_param", "style", "quiet", "constr", "drag",
    "select", "deselect", "freeze", "print", "info", "measure", "list", "find", "let",
    "dof", "cost", "undo", "redo", "history", "goto", "center", "zoom",
    "cursor", "dim_pos", "clear", "save", "load", "help", "msg",
];

const GEO_FUNCTIONS: &[&str] = &[
    "intersect", "midpoint", "project", "along", "arc_point",
    "rotate", "mirror", "tangent", "normal", "dist", "angle",
];

// Named constants recognized by the expression parser (not functions).
const MATH_CONSTANTS: &[&str] = &["pi", "e"];

/// Generate autocomplete suggestions for the command input.
/// Returns completions for the word at `cursor_pos` in `input`.
pub fn complete(
    sketch: &Sketch,
    session_names: &HashMap<String, String>,
    input: &str,
    cursor_pos: usize,
) -> Vec<String> {
    // The caller's cursor is a byte position; floor it to a char
    // boundary so a cursor inside a multibyte char cannot panic.
    let mut end = cursor_pos.min(input.len());
    while end > 0 && !input.is_char_boundary(end) { end -= 1; }
    let input = &input[..end];
    let current_line = input.lines().last().unwrap_or("");
    let word_start = current_line.rfind(|c: char| c.is_whitespace()).map(|i| i + 1).unwrap_or(0);
    let current_word = &current_line[word_start..];
    let is_first_token = current_line[..word_start].trim().is_empty();

    // No completions when nothing typed on first token (would show all commands)
    if current_word.is_empty() && is_first_token { return Vec::new(); }

    // Dot completion (context-independent)
    if let Some(dot_pos) = current_word.rfind('.') {
        let before_dot = &current_word[..dot_pos];
        let after_dot = &current_word[dot_pos + 1..];
        let mut r = complete_after_dot(sketch, before_dot, after_dot);
        r.retain(|s| s != current_word);
        return r;
    }

    let mut results = Vec::new();

    // First token: command names only
    if is_first_token {
        add_matching(&mut results, current_word, COMMAND_NAMES);
        results.sort();
        results.dedup();
        results.truncate(20);
        return results;
    }

    // Non-first token: command-specific completions
    let first_cmd = current_line.split_whitespace().next().unwrap_or("");
    let token_index = current_line[..word_start].split_whitespace().count();
    // token_index: 1 = arg1, 2 = arg2, 3 = arg3

    // Collect already-completed args (excluding current word being typed)
    let typed_args: Vec<&str> = current_line[..word_start].split_whitespace().skip(1).collect();

    match first_cmd {
        // Variadic line commands: exclude already-typed lines
        "horizontal" | "vertical" => {
            add_lines_excluding(sketch, &mut results, current_word, &typed_args);
        }

        // Two-line commands: no suggestions after 2 args
        "parallel" | "perpendicular" | "perp" | "collinear" => {
            if token_index <= 2 {
                add_lines(sketch, &mut results, current_word);
            }
        }

        // Arc-only, exactly 2 args
        "concentric" => {
            if token_index <= 2 {
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Equal: match type of first arg, exactly 2 args
        "equal" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                let arg1 = current_line.split_whitespace().nth(1).unwrap_or("");
                if arg1.starts_with('L') {
                    add_lines(sketch, &mut results, current_word);
                } else if is_arc_name(arg1) {
                    add_arcs(sketch, &mut results, current_word);
                }
            }
        }

        // Tangent: line+arc or arc+arc, exactly 2 args
        "tangent" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Delete: entity / dimension / named constraint on first token,
        // plus the multi-token relational form (`delete L0 L1 parallel`)
        // offering entity completions and a constraint-type keyword
        // for the trailing slot.
        "delete" => {
            if token_index == 1 {
                add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
                add_dimensions(sketch, &mut results, current_word);
            } else if token_index <= 3 {
                add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
                add_matching(&mut results, current_word,
                    &["horizontal", "vertical", "parallel", "perpendicular",
                      "equal", "equal_length", "equal_radius", "collinear",
                      "tangent", "concentric", "coincident", "point_on",
                      "symmetry", "midpoint", "lock"]);
            }
        }
        "select" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["all"]);
            }
            add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
            if token_index == 2 {
                add_matching(&mut results, current_word, &["chain", "linked"]);
            }
        }

        // Single entity arg
        "info" | "center" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }

        // Endpoint commands: exactly 2 args
        "coincident" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }
        "lock" | "unlock" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }

        // Midpoint: arg1=point/endpoint, arg2=line
        "midpoint" => {
            if token_index == 1 {
                add_points(sketch, &mut results, current_word);
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_lines(sketch, &mut results, current_word);
            }
        }

        // Point_on: arg1=point/endpoint, arg2=line or arc
        "point_on" => {
            if token_index == 1 {
                add_points(sketch, &mut results, current_word);
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Symmetry: arg1=entity, arg2=line(mirror), arg3=entity
        "symmetry" => {
            if token_index <= 3 {
                if token_index == 2 {
                    add_lines(sketch, &mut results, current_word);
                } else {
                    add_all_entities(sketch, &mut results, current_word);
                }
            }
        }

        // Dimension: length (arg1=line, arg2=value/derived)
        "length" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: radius (arg1=arc, arg2=value/derived)
        "radius" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: radius_b (arg1=arc, arg2=value/derived) -- ellipse minor axis
        "radius_b" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: sweep (arg1=arc, arg2=value/derived)
        "sweep" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Freeze: lines and arcs
        "freeze" => {
            add_lines(sketch, &mut results, current_word);
            add_arcs(sketch, &mut results, current_word);
        }

        // Dimension: angle (arg1=line, arg2=line, arg3=value/derived)
        "angle" => {
            if token_index <= 2 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: distance (arg1=endpoint, arg2=endpoint/line, arg3=value/derived)
        "distance" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // hdistance/vdistance: arg1=endpoint, arg2=endpoint, arg3=value/derived
        "hdistance" | "vdistance" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // xangle: arg1=line, arg2=value/derived
        "xangle" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived", "driven"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension management: single dim arg
        "set_derived" | "set_driven" => {
            if token_index == 1 {
                add_dimensions(sketch, &mut results, current_word);
            }
        }

        // dim_pos: arg1=dim, arg2=offset/along, arg3=value
        "dim_pos" => {
            if token_index == 1 {
                add_dimensions(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["offset", "along"]);
            }
        }

        // Style: arg1=entity, arg2=style value
        "style" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["solid", "dashed", "dashdot"]);
            }
        }

        // List: single filter arg
        "list" => {
            if token_index == 1 {
                add_matching(&mut results, current_word,
                    &["all", "lines", "points", "arcs", "dims", "params", "constraints", "constr", "selection",
                      "horizontal", "vertical", "parallel", "perpendicular", "equal", "collinear",
                      "tangent", "coincident", "concentric", "midpoint", "symmetry", "point_on", "lock",
                      "angle", "length", "radius", "sweep", "distance"]);
            }
        }

        // Help: single arg (full or command name)
        "help" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["full"]);
                add_matching(&mut results, current_word, COMMAND_NAMES);
            }
        }

        // Cursor: single keyword arg
        "cursor" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["on", "off", "show", "hide"]);
            }
        }


        // Param commands: single param name
        "param" | "del_param" | "rename_param" => {
            if token_index == 1 {
                add_params(sketch, &mut results, current_word);
            }
        }

        // Offset: arg1=line, arg2=expression
        "offset_line" | "offset" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else {
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Fillet: arg1=line-or-endpoint, arg2=line-or-radius, arg3=radius,
        // trailing=keywords. Line completions first, then expressions,
        // then the fixed keyword options.
        "fillet" => {
            match token_index {
                1 => add_lines(sketch, &mut results, current_word),
                2 => {
                    add_lines(sketch, &mut results, current_word);
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
                _ => {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                    for k in &["notangent", "noradius"] {
                        if k.starts_with(current_word) { results.push((*k).to_string()); }
                    }
                }
            }
        }

        // Chamfer: same arg shape as fillet, no keyword options yet.
        "chamfer" => {
            match token_index {
                1 => add_lines(sketch, &mut results, current_word),
                2 => {
                    add_lines(sketch, &mut results, current_word);
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
                _ => {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
        }

        // Geometry creation: position-aware completions
        // add_line: [coord1] [coord2] [noconnect] [nocursor]
        // add_point: [coord] (no flags)
        // add_circle: [center] [radius] [noconnect] [nocursor]
        // add_arc: [start] [end] [mid] [noconnect] [nocursor]
        "add_line" => {
            let line_kws = ["noconnect", "nocursor", "notangent", "driven", "quiet"];
            let coord_args = typed_args.iter().filter(|a| !line_kws.contains(a)).count();
            if coord_args < 2 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 1 {
                for kw in &line_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_rect" | "add_rectcenter" => {
            let rect_kws = ["hv", "noconnect", "noconstraint", "driven", "strict"];
            let coord_args = typed_args.iter().filter(|a| !rect_kws.contains(a)).count();
            if coord_args < 2 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 2 {
                for kw in &rect_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_rect3" => {
            let rect_kws = ["noconnect", "noconstraint", "driven", "strict"];
            let coord_args = typed_args.iter().filter(|a| !rect_kws.contains(a)).count();
            if coord_args < 3 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 3 {
                for kw in &rect_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_point" => {
            let coord_args = typed_args.iter().filter(|a| **a != "nocursor").count();
            if coord_args < 1 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 1 && !typed_args.contains(&"nocursor") {
                add_matching(&mut results, current_word, &["nocursor"]);
            }
        }
        "add_circle" => {
            let circle_kws = ["noconnect", "nocursor", "driven", "quiet"];
            let coord_args = typed_args.iter().filter(|a| !circle_kws.contains(a)).count();
            if coord_args < 2 {
                if coord_args == 0 {
                    add_matching(&mut results, current_word, &["cursor"]);
                    add_all_entities(sketch, &mut results, current_word);
                    add_session_names(session_names, &mut results, current_word);
                } else {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
            if coord_args >= 2 {
                for kw in &circle_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_circle2" | "add_circle3" => {
            let circle_kws = ["noconnect", "nocursor", "driven", "quiet"];
            let max_coords = if first_cmd == "add_circle2" { 2 } else { 3 };
            let coord_args = typed_args.iter().filter(|a| !circle_kws.contains(a)).count();
            if coord_args < max_coords {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= max_coords {
                for kw in &circle_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_circle2t" => {
            let ct_kws = ["noconnect", "noconstraint", "driven", "strict"];
            let non_kw_args = typed_args.iter().filter(|a| !ct_kws.contains(a)).count();
            if non_kw_args < 2 {
                add_lines(sketch, &mut results, current_word);
            } else if non_kw_args == 2 {
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
            if non_kw_args >= 3 {
                for kw in &ct_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_circle3t" => {
            let ct_kws = ["noconnect", "noconstraint", "driven", "strict"];
            let non_kw_args = typed_args.iter().filter(|a| !ct_kws.contains(a)).count();
            if non_kw_args < 3 {
                add_lines(sketch, &mut results, current_word);
            }
            if non_kw_args >= 3 {
                for kw in &ct_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "add_ellipse" => {
            let ellipse_kws = ["noconnect", "nocursor", "driven", "quiet"];
            let coord_args = typed_args.iter().filter(|a| !ellipse_kws.contains(a)).count();
            if coord_args < 4 {
                if coord_args == 0 {
                    add_matching(&mut results, current_word, &["cursor"]);
                    add_all_entities(sketch, &mut results, current_word);
                    add_session_names(session_names, &mut results, current_word);
                } else {
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
            if coord_args >= 4 {
                for kw in &ellipse_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }
        "mirror" => {
            let has_about = typed_args.contains(&"about");
            if !has_about {
                // Before "about": offer entities and "selection"/"about"
                add_matching(&mut results, current_word, &["selection", "about"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            } else {
                // After "about": first the mirror line, then keywords
                let after_about_count = typed_args.iter().skip_while(|&&a| a != "about").skip(1)
                    .filter(|&&a| a != "noconstraint" && a != "strict").count();
                if after_about_count == 0 {
                    add_lines(sketch, &mut results, current_word);
                } else {
                    let mirror_kws = ["noconstraint", "strict"];
                    for kw in &mirror_kws {
                        if !typed_args.contains(kw) {
                            add_matching(&mut results, current_word, &[kw]);
                        }
                    }
                }
            }
        }
        "add_arc" => {
            let arc_kws = ["noconnect", "nocursor", "notangent", "quiet", "driven"];
            let coord_args = typed_args.iter().filter(|a| !arc_kws.contains(a)).count();
            if coord_args < 3 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 3 {
                for kw in &arc_kws {
                    if !typed_args.contains(kw) {
                        add_matching(&mut results, current_word, &[kw]);
                    }
                }
            }
        }

        // Expression-only: print, let
        "print" => {
            add_expression_completions(sketch, session_names, &mut results, current_word);
            add_all_entities(sketch, &mut results, current_word);
        }
        "let" => {
            add_expression_completions(sketch, session_names, &mut results, current_word);
            add_all_entities(sketch, &mut results, current_word);
        }

        // No completions for these
        // dof: single arg
        "dof" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["analyze"]);
            }
        }

        "undo" | "redo" | "history" | "goto" | "cost" | "clear"
        | "deselect" | "save" | "load" | "msg" | "find" | "zoom" => {}

        _ => {}
    }

    results.sort();
    results.dedup();
    results.truncate(20);
    results
}

// --- Completion helpers ---

fn add_matching(results: &mut Vec<String>, prefix: &str, candidates: &[&str]) {
    for &c in candidates {
        if c.starts_with(prefix) && c != prefix {
            results.push(c.to_string());
        }
    }
}

fn add_lines(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

fn add_arcs(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.arcs.refs() {
        let name = &sketch.arcs[r].name;
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

fn add_points(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        if p.helper { continue; }
        if p.name.starts_with(prefix) && p.name != prefix {
            results.push(p.name.clone());
        }
    }
}

fn add_all_entities(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    add_lines(sketch, results, prefix);
    add_points(sketch, results, prefix);
    add_arcs(sketch, results, prefix);
}

fn add_lines_excluding(sketch: &Sketch, results: &mut Vec<String>, prefix: &str, exclude: &[&str]) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
}

fn add_all_entities_excluding(sketch: &Sketch, results: &mut Vec<String>, prefix: &str, exclude: &[&str]) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        if p.helper { continue; }
        if p.name.starts_with(prefix) && p.name != prefix && !exclude.contains(&p.name.as_str()) {
            results.push(p.name.clone());
        }
    }
    for r in sketch.arcs.refs() {
        let name = &sketch.arcs[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
}

fn add_dimensions(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for d in &sketch.dimensions {
        if d.name.starts_with(prefix) && d.name != prefix {
            results.push(d.name.clone());
        }
    }
}

fn add_params(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for p in &sketch.user_params {
        if p.name.starts_with(prefix) && p.name != prefix {
            results.push(p.name.clone());
        }
    }
}

fn add_session_names(session_names: &HashMap<String, String>, results: &mut Vec<String>, prefix: &str) {
    for name in session_names.keys() {
        if name == "_" { continue; }
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

fn add_expression_completions(sketch: &Sketch, session_names: &HashMap<String, String>, results: &mut Vec<String>, prefix: &str) {
    add_dimensions(sketch, results, prefix);
    add_params(sketch, results, prefix);
    add_session_names(session_names, results, prefix);
    add_matching(results, prefix, GEO_FUNCTIONS);
    // Math functions come from arael-sym's authoritative table.
    for name in arael_sym::function_names() {
        if name.starts_with(prefix) && name != prefix {
            results.push(name.to_string());
        }
    }
    add_matching(results, prefix, MATH_CONSTANTS);
}

/// Complete after a dot: "L0." → ["p1", "p2"], "A0." → ["center", "start", "end"], etc.
fn complete_after_dot(sketch: &Sketch, before_dot: &str, after_dot: &str) -> Vec<String> {
    let mut results = Vec::new();

    // Check for double-dot: "L0.p1." → x, y
    if let Some(first_dot) = before_dot.rfind('.') {
        let entity = &before_dot[..first_dot];
        let prop = &before_dot[first_dot + 1..];
        // L<N>.p1. or L<N>.p2. → x, y
        if entity.starts_with('L') && (prop == "p1" || prop == "p2") {
            for &s in &["x", "y"] {
                if s.starts_with(after_dot) {
                    results.push(format!("{}.{}", before_dot, s));
                }
            }
        }
        // A<N>.center. → x, y
        if is_arc_name(entity) && prop == "center" {
            for &s in &["x", "y"] {
                if s.starts_with(after_dot) {
                    results.push(format!("{}.{}", before_dot, s));
                }
            }
        }
        // P<N>. after P<N>.pos or similar
        return results;
    }

    // Single dot: entity.suffix
    if before_dot.starts_with('L') && sketch.lines.refs().any(|r| sketch.lines[r].name == before_dot) {
        for &s in &["p1", "p2", "length", "angle"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    } else if is_arc_name(before_dot) && sketch.arcs.refs().any(|r| sketch.arcs[r].name == before_dot) {
        for &s in &["center", "start", "end", "radius", "start_angle", "end_angle"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    } else if before_dot.starts_with('P') && sketch.points.refs().any(|r| sketch.points[r].name == before_dot) {
        for &s in &["x", "y"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run(ctx: &mut CommandContext, cmd: &str) -> CommandResult {
        let results = execute(ctx, cmd);
        assert!(!results.is_empty());
        results.into_iter().next().unwrap()
    }

    fn run_ok(ctx: &mut CommandContext, cmd: &str) -> String {
        let r = run(ctx, cmd);
        assert!(!r.is_error, "Command '{}' failed: {}", cmd, r.output);
        r.output
    }

    fn run_err(ctx: &mut CommandContext, cmd: &str) -> String {
        let r = run(ctx, cmd);
        assert!(r.is_error, "Command '{}' should have failed but got: {}", cmd, r.output);
        r.output
    }

    fn line_len(ctx: &CommandContext, name: &str) -> f64 {
        let r = resolve_line(&ctx.sketch, name).unwrap();
        let l = &ctx.sketch.lines[r];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        (dx * dx + dy * dy).sqrt()
    }

    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 0.1 }

    // -- Geometry creation --

    #[test]
    fn test_add_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.sketch.lines.refs().count(), 1);
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 0.0));
        assert!(near(ctx.sketch.lines[r].p2.value.x, 5.0));
    }

    #[test]
    fn test_add_line_chaining() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line @0,3");
        let r = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
        assert!(near(ctx.sketch.lines[r].p2.value.y, 3.0));
    }

    #[test]
    fn test_add_line_single_arg() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,3");
        let r = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
        assert!(near(ctx.sketch.lines[r].p2.value.y, 3.0));
    }

    #[test]
    fn test_add_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 3,4");
        assert!(ctx.sketch.points.refs().count() >= 1);
    }

    #[test]
    fn test_add_circle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2");
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    }

    // -- Coordinate parsing --

    #[test]
    fn test_coord_endpoint_ref() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line L0.p2 10,0");
        let r = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
    }

    #[test]
    fn test_coord_relative() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 @3,4");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p2.value.x, 3.0));
        assert!(near(ctx.sketch.lines[r].p2.value.y, 4.0));
    }

    #[test]
    fn test_coord_expression() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param w 5");
        run_ok(&mut ctx, "add_line 0,0 w,0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p2.value.x, 5.0));
    }

    #[test]
    fn test_coord_geo_function() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        run_ok(&mut ctx, "add_point midpoint(L0)");
        let p = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[p].pos.value.x, 2.0));
    }

    // -- Constraints --

    #[test]
    fn test_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1");
        run_ok(&mut ctx, "horizontal L0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_vertical() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,5");
        run_ok(&mut ctx, "vertical L0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.vertical);
    }

    #[test]
    fn test_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "parallel L0 L1");
        assert!(!ctx.sketch.parallel.is_empty());
    }

    #[test]
    fn test_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,0.1 10,0");
        run_ok(&mut ctx, "coincident L0.p2 L1.p1");
        assert!(!ctx.sketch.coincident_ll21.is_empty());
    }

    // -- Dimensions --

    #[test]
    fn test_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        // Solve happened inside exec
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    #[test]
    fn test_hdistance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        assert!(near((l.p2.value.x - l.p1.value.x).abs(), 4.0));
    }

    #[test]
    fn test_vdistance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "vdistance L0.p1 L0.p2 2");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        assert!(near((l.p2.value.y - l.p1.value.y).abs(), 2.0));
    }

    #[test]
    fn test_xangle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "xangle L0 45");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let angle = dy.atan2(dx).to_degrees();
        assert!(near(angle, 45.0));
    }

    #[test]
    fn test_xangle_ellipse_rotation() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 2 1 0");
        run_ok(&mut ctx, "xangle EA0 30");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
        let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
        assert!(near(rot_deg, 30.0), "rotation = {rot_deg}");
        assert!(matches!(ctx.sketch.dimensions[0].kind,
                         DimensionKind::ArcRotation(_)));
    }

    #[test]
    fn test_xangle_ellipse_expression() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,5"); // L0 at 45deg
        run_ok(&mut ctx, "xangle L0 derived");
        run_ok(&mut ctx, "add_ellipse 10,10 2 1 0");
        // Ellipse rotation tracks L0's angle via the expression
        // pipeline. The reference is LIVE: the constraint couples the
        // rotation to L0's measured angle, and with both sides free
        // the solve distributes the correction -- so assert the
        // relation, not a fixed value.
        run_ok(&mut ctx, "xangle EA0 d0");
        let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
        let line = resolve_line(&ctx.sketch, "L0").unwrap();
        let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
        let l = &ctx.sketch.lines[line];
        let angle_deg = (l.p2.value.y - l.p1.value.y)
            .atan2(l.p2.value.x - l.p1.value.x).to_degrees();
        assert!(near(rot_deg, angle_deg), "rotation = {rot_deg}, L0 angle = {angle_deg}");
    }

    #[test]
    fn test_xangle_rejects_circular_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 0,1 -1,0");
        run_err(&mut ctx, "xangle A0 45");
    }

    #[test]
    fn test_xangle_ellipse_normalised() {
        // User input and effective rotation both fold into (-180, 180].
        // Value stored on the dimension is the normalised form, so
        // `list dims` / `info` / GUI never show a > 180 angle.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 2 1 0");
        run_ok(&mut ctx, "xangle EA0 200");
        let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
        let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
        assert!(near(rot_deg, -160.0), "rot = {rot_deg}");
        assert!(near(ctx.sketch.dimensions[0].value, -160.0));

        // Large-magnitude input (-540deg) folds to -180deg.
        run_ok(&mut ctx, "xangle EA0 -540");
        let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
        assert!(near(rot_deg, -180.0) || near(rot_deg, 180.0), "rot = {rot_deg}");
    }

    #[test]
    fn test_parallel_arc_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,5");
        run_ok(&mut ctx, "lock L0.p1 0,0");
        run_ok(&mut ctx, "lock L0.p2 5,5");
        run_ok(&mut ctx, "add_ellipse 10,10 2 1 0");
        run_ok(&mut ctx, "parallel L0 EA0");
        assert_eq!(ctx.sketch.arc_line_parallel.len(), 1);
        let arc = resolve_arc(&ctx.sketch, "EA0").unwrap();
        let rot_deg = ctx.sketch.arcs[arc].rotation.value.to_degrees();
        assert!(near(rot_deg, 45.0), "rotation = {rot_deg}");
    }

    #[test]
    fn test_parallel_arc_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 2 1 30");
        run_ok(&mut ctx, "lock EA0.center 0,0");
        run_ok(&mut ctx, "xangle EA0 30");
        run_ok(&mut ctx, "add_ellipse 5,5 3 1 0");
        run_ok(&mut ctx, "parallel EA0 EA1");
        assert_eq!(ctx.sketch.arc_arc_parallel.len(), 1);
        let ea1 = resolve_arc(&ctx.sketch, "EA1").unwrap();
        let rot_deg = ctx.sketch.arcs[ea1].rotation.value.to_degrees();
        assert!(near(rot_deg, 30.0), "EA1 rotation = {rot_deg}");
    }

    #[test]
    fn test_parallel_rejects_circular_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_arc 0,0 0,1 -1,0");
        run_err(&mut ctx, "parallel L0 A0");
    }

    #[test]
    fn test_parallel_dedup_arc_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,5");
        run_ok(&mut ctx, "add_ellipse 10,10 2 1 0");
        run_ok(&mut ctx, "parallel L0 EA0");
        run_err(&mut ctx, "parallel L0 EA0");
        run_err(&mut ctx, "parallel EA0 L0");
        assert_eq!(ctx.sketch.arc_line_parallel.len(), 1);
    }

    #[test]
    fn test_hdistance_update_and_remove() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 6");
        assert_eq!(ctx.sketch.dimensions.len(), 1); // updated, not duplicated
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        assert!(near((l.p2.value.x - l.p1.value.x).abs(), 6.0));
        run_ok(&mut ctx, "delete d0");
        assert_eq!(ctx.sketch.dimensions.len(), 0);
    }

    #[test]
    fn test_xangle_negative() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "xangle L0 -30");
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let angle = dy.atan2(dx).to_degrees();
        assert!(near(angle, -30.0));
    }

    #[test]
    fn test_xangle_update_and_remove() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "xangle L0 45");
        run_ok(&mut ctx, "xangle L0 60");
        assert_eq!(ctx.sketch.dimensions.len(), 1); // updated, not duplicated
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let angle = (l.p2.value.y - l.p1.value.y).atan2(l.p2.value.x - l.p1.value.x).to_degrees();
        assert!(near(angle, 60.0));
        run_ok(&mut ctx, "delete d0");
        assert_eq!(ctx.sketch.dimensions.len(), 0);
    }

    #[test]
    fn test_xangle_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "xangle L0 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before); // derived does not constrain
    }

    #[test]
    fn test_hdistance_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before);
    }

    #[test]
    fn test_vdistance_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "vdistance L0.p1 L0.p2 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before);
    }

    #[test]
    fn test_length_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "length L0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived); // constraining, not derived
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1); // DOF decreased
        assert!(near(line_len(&ctx, "L0"), (5.0f64 * 5.0 + 3.0 * 3.0).sqrt()));
    }

    #[test]
    fn test_radius_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "radius A0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_hdistance_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_xangle_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "xangle L0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_vdistance_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "vdistance L0.p1 L0.p2 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_sweep_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 5,0 0,5");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "sweep A0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_angle_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "angle L0 L1 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_distance_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 8,3 12,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "distance L0.p2 L1.p1 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_distance_pl_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,3; add_line 0,0 5,0");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "distance P0 L0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
    }

    #[test]
    fn test_sweep() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 5,0 0,5");
        run_ok(&mut ctx, "sweep A0 120");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let sweep_deg = arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs());
        assert!(near(sweep_deg, 120.0));
    }

    #[test]
    fn test_angle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
        run_ok(&mut ctx, "angle L0 L1 60");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_distance_pl_arc_end() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 3,0 0,3; add_line 0,10 5,10");
        run_ok(&mut ctx, "distance A0.end L0 7");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_end_l.len(), 1);
    }

    #[test]
    fn test_axis_distance_dof() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 1);
        run_ok(&mut ctx, "vdistance L0.p1 L0.p2 2");
        let dof_after2 = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after2, dof_after - 1);
    }

    #[test]
    fn test_hdistance_preserves_direction() {
        // hdistance is signed internally: can't swap endpoints to satisfy
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3");
        run_ok(&mut ctx, "hdistance L0.p1 L0.p2 4");
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        // p2.x should be to the right of p1.x (positive direction preserved)
        assert!(l.p2.value.x > l.p1.value.x);
    }

    #[test]
    fn test_axis_distance_ll_combinations() {
        // LL11: p1-p1
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
        run_ok(&mut ctx, "hdistance L0.p1 L1.p1 6");
        let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
        assert!(near((l1.p1.value.x - l0.p1.value.x).abs(), 6.0));

        // LL12: p1-p2
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
        run_ok(&mut ctx, "vdistance L0.p1 L1.p2 3");
        let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
        assert!(near((l1.p2.value.y - l0.p1.value.y).abs(), 3.0));

        // LL21: p2-p1
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
        run_ok(&mut ctx, "hdistance L0.p2 L1.p1 2");
        let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
        assert!(near((l1.p1.value.x - l0.p2.value.x).abs(), 2.0));

        // LL22: p2-p2
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_line 8,1 12,4");
        run_ok(&mut ctx, "vdistance L0.p2 L1.p2 5");
        let l0 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let l1 = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L1").unwrap()];
        assert!(near((l1.p2.value.y - l0.p2.value.y).abs(), 5.0));
    }

    #[test]
    fn test_axis_distance_lp_combinations() {
        // LP1: line.p1 to point
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_point 8,2");
        run_ok(&mut ctx, "hdistance L0.p1 P0 7");
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        assert!(near((p.pos.value.x - l.p1.value.x).abs(), 7.0));

        // LP2: line.p2 to point
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_point 8,2");
        run_ok(&mut ctx, "vdistance L0.p2 P0 4");
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        assert!(near((p.pos.value.y - l.p2.value.y).abs(), 4.0));

        // Reversed: point first
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,3; add_point 8,2");
        run_ok(&mut ctx, "hdistance P0 L0.p1 7");
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        assert!(near((p.pos.value.x - l.p1.value.x).abs(), 7.0));
    }

    #[test]
    fn test_axis_distance_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,0; add_point 5,3");
        run_ok(&mut ctx, "hdistance P0 P1 4");
        let p0 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        let p1 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P1").unwrap()];
        assert!(near((p1.pos.value.x - p0.pos.value.x).abs(), 4.0));

        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,0; add_point 5,3");
        run_ok(&mut ctx, "vdistance P0 P1 2");
        let p0 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        let p1 = &ctx.sketch.points[resolve_point(&ctx.sketch, "P1").unwrap()];
        assert!(near((p1.pos.value.y - p0.pos.value.y).abs(), 2.0));
    }

    #[test]
    fn test_axis_distance_arc_point() {
        // Arc center to point
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,5 2; add_point 10,3");
        run_ok(&mut ctx, "hdistance A0.center P0 3");
        let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        assert!(near((p.pos.value.x - a.center.value.x).abs(), 3.0));

        // Arc center to point, vdistance
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,5 2; add_point 10,3");
        run_ok(&mut ctx, "vdistance A0.center P0 4");
        let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        assert!(near((p.pos.value.y - a.center.value.y).abs(), 4.0));
    }

    #[test]
    fn test_axis_distance_arc_line() {
        // Arc center to line endpoint
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,5 2; add_line 10,3 15,7");
        run_ok(&mut ctx, "hdistance A0.center L0.p1 4");
        let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        assert!(near((l.p1.value.x - a.center.value.x).abs(), 4.0));
    }

    #[test]
    fn test_axis_distance_arc_arc() {
        // Arc center to arc center
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 1; add_circle 8,5 2");
        run_ok(&mut ctx, "hdistance A0.center A1.center 6");
        let a0 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
        assert!(near((a1.center.value.x - a0.center.value.x).abs(), 6.0));
    }

    #[test]
    fn test_distance_arc_center_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,5 2; add_point 10,3");
        run_ok(&mut ctx, "distance A0.center P0 4");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_center_p.len(), 1);
        let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let p = &ctx.sketch.points[resolve_point(&ctx.sketch, "P0").unwrap()];
        let dx = p.pos.value.x - a.center.value.x;
        let dy = p.pos.value.y - a.center.value.y;
        assert!(near((dx * dx + dy * dy).sqrt(), 4.0));
    }

    #[test]
    fn test_distance_arc_start_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 5,0 0,5; add_point 10,3");
        run_ok(&mut ctx, "distance A0.start P0 3");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_start_p.len(), 1);
    }

    #[test]
    fn test_distance_arc_end_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 5,0 0,5; add_point 10,3");
        run_ok(&mut ctx, "distance A0.end P0 4");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_end_p.len(), 1);
    }

    #[test]
    fn test_distance_arc_center_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,5 2; add_line 10,3 15,7");
        run_ok(&mut ctx, "distance A0.center L0.p1 3");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_center_l1.len(), 1);
        let a = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        let dx = l.p1.value.x - a.center.value.x;
        let dy = l.p1.value.y - a.center.value.y;
        assert!(near((dx * dx + dy * dy).sqrt(), 3.0));
    }

    #[test]
    fn test_distance_arc_start_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 5,0 0,5; add_line 10,3 15,7");
        run_ok(&mut ctx, "distance A0.start L0.p2 4");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_start_l2.len(), 1);
    }

    #[test]
    fn test_distance_arc_center_arc_center() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 1; add_circle 8,5 2");
        run_ok(&mut ctx, "distance A0.center A1.center 5");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_aa_ce_ce.len(), 1);
        let a0 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A0").unwrap()];
        let a1 = &ctx.sketch.arcs[resolve_arc(&ctx.sketch, "A1").unwrap()];
        let dx = a1.center.value.x - a0.center.value.x;
        let dy = a1.center.value.y - a0.center.value.y;
        assert!(near((dx * dx + dy * dy).sqrt(), 5.0));
    }

    #[test]
    fn test_distance_arc_start_arc_start() {
        // Use arcs with locked radii so solver can't collapse them
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 3,0 0,3; add_arc 20,0 23,0 20,3");
        run_ok(&mut ctx, "radius A0 3; radius A1 3");
        run_ok(&mut ctx, "distance A0.start A1.start 18");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_aa_s_s.len(), 1);
    }

    #[test]
    fn test_remove_dim() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        run_ok(&mut ctx, "delete d0");
        assert_eq!(ctx.sketch.dimensions.len(), 0);
    }

    // -- Parameters --

    #[test]
    fn test_param_create() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param width 10");
        assert_eq!(ctx.sketch.user_params.len(), 1);
        assert_eq!(ctx.sketch.user_params[0].name, "width");
    }

    #[test]
    fn test_param_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param w 5");
        run_ok(&mut ctx, "param w 10");
        assert_eq!(ctx.sketch.user_params[0].value, 10.0);
    }

    #[test]
    fn test_del_param() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param w 5; del_param w");
        assert!(ctx.sketch.user_params.is_empty());
    }

    #[test]
    fn test_rename_param() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param width 5; rename_param width w");
        assert_eq!(ctx.sketch.user_params[0].name, "w");
    }

    // -- Style --

    #[test]
    fn test_style_set() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "style L0 dashed");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert_eq!(ctx.sketch.lines[r].style, LineStyle::Dashed);
    }

    #[test]
    fn test_style_query() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "style L0");
        assert!(out.contains("solid"));
    }

    // -- Introspection --

    #[test]
    fn test_print_expr() {
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "print 2+3");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_print_entity() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        let out = run_ok(&mut ctx, "print L0.length");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_info_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "info L0");
        assert!(out.contains("L0"));
    }

    #[test]
    fn test_list() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_point 3,4");
        let out = run_ok(&mut ctx, "list");
        assert!(out.contains("L0"));
        assert!(out.contains("P0"));
    }

    #[test]
    fn test_list_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
        let out = run_ok(&mut ctx, "list constraints");
        assert!(out.contains("horizontal"));
    }

    #[test]
    fn test_find() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0");
        let out = run_ok(&mut ctx, "find 5,0 1");
        assert!(out.contains("L0"));
    }

    // -- Geometric functions --

    #[test]
    fn test_intersect() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line -1,-1 1,1; add_line -1,1 1,-1");
        run_ok(&mut ctx, "add_point intersect(L0,L1)");
        let p = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[p].pos.value.x, 0.0));
        assert!(near(ctx.sketch.points[p].pos.value.y, 0.0));
    }

    #[test]
    fn test_midpoint_func() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0; add_point midpoint(L0)");
        let p = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[p].pos.value.x, 2.0));
    }

    #[test]
    fn test_tangent_normal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        let out = run_ok(&mut ctx, "print tangent(L0)");
        assert!(out.contains("1.0"));
        let out = run_ok(&mut ctx, "print normal(L0)");
        assert!(out.contains("1.0")); // normal is (0,1), output has "1.0"
    }

    #[test]
    fn test_dist_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        let out = run_ok(&mut ctx, "print dist(L0.p1,L0.p2)");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_dist_pl() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,3; add_line 0,0 10,0");
        let out = run_ok(&mut ctx, "print dist(P0,L0)");
        assert!(out.contains("3"));
    }

    // -- Session variables --

    #[test]
    fn test_let_scalar() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "let d = 5");
        let out = run_ok(&mut ctx, "print d");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_let_vec() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        run_ok(&mut ctx, "let p = L0.p2");
        let out = run_ok(&mut ctx, "print p");
        assert!(out.contains("3") && out.contains("4"));
    }

    #[test]
    fn test_let_in_coord() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4; let p = L0.p2");
        run_ok(&mut ctx, "add_point p");
        let pt = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[pt].pos.value.x, 3.0));
    }

    // -- Selection --

    #[test]
    fn test_convert_dimension_preserves_identity() {
        // set_derived/set_driven used to delete and recreate the
        // dimension, churning its did and losing placement on the
        // driven path. ConvertDimension flips it in place.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0 noconnect");
        run_ok(&mut ctx, "length L0 3");
        let did = ctx.sketch.dimensions[0].did;
        run_ok(&mut ctx, "dim_pos d0 offset 2");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.has_length);

        run_ok(&mut ctx, "set_derived d0");
        let d = &ctx.sketch.dimensions[0];
        assert_eq!((d.did, d.name.as_str(), d.derived), (did, "d0", true));
        assert!(near(d.offset.y, 2.0), "placement must survive");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(!ctx.sketch.lines[r].constraints.has_length, "backing constraint must be gone");

        run_ok(&mut ctx, "set_driven d0 4");
        let d = &ctx.sketch.dimensions[0];
        assert_eq!((d.did, d.derived), (did, false));
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.has_length);
        assert!(near(line_len(&ctx, "L0"), 4.0));

        run_ok(&mut ctx, "undo");
        assert!(ctx.sketch.dimensions[0].derived, "undo reverts the conversion");
    }

    #[test]
    fn test_dimension_identity_survives_removal() {
        // Dimension actions used to carry Vec indices; removing one
        // dimension shifted the rest and the next removal deleted the
        // wrong one. Identity is the permanent did now.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
        run_ok(&mut ctx, "add_line 0,1 2,1 noconnect");
        run_ok(&mut ctx, "add_line 0,2 3,2 noconnect");
        run_ok(&mut ctx, "length L0 1");
        run_ok(&mut ctx, "length L1 2");
        run_ok(&mut ctx, "length L2 3");
        let did0 = ctx.sketch.dimensions[0].did;
        let did2 = ctx.sketch.dimensions[2].did;
        assert!(did0 != 0 && did2 != 0 && did0 != did2);
        // Remove first and third the way the GUI multi-delete does.
        ctx.begin_group();
        ctx.exec(Action::RemoveDimension { did: did0 });
        ctx.exec(Action::RemoveDimension { did: did2 });
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert_eq!(ctx.sketch.dimensions[0].name, "d1", "wrong dimension deleted");
        // A did that no longer resolves is a no-op, never a
        // wrong-target delete.
        ctx.exec(Action::RemoveDimension { did: did2 });
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_delete_by_name_is_nid_stable() {
        // ConstraintId used to carry a positional index resolved at
        // parse time; any retain in between shifted it and the wrong
        // constraint died. Identity is the nid now.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
        run_ok(&mut ctx, "add_line 0,1 1,1 noconnect");
        run_ok(&mut ctx, "add_line 0,2 1,2 noconnect");
        run_ok(&mut ctx, "parallel L0 L1");
        run_ok(&mut ctx, "parallel L1 L2");
        assert_eq!(ctx.sketch.parallel.len(), 2);
        let second_nid = ctx.sketch.parallel[1].nid;
        run_ok(&mut ctx, "delete C1");
        assert_eq!(ctx.sketch.parallel.len(), 1);
        assert_eq!(ctx.sketch.parallel[0].nid, second_nid, "wrong constraint deleted");
        run_ok(&mut ctx, "delete C2");
        assert!(ctx.sketch.parallel.is_empty());
    }

    #[test]
    fn test_hidden_helper_bridge_not_addressable() {
        // point_on with an arc endpoint mints a hidden bridge with the
        // next C number; deleting it used to cascade the user's
        // visible constraint away with the helper point.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        run_ok(&mut ctx, "add_circle 2,3 1");
        run_ok(&mut ctx, "point_on A0.start L0");
        assert_eq!(ctx.sketch.point_on_line.len(), 1);
        run_err(&mut ctx, "delete C2");
        assert_eq!(ctx.sketch.point_on_line.len(), 1, "bridge delete destroyed the user constraint");
        // The visible constraint stays addressable.
        run_ok(&mut ctx, "delete C1");
        assert!(ctx.sketch.point_on_line.is_empty());
    }

    #[test]
    fn test_history_covers_drag_dim_pos_and_relational_delete() {
        // drag: undo restores the pre-drag position instead of
        // undoing the previous action (drag was absent from history).
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
        run_ok(&mut ctx, "drag L0.p2 5,5");
        run_ok(&mut ctx, "undo");
        assert_eq!(ctx.sketch.lines.refs().count(), 1, "undo of drag deleted the line");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p2.value.x, 1.0));

        // dim_pos: placement is undoable.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
        run_ok(&mut ctx, "length L0 1");
        let before = ctx.sketch.dimensions[0].offset.y;
        run_ok(&mut ctx, "dim_pos d0 offset 3");
        assert!(near(ctx.sketch.dimensions[0].offset.y, 3.0));
        run_ok(&mut ctx, "undo");
        assert!(near(ctx.sketch.dimensions[0].offset.y, before), "undo must revert the placement");

        // relational delete: one undoable step through DeleteConstraint.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2");
        run_ok(&mut ctx, "add_circle 5,5 1 noconnect");
        run_ok(&mut ctx, "concentric A0 A1");
        assert_eq!(ctx.sketch.concentric.len(), 1);
        run_ok(&mut ctx, "delete A0 A1 concentric");
        assert!(ctx.sketch.concentric.is_empty());
        run_ok(&mut ctx, "undo");
        assert_eq!(ctx.sketch.concentric.len(), 1, "undo must restore the deleted constraint");
    }

    #[test]
    fn test_flags_survive_undo_redo() {
        // quiet/constr used to mutate the sketch after the history
        // snapshot (or outside history entirely): undo+redo lost the
        // flag, and undo after the standalone commands undid the
        // wrong action.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect constr");
        run_ok(&mut ctx, "undo");
        assert_eq!(ctx.sketch.lines.refs().count(), 0);
        run_ok(&mut ctx, "redo");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].construction, "constr flag lost across undo+redo");

        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
        run_ok(&mut ctx, "quiet L0 on");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].quiet);
        run_ok(&mut ctx, "undo");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(!ctx.sketch.lines[r].quiet, "undo must revert the quiet flag, not the line");
        assert_eq!(ctx.sketch.lines.refs().count(), 1, "undo of quiet must not delete the line");
    }

    #[test]
    fn test_created_identity_after_slot_reuse() {
        // The arena refills freed slots, so refs().last() after a
        // delete named a pre-existing entity and flags landed on it.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 noconnect");
        run_ok(&mut ctx, "add_line 2,0 3,0 noconnect");
        run_ok(&mut ctx, "delete L0");
        let out = run_ok(&mut ctx, "add_line 5,5 6,6 noconnect constr");
        assert!(out.contains("Added L2"), "reported name: {}", out);
        let new = resolve_line(&ctx.sketch, "L2").unwrap();
        assert!(ctx.sketch.lines[new].construction);
        let old = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(!ctx.sketch.lines[old].construction, "flag leaked to the reused-slot neighbor");
    }

    #[test]
    fn test_selection_pruned_after_delete() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,1");
        run_ok(&mut ctx, "select L0");
        run_ok(&mut ctx, "delete L0");
        assert!(ctx.selection.is_empty());
        // The stale-ref read that used to panic.
        run_ok(&mut ctx, "list selection");
    }

    #[test]
    fn test_selection_pruned_after_clear_and_undo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,1");
        run_ok(&mut ctx, "select L0");
        run_ok(&mut ctx, "clear");
        run_ok(&mut ctx, "list selection");
        assert!(ctx.selection.is_empty());

        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,1");
        run_ok(&mut ctx, "select L0");
        run_ok(&mut ctx, "undo");
        run_ok(&mut ctx, "list selection");
        assert!(ctx.selection.is_empty());
    }

    #[test]
    fn test_add_arc_collinear_points_rejected() {
        let mut ctx = CommandContext::new();
        run_err(&mut ctx, "add_arc 0,0 2,0 1,0");
        assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    }

    #[test]
    fn test_tangent_degenerate_geometry_rejected() {
        // Zero-length line: the tangent sign is 0/0 and the solve
        // never converges; must reject cleanly and fast. Creation of
        // such a line is rejected at the gate, but a solve can still
        // collapse one -- force the state through the value door.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "add_circle 5,5 1");
        let l = ctx.sketch.lines.refs().next().unwrap();
        ctx.sketch.mutate_values(|s| {
            let p1 = s.lines[l].p1.value;
            s.lines[l].p2.value = p1;
        });
        let msg = run_err(&mut ctx, "tangent L0 A0");
        assert!(msg.contains("zero length"), "{}", msg);

        // Concentric arcs: tangent divides by center distance.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 2,2 1");
        run_ok(&mut ctx, "add_circle 2,2 3");
        let msg = run_err(&mut ctx, "tangent A0 A1");
        assert!(msg.contains("concentric"), "{}", msg);
    }

    #[test]
    fn test_delete_dim_removes_line_endpoint_arc_backing_constraint() {
        // Line-endpoint x arc-point distance: the backing constraint
        // lives in distance_arc_center_l1 and must die with the dim.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "add_circle 6,3 1 noconnect");
        run_ok(&mut ctx, "distance L0.p1 A0.center 6");
        assert_eq!(ctx.sketch.distance_arc_center_l1.len(), 1);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        run_ok(&mut ctx, "delete d0");
        assert_eq!(ctx.sketch.dimensions.len(), 0);
        assert!(ctx.sketch.distance_arc_center_l1.is_empty(),
            "backing constraint must be deleted with the dimension");
    }

    #[test]
    fn test_redo_restores_post_group_cursor() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "add_line 4,0 4,3 noconnect");
        run_ok(&mut ctx, "add_line 4,3 0,3 noconnect");
        run_ok(&mut ctx, "undo 2");
        let c = ctx.cursor.unwrap();
        assert!((c.x - 4.0).abs() < 1e-9 && c.y.abs() < 1e-9, "{:?}", (c.x, c.y));
        // Redo of the middle group restores the cursor state that
        // followed it (the pre-state of the next group).
        run_ok(&mut ctx, "redo");
        let c = ctx.cursor.unwrap();
        assert!((c.x - 4.0).abs() < 1e-9 && (c.y - 3.0).abs() < 1e-9, "{:?}", (c.x, c.y));
    }

    #[test]
    fn test_output_surfaces_ids() {
        // Rect constraints carry their ids.
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "add_rect 0,0 4,3");
        assert!(out.contains("(C"), "{}", out);
        // Auto-coincident feedback names the constraint it added.
        let out = run_ok(&mut ctx, "add_line 4,3 6,5");
        assert!(out.contains("[connected:") && out.contains("(C"), "{}", out);
        // Freeze names the dimension it created.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        let out = run_ok(&mut ctx, "freeze L0");
        assert!(out.contains("d0"), "{}", out);
        // Delete-relational names the removed constraint.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect; add_line 0,2 4,2 noconnect; parallel L0 L1");
        let out = run_ok(&mut ctx, "delete L0 L1 parallel");
        assert!(out.contains("C1"), "{}", out);
    }

    #[test]
    fn test_dof_commands_error_on_degenerate_geometry() {
        // Solver-collapsed zero-length line under a point-on-line
        // residual: every dof command variant errors cleanly.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "add_point 2,1");
        run_ok(&mut ctx, "point_on P0 L0");
        let l = ctx.sketch.lines.refs().next().unwrap();
        ctx.sketch.mutate_values(|s| {
            let p1 = s.lines[l].p1.value;
            s.lines[l].p2.value = p1;
            // Value-only change that alters the instantaneous rank.
            s.clear_cached_dof();
        });
        for cmd in ["dof", "dof analyze", "dof eigenvalues", "dof singular"] {
            let msg = run_err(&mut ctx, cmd);
            assert!(msg.contains("non-finite"), "{}: {}", cmd, msg);
        }
    }

    #[test]
    fn test_expr_radius_follows_drag_frames() {
        // A circle whose radius is the expression `d+2`, with `d` a
        // derived distance to the dragged endpoint: the radius must
        // track during per-frame drag solves, not only at drag end.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "l = add_line 0,0 10,0");
        run_ok(&mut ctx, "lock l.p1; horizontal l; length l 10");
        run_ok(&mut ctx, "pivot = add_line l.p2 @5,0");
        run_ok(&mut ctx, "d = distance l.p1 pivot.p2 derived");
        run_ok(&mut ctx, "c = add_circle l.p1 20");
        run_ok(&mut ctx, "radius c d+2");
        let circle = ctx.sketch.arcs.refs().next().unwrap();
        let pivot = ctx.sketch.lines.refs().nth(1).unwrap();
        // The live constraint holds the relation radius = d + 2; with
        // both sides free the solve distributes the correction.
        let dist = |s: &Sketch| {
            let p = s.lines[pivot].p2.value;
            (p.x * p.x + p.y * p.y).sqrt()
        };
        let r0 = ctx.sketch.arcs[circle].radius.value;
        assert!((r0 - (dist(&ctx.sketch) + 2.0)).abs() < 0.1,
            "creation: radius {} vs d+2 {}", r0, dist(&ctx.sketch) + 2.0);

        // Simulate GUI drag frames on pivot.p2: install the
        // apparatus, move the helper, per-frame cell solve.
        let grab_pos = ctx.sketch.lines[pivot].p2.value;
        let app = ctx.sketch.get_mut().install_drag(
            arael_sketch_solver::DragTarget::LineP2(pivot),
            grab_pos, None, Some(DRAG_PULL_WEIGHT));
        ctx.sketch.mutate_values(|s| s.move_drag_helper(app.helper, vect2d::new(25.0, 0.0)));
        ctx.sketch.solve();
        let mid_radius = ctx.sketch.arcs[circle].radius.value;
        let mid_dist = dist(&ctx.sketch);
        ctx.sketch.get_mut().remove_drag(&app);
        assert!(mid_dist > r0 + 1.0, "drag must have pulled the pivot out, d at {}", mid_dist);
        assert!((mid_radius - (mid_dist + 2.0)).abs() < 0.5,
            "radius {} must track d+2 (~{}) during the drag frame", mid_radius, mid_dist + 2.0);
    }

    #[test]
    fn test_all_trailing_keywords_are_peeled() {
        // The circle family's hand-rolled peel loop capped at 3 of
        // its 5 keywords, silently leaving the rest as arguments.
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "add_circle 2,2 1 nocursor noconnect quiet constr driven");
        assert!(out.contains("[driven") && out.contains("[quiet]"), "{}", out);
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!(ctx.sketch.arcs[r].construction, "constr keyword must be honored");
        assert!(ctx.cursor.is_none(), "nocursor keyword must be honored");
    }

    #[test]
    fn test_deselect_unknown_entity_errors() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0; select L0");
        let msg = run_err(&mut ctx, "deselect L99");
        assert!(msg.contains("Unknown line"), "{}", msg);
        let msg = run_err(&mut ctx, "deselect x9");
        assert!(msg.contains("Unknown entity"), "{}", msg);
        // Deselecting an existing but unselected entity stays a no-op.
        run_ok(&mut ctx, "add_line 0,2 4,2; deselect L1");
        assert_eq!(ctx.selection.len(), 1);
    }

    #[test]
    fn test_driven_fragment_reports_rejection() {
        // A rejected driven dimension must not report the previous
        // dimension's name as if it were the new one.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "length L0 4");
        let line = ctx.sketch.lines.refs().next().unwrap();
        let frag = driven_dim_fragment(&mut ctx, Action::AddDimension {
            kind: DimensionKind::LineLength(line),
            value: 4.0, expr: None, derived: false, range: None,
        }, "length", 4.0);
        assert!(frag.contains("rejected"), "{}", frag);
        assert!(!frag.contains("d0 "), "{}", frag);

        // The success path names the dimension actually created.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        let line = ctx.sketch.lines.refs().next().unwrap();
        let frag = driven_dim_fragment(&mut ctx, Action::AddDimension {
            kind: DimensionKind::LineLength(line),
            value: 4.0, expr: None, derived: false, range: None,
        }, "length", 4.0);
        assert!(frag.contains("d0") && frag.contains("length=4.0000"), "{}", frag);
    }

    #[test]
    fn test_info_reaches_user_params_starting_with_d() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        run_ok(&mut ctx, "param depth = 2.5");
        let out = run_ok(&mut ctx, "info depth");
        assert!(out.contains("depth") && out.contains("2.5"), "{}", out);
        // Numeric d-names still report as dimensions.
        let out = run_err(&mut ctx, "info d99");
        assert!(out.contains("Unknown dimension"), "{}", out);
    }

    #[test]
    fn test_force_strip_ordering() {
        // A comment must not hide a trailing force keyword.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "add_line 0,2 4,2 noconnect");
        run_ok(&mut ctx, "length L0 4");
        run_ok(&mut ctx, "length L1 4");
        run_ok(&mut ctx, "equal L0 L1 force # redundant on purpose");
        assert_eq!(ctx.sketch.equal_length.len(), 1);

        // msg text is verbatim: no force stripping, no comment stripping.
        let out = run_ok(&mut ctx, "msg use the force # not a comment");
        assert_eq!(out, "use the force # not a comment");
    }

    #[test]
    fn test_classify_direction_with_radius_is_not_rigid_motion() {
        // A radius-grow direction moves two on-circle points diagonally
        // and changes the radius: neither a rotation nor a translation.
        let parts = vec![
            ("P1.x".to_string(), 0.7), ("P1.y".to_string(), 0.7),
            ("P2.x".to_string(), -0.7), ("P2.y".to_string(), 0.7),
            ("A0.radius".to_string(), 1.0),
        ];
        let label = classify_free_direction(&parts);
        assert!(!label.starts_with("rotate") && !label.starts_with("translate"), "{}", label);

        // Radius plus one moving coordinate must not read as translate X.
        let parts = vec![
            ("A0.center.x".to_string(), 1.0),
            ("A0.center.y".to_string(), 0.0),
            ("A0.radius".to_string(), 1.0),
        ];
        let label = classify_free_direction(&parts);
        assert!(!label.starts_with("translate"), "{}", label);

        // Control: a genuine rotation still classifies as one.
        let parts = vec![
            ("P1.x".to_string(), 0.7), ("P1.y".to_string(), 0.7),
            ("P2.x".to_string(), -0.7), ("P2.y".to_string(), 0.7),
        ];
        let label = classify_free_direction(&parts);
        assert!(label.starts_with("rotate"), "{}", label);
    }

    #[test]
    fn test_command_path_runs_conflict_checks() {
        // Transitive H/V contradiction through a parallel link used to
        // pass the command path (only the GUI ran the conflict checks)
        // and collapse L1 to zero length.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,5 noconnect");
        run_ok(&mut ctx, "add_line 2,0 2,5 noconnect");
        run_ok(&mut ctx, "vertical L0");
        run_ok(&mut ctx, "parallel L0 L1");
        let msg = run_err(&mut ctx, "horizontal L1");
        assert!(msg.contains("vertical"), "{}", msg);

        // force overrides only the DOF check, never a contradiction.
        let msg = run_err(&mut ctx, "horizontal L0 force");
        assert!(msg.contains("vertical"), "{}", msg);
    }

    #[test]
    fn test_collinear_transitivity_rejected() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0 noconnect");
        run_ok(&mut ctx, "add_line 5,0 9,0 noconnect");
        run_ok(&mut ctx, "add_line 10,0 14,0 noconnect");
        run_ok(&mut ctx, "collinear L0 L1");
        run_ok(&mut ctx, "collinear L1 L2");
        let msg = run_err(&mut ctx, "collinear L0 L2");
        assert!(msg.contains("already collinear"), "{}", msg);
    }

    #[test]
    fn test_degenerate_creation_rejected() {
        let mut ctx = CommandContext::new();
        let msg = run_err(&mut ctx, "add_line 1,1 1,1");
        assert!(msg.contains("zero length"), "{}", msg);
        let msg = run_err(&mut ctx, "add_circle 0,0 0");
        assert!(msg.contains("zero radius"), "{}", msg);
        assert_eq!(ctx.sketch.lines.refs().count(), 0);
        assert_eq!(ctx.sketch.arcs.refs().count(), 0);
    }

    #[test]
    fn test_select() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; select L0");
        assert_eq!(ctx.selection.len(), 1);
    }

    #[test]
    fn test_deselect() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; select L0; deselect");
        assert!(ctx.selection.is_empty());
    }

    // -- Undo/redo --

    #[test]
    fn test_undo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.sketch.lines.refs().count(), 1);
        run_ok(&mut ctx, "undo");
        assert_eq!(ctx.sketch.lines.refs().count(), 0);
    }

    #[test]
    fn test_redo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; undo; redo");
        assert_eq!(ctx.sketch.lines.refs().count(), 1);
    }

    #[test]
    fn test_history() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 1,1 2,2");
        let out = run_ok(&mut ctx, "history");
        assert!(out.contains("Add line"));
    }

    // -- Remove constraint --

    #[test]
    fn test_remove_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
        run_ok(&mut ctx, "delete L0 horizontal");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(!ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_remove_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1; parallel L0 L1");
        assert!(!ctx.sketch.parallel.is_empty());
        run_ok(&mut ctx, "delete L0 L1 parallel");
        assert!(ctx.sketch.parallel.is_empty());
    }

    // -- Multi-command --

    #[test]
    fn test_semicolon() {
        let mut ctx = CommandContext::new();
        let results = execute(&mut ctx, "add_line 0,0 5,0; horizontal L0");
        // Two commands + one trailing DOF summary line.
        assert_eq!(results.len(), 3);
        assert!(!results[0].is_error);
        assert!(!results[1].is_error);
        assert!(results[2].output.starts_with("DOF:"));
    }

    // -- Error handling --

    #[test]
    fn test_unknown_command() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "foobar"); }

    #[test]
    fn test_unknown_entity() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "info L99"); }

    #[test]
    fn test_bad_coord() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "add_line abc xyz"); }

    #[test]
    fn test_help() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "help"); }

    #[test]
    fn test_help_command() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "help add_line"); }

    #[test]
    fn test_dof() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "dof"); }

    #[test]
    fn test_cost() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "cost"); }

    // -- Entity name capture --

    #[test]
    fn test_auto_underscore() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1");
        assert!(ctx.session_names.contains_key("_"));
        run_ok(&mut ctx, "vertical _");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.vertical);
    }

    #[test]
    fn test_auto_underscore_updates() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.session_names["_"], "L0");
        run_ok(&mut ctx, "add_line 1,1 2,2");
        assert_eq!(ctx.session_names["_"], "L1");
    }

    #[test]
    fn test_assign_entity_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "base = add_line 0,0 5,1");
        assert_eq!(ctx.session_names.get("base").unwrap(), "L0");
        run_ok(&mut ctx, "horizontal base");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_let_entity_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "let l = add_line 0,0 5,1");
        assert_eq!(ctx.session_names.get("l").unwrap(), "L0");
        run_ok(&mut ctx, "horizontal l");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_let_entity_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "a = add_line 0,0 5,0");
        run_ok(&mut ctx, "b = add_line 5,0.1 10,0");
        run_ok(&mut ctx, "coincident a.p2 b.p1");
        assert!(!ctx.sketch.coincident_ll21.is_empty());
    }

    #[test]
    fn test_let_entity_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "l = add_line 0,0 5,0");
        run_ok(&mut ctx, "length l 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    #[test]
    fn test_let_entity_info() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "l = add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "info l");
        assert!(out.contains("L0"));
    }

    #[test]
    fn test_underscore_chain() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; horizontal _; length _ 3");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    // -- Auto-coincident --

    #[test]
    fn test_auto_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_line 5,0 5,3");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        // L1.p1==L0.p2 -> coincident_ll12 (a.p1 == b.p2 where a=L1, b=L0)
        let has_coincident = !ctx.sketch.coincident_ll12.is_empty()
            || !ctx.sketch.coincident_ll21.is_empty()
            || !ctx.sketch.coincident_ll11.is_empty()
            || !ctx.sketch.coincident_ll22.is_empty();
        assert!(has_coincident, "Should have coincident constraint");
    }

    #[test]
    fn test_noconnect() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_line 5,0 5,3 noconnect");
        assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
        assert!(ctx.sketch.coincident_ll21.is_empty());
    }

    // -- Auto-coincident for arcs/circles --

    #[test]
    fn test_auto_coincident_circle_to_line_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_circle 5,0 1");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(out.contains("A0.center=L0.p2"), "Should mention A0.center=L0.p2: {}", out);
        assert!(!ctx.sketch.coincident_lp2_arc_center.is_empty(),
            "Should have coincident_lp2_arc_center");
    }

    #[test]
    fn test_auto_coincident_circle_to_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 3,3");
        let out = run_ok(&mut ctx, "add_circle 3,3 1");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(!ctx.sketch.coincident_arc_center.is_empty(),
            "Should have coincident_arc_center");
    }

    #[test]
    fn test_auto_coincident_circle_concentric() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2");
        let out = run_ok(&mut ctx, "add_circle 0,0 3");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(out.contains("A1.center=A0.center"), "Should mention concentric: {}", out);
        assert!(!ctx.sketch.concentric.is_empty(), "Should have concentric constraint");
    }

    #[test]
    fn test_auto_coincident_line_to_arc_center() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,3 1");
        let out = run_ok(&mut ctx, "add_line 0,0 5,3");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(out.contains("L0.p2=A0.center"), "Should mention A0.center: {}", out);
        assert!(!ctx.sketch.coincident_lp2_arc_center.is_empty(),
            "Should have coincident_lp2_arc_center");
    }

    #[test]
    fn test_noconnect_circle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_circle 5,0 1 noconnect");
        assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
        assert!(ctx.sketch.coincident_lp2_arc_center.is_empty());
    }

    #[test]
    fn test_noconnect_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_arc 5,0 5,3 6,1.5 noconnect");
        assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
    }

    // -- Duplicate constraint rejection --

    #[test]
    fn test_duplicate_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        let e = run_err(&mut ctx, "horizontal L0");
        assert!(e.contains("already horizontal"));
    }

    #[test]
    fn test_duplicate_vertical() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,5");
        run_ok(&mut ctx, "vertical L0");
        let e = run_err(&mut ctx, "vertical L0");
        assert!(e.contains("already vertical"));
    }

    #[test]
    fn test_duplicate_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "parallel L0 L1");
        let e = run_err(&mut ctx, "parallel L0 L1");
        assert!(e.contains("already exists"));
        let e = run_err(&mut ctx, "parallel L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_perpendicular() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
        run_ok(&mut ctx, "perpendicular L0 L1");
        let e = run_err(&mut ctx, "perpendicular L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_equal_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "equal L0 L1");
        let e = run_err(&mut ctx, "equal L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_equal_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 3");
        run_ok(&mut ctx, "equal A0 A1");
        let e = run_err(&mut ctx, "equal A1 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_collinear() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 6,0 10,0");
        run_ok(&mut ctx, "collinear L0 L1");
        let e = run_err(&mut ctx, "collinear L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_tangent_la() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 2.5,1 1");
        run_ok(&mut ctx, "tangent L0 A0");
        let e = run_err(&mut ctx, "tangent L0 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_tangent_aa() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 2");
        run_ok(&mut ctx, "tangent A0 A1");
        let e = run_err(&mut ctx, "tangent A1 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_coincident_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
        run_ok(&mut ctx, "coincident L0.p2 L1.p1");
        let e = run_err(&mut ctx, "coincident L0.p2 L1.p1");
        assert!(e.contains("already exists"));
        // Cross-type: L0.p2=L1.p1 is same as L1.p1=L0.p2 (swapped order)
        let e = run_err(&mut ctx, "coincident L1.p1 L0.p2");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_concentric() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 3");
        run_ok(&mut ctx, "concentric A0 A1");
        let e = run_err(&mut ctx, "concentric A1 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_point_on_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2.5,0; add_line 0,0 5,0");
        run_ok(&mut ctx, "point_on P0 L0");
        let e = run_err(&mut ctx, "point_on P0 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_midpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2.5,0; add_line 0,0 5,0");
        run_ok(&mut ctx, "midpoint P0 L0");
        let e = run_err(&mut ctx, "midpoint P0 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_midpoint_arc_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -4,0 4,0 0,4; add_point 0,5");
        run_ok(&mut ctx, "midpoint P0 A0");
        assert_eq!(ctx.sketch.midpoint_arc_point.len(), 1);
        // Duplicate check
        let e = run_err(&mut ctx, "midpoint P0 A0");
        assert!(e.contains("already exists"), "{}", e);
    }

    #[test]
    fn test_midpoint_arc_line_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -4,0 4,0 0,4; add_line -1,5 1,5");
        run_ok(&mut ctx, "midpoint L0.p1 A0");
        assert_eq!(ctx.sketch.midpoint_lp1_arc.len(), 1);
    }

    #[test]
    fn test_midpoint_arc_circle_rejected() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_point 0,5");
        let e = run_err(&mut ctx, "midpoint P0 A0");
        assert!(e.contains("full circle"), "{}", e);
    }

    #[test]
    fn test_remove_constraint_midpoint_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -4,0 4,0 0,4; add_point 0,5");
        run_ok(&mut ctx, "midpoint P0 A0");
        assert_eq!(ctx.sketch.midpoint_arc_point.len(), 1);
        run_ok(&mut ctx, "delete P0 A0 midpoint");
        assert_eq!(ctx.sketch.midpoint_arc_point.len(), 0);
    }

    #[test]
    fn test_duplicate_symmetry_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line -2,0 -2,3; add_line 0,0 0,5; add_line 2,0 2,3");
        run_ok(&mut ctx, "symmetry L0 L1 L2");
        let e = run_err(&mut ctx, "symmetry L2 L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_self_reference_rejected() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 0,0 2");
        let e = run_err(&mut ctx, "parallel L0 L0");
        assert!(e.contains("itself"));
        let e = run_err(&mut ctx, "equal L0 L0");
        assert!(e.contains("itself"));
        let e = run_err(&mut ctx, "concentric A0 A0");
        assert!(e.contains("itself"));
    }

    // -- Info with constraints --

    #[test]
    fn test_info_shows_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
        let out = run_ok(&mut ctx, "info L0");
        assert!(out.contains("horizontal"), "info should show constraints: {}", out);
    }

    #[test]
    fn test_info_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "info L0.p1");
        assert!(out.contains("0.0000"), "info L0.p1 should show position: {}", out);
    }

    // -- Select endpoints --

    #[test]
    fn test_select_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "select L0.p1");
        assert_eq!(ctx.selection.len(), 1);
        assert!(matches!(ctx.selection[0], Selection::LineP1(_)));
    }

    // -- Param shows value --

    #[test]
    fn test_param_shows_value() {
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "param kala 12+3*4");
        assert!(out.contains("24"), "Should show evaluated value: {}", out);
    }

    // -- Cursor --

    #[test]
    fn test_cursor_set() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 5,3");
        assert!(ctx.cursor.is_some());
        assert!(near(ctx.cursor.unwrap().x, 5.0));
        assert!(near(ctx.cursor.unwrap().y, 3.0));
    }

    #[test]
    fn test_cursor_from_add_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert!(ctx.cursor.is_some());
        assert!(near(ctx.cursor.unwrap().x, 5.0));
        assert!(near(ctx.cursor.unwrap().y, 0.0));
    }

    #[test]
    fn test_cursor_relative() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 1,1");
        run_ok(&mut ctx, "cursor @2,3");
        assert!(near(ctx.cursor.unwrap().x, 3.0));
        assert!(near(ctx.cursor.unwrap().y, 4.0));
    }

    #[test]
    fn test_cursor_as_coord() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 5,0");
        run_ok(&mut ctx, "add_line cursor 5,3");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
        assert!(near(ctx.sketch.lines[r].p1.value.y, 0.0));
    }

    #[test]
    fn test_cursor_off() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 5,3");
        assert!(ctx.cursor.is_some());
        run_ok(&mut ctx, "cursor off");
        assert!(ctx.cursor.is_none());
    }

    #[test]
    fn test_cursor_nocursor() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 1,1");
        run_ok(&mut ctx, "add_line 0,0 5,0 nocursor");
        // Cursor should still be at 1,1, not moved to 5,0
        assert!(near(ctx.cursor.unwrap().x, 1.0));
    }

    #[test]
    fn test_cursor_endpoint_ref() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "cursor L0.p1");
        assert!(near(ctx.cursor.unwrap().x, 0.0));
    }

    #[test]
    fn test_cursor_query() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 3,7");
        let out = run_ok(&mut ctx, "cursor");
        assert!(out.contains("3.0000") && out.contains("7.0000"));
    }

    // -- Dimension text position --

    #[test]
    fn test_dim_pos_offset() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        run_ok(&mut ctx, "dim_pos d0 offset 2.0");
        assert!(near(ctx.sketch.dimensions[0].offset.y, 2.0));
    }

    #[test]
    fn test_dim_pos_along() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        run_ok(&mut ctx, "dim_pos d0 along 0.5");
        assert!(near(ctx.sketch.dimensions[0].text_along, 0.5));
    }

    #[test]
    fn test_dim_info_shows_position() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        let out = run_ok(&mut ctx, "info d0");
        assert!(out.contains("offset=") && out.contains("along="));
    }

    // -- Point symmetry command --

    #[test]
    fn test_cmd_symmetry_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 3,2; add_point 7,2; add_line 5,0 5,10");
        run_ok(&mut ctx, "symmetry P0 L0 P1");
        assert!(!ctx.sketch.symmetry_pp.is_empty());
    }

    // -- Derived dimensions --

    #[test]
    fn test_cmd_derived_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 5 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        // Derived should NOT constrain — line length should stay at original ~5
        // (no has_length constraint set)
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(!ctx.sketch.lines[r].constraints.has_length);
    }

    #[test]
    fn test_cmd_set_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        assert!(!ctx.sketch.dimensions[0].derived);
        run_ok(&mut ctx, "set_derived d0");
        // Find the derived dim (might be re-added with same name)
        let dim = ctx.sketch.dimensions.iter().find(|d| d.name == "d0");
        assert!(dim.is_some());
        assert!(dim.unwrap().derived);
    }

    #[test]
    fn test_cmd_set_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        // Create derived dim first
        run_ok(&mut ctx, "length L0 5 derived");
        assert!(ctx.sketch.dimensions[0].derived);
        run_ok(&mut ctx, "set_driven d0 3");
        // Should now be driven
        let dim = ctx.sketch.dimensions.last().unwrap();
        assert!(!dim.derived);
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    #[test]
    fn test_cmd_derived_length_measure() {
        // "length L0 derived" should measure current geometry
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        run_ok(&mut ctx, "length L0 derived");
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 5.0));
    }

    #[test]
    fn test_cmd_derived_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3");
        run_ok(&mut ctx, "radius A0 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 3.0));
    }

    #[test]
    fn test_cmd_derived_angle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
        run_ok(&mut ctx, "angle L0 L1 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 90.0));
    }

    #[test]
    fn test_cmd_derived_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 4,0 7,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 1.0));
    }

    // -- Helper point display and cleanup tests --

    fn has_helper_points(ctx: &CommandContext) -> bool {
        ctx.sketch.points.refs().any(|r| ctx.sketch.points[r].helper)
    }

    fn list_constraints_output(ctx: &mut CommandContext) -> String {
        run_ok(ctx, "list constraints")
    }

    // 6A: Display tests -- list constraints shows no Pc names

    #[test]
    fn test_list_no_pc_distance_ll_endpoints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
        assert!(out.contains("distance"), "should list distance constraint: {}", out);
    }

    #[test]
    fn test_list_no_pc_distance_arc_endpoints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center A1.center 10");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
        assert!(out.contains("distance") && out.contains("A0.center") && out.contains("A1.center"),
            "should show semantic names: {}", out);
    }

    #[test]
    fn test_list_no_pc_distance_mixed_arc_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center L0.p1 10");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    }

    #[test]
    fn test_list_no_pc_distance_point_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    }

    #[test]
    fn test_list_no_pc_symmetry_pp_endpoints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
        run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
        assert!(out.contains("symmetry") && out.contains("L0.p1") && out.contains("L0.p2"),
            "should show semantic names: {}", out);
    }

    #[test]
    fn test_list_no_pc_symmetry_pp_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 5,-5 5,5; add_circle 10,0 3");
        run_ok(&mut ctx, "symmetry A0.center L0 A1.center");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    }

    #[test]
    fn test_list_no_bridge_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
        let out = list_constraints_output(&mut ctx);
        // Should not contain bridge coincident entries
        let lines: Vec<&str> = out.lines().collect();
        for line in &lines {
            if line.starts_with("coincident") {
                assert!(!line.contains("Pc"), "bridge constraint should be hidden: {}", line);
            }
        }
    }

    // 6B: Cleanup on object deletion
    // Note: Line-Line endpoint distances (DistanceLL*) don't create helpers.
    // Helpers are created for Arc endpoint distances and PointLineDistance
    // with non-Point endpoints.

    #[test]
    fn test_cleanup_delete_line_removes_symmetry_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
        run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
        assert!(has_helper_points(&ctx), "should have helpers after symmetry");
        run_ok(&mut ctx, "delete L0");
        assert!(!has_helper_points(&ctx), "helpers should be cleaned up after delete L0");
        assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry_pp should be empty");
    }

    #[test]
    fn test_cleanup_delete_arc_removes_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center A1.center 10");
        assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
        assert_eq!(ctx.sketch.distance_aa_ce_ce.len(), 1);
        run_ok(&mut ctx, "delete A0");
        assert!(ctx.sketch.distance_aa_ce_ce.is_empty(), "constraint should be cleaned up");
    }

    #[test]
    fn test_cleanup_delete_arc_removes_symmetry_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 5,-5 5,5; add_circle 10,0 3");
        run_ok(&mut ctx, "symmetry A0.center L0 A1.center");
        assert!(has_helper_points(&ctx), "should have helpers");
        run_ok(&mut ctx, "delete A0");
        assert!(!has_helper_points(&ctx), "helpers should be cleaned up");
        assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry_pp should be empty");
    }

    #[test]
    fn test_cleanup_delete_mirror_line_removes_symmetry_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
        run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
        assert!(!ctx.sketch.symmetry_pp.is_empty());
        run_ok(&mut ctx, "delete L1");
        assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry gone after mirror line deleted");
        assert!(!has_helper_points(&ctx), "helpers cleaned up");
    }

    #[test]
    fn test_cleanup_delete_line_removes_pl_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
        assert_eq!(ctx.sketch.distance_lp1l.len(), 1);
        run_ok(&mut ctx, "delete L0");
        assert!(ctx.sketch.distance_lp1l.is_empty(), "constraint cleaned up");
    }

    // 6C: Cleanup on dimension removal

    #[test]
    fn test_cleanup_remove_dim_distance_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center A1.center 10");
        assert_eq!(ctx.sketch.distance_aa_ce_ce.len(), 1);
        run_ok(&mut ctx, "delete d0");
        assert!(ctx.sketch.distance_aa_ce_ce.is_empty(), "constraint cleaned up after remove_dim");
    }

    #[test]
    fn test_cleanup_remove_dim_distance_point_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
        assert_eq!(ctx.sketch.distance_lp1l.len(), 1);
        run_ok(&mut ctx, "delete d0");
        assert!(ctx.sketch.distance_lp1l.is_empty(), "constraint cleaned up after remove_dim");
    }

    #[test]
    fn test_cleanup_remove_dim_distance_arc_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 0,5 5,5");
        run_ok(&mut ctx, "distance A0.center L0 5");
        assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
        assert_eq!(ctx.sketch.distance_arc_center_l.len(), 1);
        run_ok(&mut ctx, "delete d0");
        assert!(ctx.sketch.distance_arc_center_l.is_empty(), "constraint cleaned up after remove_dim");
    }

    #[test]
    fn test_distance_pl_line_endpoint() {
        // LineP1 to line (perpendicular distance)
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
        assert_eq!(ctx.sketch.distance_lp1l.len(), 1);

        // LineP2 to line
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p2 L1 3");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_lp2l.len(), 1);
    }

    #[test]
    fn test_distance_pl_arc_center() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_line 0,5 5,5");
        run_ok(&mut ctx, "distance A0.center L0 4");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_center_l.len(), 1);
    }

    #[test]
    fn test_distance_pl_arc_start() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 3,0 0,3; add_line 0,10 5,10");
        run_ok(&mut ctx, "distance A0.start L0 8");
        assert!(!has_helper_points(&ctx));
        assert_eq!(ctx.sketch.distance_arc_start_l.len(), 1);
    }

    #[test]
    fn test_cleanup_remove_dim_distance_arc_mixed() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 10,0 15,0");
        run_ok(&mut ctx, "distance A0.center L0.p1 10");
        assert!(!has_helper_points(&ctx), "direct constraint, no helpers");
        assert_eq!(ctx.sketch.distance_arc_center_l1.len(), 1);
        run_ok(&mut ctx, "delete d0");
        assert!(ctx.sketch.distance_arc_center_l1.is_empty(), "constraint cleaned up after remove_dim");
    }

    // 6D: No Pc in distance constraints that don't need helpers (regression)

    #[test]
    fn test_no_helpers_for_line_line_distance() {
        // Line-Line endpoint distances use specialized constraints, no helpers
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
        assert!(!has_helper_points(&ctx), "DistanceLL should not create helpers");
    }

    // -- Autocomplete tests --

    fn setup_complete_ctx() -> CommandContext {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");   // L0
        run_ok(&mut ctx, "add_line 5,0 5,5");   // L1
        run_ok(&mut ctx, "add_point 2,3");       // P0
        run_ok(&mut ctx, "add_circle 3,3 2");    // A0
        run_ok(&mut ctx, "length L0 5");         // d0
        run_ok(&mut ctx, "param width 10");
        ctx
    }

    fn completions(ctx: &CommandContext, input: &str) -> Vec<String> {
        complete(&ctx.sketch, &ctx.session_names, input, input.len())
    }

    // -- DOF check on constraints --

    #[test]
    fn test_dof_check_accepts_valid() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        // horizontal removes 1 DOF, should succeed
    }

    #[test]
    fn test_dof_check_force_overrides() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        // Two parallel lines, then collinear (removes 1 more DOF beyond parallel)
        run_ok(&mut ctx, "add_line 0,1 5,1");
        run_ok(&mut ctx, "parallel L0 L1");
        run_ok(&mut ctx, "collinear L0 L1");
    }

    // -- DOF analysis --

    #[test]
    fn test_dof_analyze_unconstrained_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "dof analyze");
        assert!(out.contains("DOF: 4"), "Unconstrained line should have 4 DOF: {}", out);
    }

    #[test]
    fn test_dof_analyze_constrained() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; horizontal L0; length L0 5; lock L0.p1 0,0");
        let out = run_ok(&mut ctx, "dof analyze");
        assert!(out.contains("DOF: 0"), "Fully constrained should be DOF 0: {}", out);
    }

    #[test]
    fn test_dof_analyze_partial() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; horizontal L0; length L0 5");
        let out = run_ok(&mut ctx, "dof analyze");
        // 4 DOF - 1 (horizontal) - 1 (length) = 2 DOF (translate X, Y)
        assert!(out.contains("DOF: 2"), "Should have 2 DOF: {}", out);
        assert!(out.contains("translate"), "Should identify translation: {}", out);
    }

    #[test]
    fn test_dof_analyze_empty() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "clear");
        let out = run_ok(&mut ctx, "dof analyze");
        assert!(out.contains("DOF: 0"), "Empty sketch should be 0 DOF: {}", out);
    }

    // -- point_on with arc endpoints --

    #[test]
    fn test_point_on_arc_center_on_line_no_duplicate() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 2,1 1");
        run_ok(&mut ctx, "point_on A0.center L0");
        let arc_center_count = ctx.sketch.coincident_arc_center.len();
        let pol_count = ctx.sketch.point_on_line.len();
        eprintln!("After first: arc_center={}, point_on_line={}", arc_center_count, pol_count);
        let out2 = run_err(&mut ctx, "point_on A0.center L0");
        eprintln!("Second attempt: {}", out2);
        assert!(out2.contains("already exists"), "Should reject duplicate: {}", out2);
    }

    #[test]
    fn test_point_on_arc_center_on_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0.5 2; add_line -5,0 5,0");
        let out = run_ok(&mut ctx, "point_on A0.center L0");
        assert!(out.contains("point-on-line"), "Should succeed: {}", out);
        // Verify helper point was created and point_on_line constraint exists
        assert!(!ctx.sketch.point_on_line.is_empty(),
            "Should have point_on_line constraint");
    }

    #[test]
    fn test_point_on_arc_center_on_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_circle 4.5,0 1");
        let out = run_ok(&mut ctx, "point_on A1.center A0");
        assert!(out.contains("point-on-arc"), "Should succeed: {}", out);
        assert!(!ctx.sketch.point_on_arc.is_empty(),
            "Should have point_on_arc constraint");
    }

    // -- Dimension update (no duplicates) --

    #[test]
    fn test_dimension_update_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 5");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "length L0 10");
        assert!(out.contains("Updated"), "Should update existing: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1, "Should still be 1 dimension");
    }

    #[test]
    fn test_dimension_update_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "radius A0 10");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_update_radius_to_expr() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5");
        run_ok(&mut ctx, "param scale 2");
        let out = run_ok(&mut ctx, "radius A0 \"5*scale\"");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_update_angle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
        run_ok(&mut ctx, "angle L0 L1 45");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "angle L0 L1 90");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_update_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 7,0 10,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "distance L0.p2 L1.p1 5");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_expr_constrains() {
        // Bare expression is live under the current grammar (commit 76947c1).
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "clear");
        run_ok(&mut ctx, "param scale 1");
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5*scale");
        // Check that expr_constraints were created
        ctx.sketch.solve();
        assert!(!ctx.sketch.expr_constraints.is_empty(),
            "Expression dimension should create expr_constraint, got none. dims: {:?}",
            ctx.sketch.dimensions.iter().map(|d| (&d.name, &d.expr_str, d.value)).collect::<Vec<_>>());
        // Verify it actually constrains the radius
        let r = ctx.sketch.arcs.refs().next().unwrap();
        let radius = ctx.sketch.arcs[r].radius.value;
        assert!((radius - 5.0).abs() < 0.1,
            "radius should be 5*1=5, got {}", radius);
    }

    #[test]
    fn test_dimension_expr_constrains_fresh() {
        // Fresh expression dim without prior numeric.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param scale 1");
        run_ok(&mut ctx, "add_circle 0,0 5");
        // Bare form is live (tracks `scale`).
        run_ok(&mut ctx, "radius A0 5*scale");
        // Check dimension was created with expression
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert_eq!(ctx.sketch.dimensions[0].expr_str.as_deref(), Some("5*scale"));
        // Solve and check expr constraints are built
        let result = ctx.sketch.solve();
        assert!(!ctx.sketch.expr_constraints.is_empty(),
            "Should have expr_constraints after solve");
        // Check radius is actually constrained to 5
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!((ctx.sketch.arcs[r].radius.value - 5.0).abs() < 0.1,
            "radius should be 5*1=5, got {}", ctx.sketch.arcs[r].radius.value);
        assert!(result.end_cost < 0.01, "cost should be near zero, got {}", result.end_cost);
    }

    #[test]
    fn test_dimension_expr_update_constrains() {
        // Updating numeric dim to an expression should constrain. The old
        // `{2*scale}` brace form was removed by the grammar flip; use the
        // bare-expression form for live tracking.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5");
        run_ok(&mut ctx, "param scale 3");
        run_ok(&mut ctx, "radius A0 2*scale");
        ctx.sketch.solve();
        assert!(!ctx.sketch.expr_constraints.is_empty(),
            "Updated expression should create expr_constraint");
        let r = ctx.sketch.arcs.refs().next().unwrap();
        let radius = ctx.sketch.arcs[r].radius.value;
        assert!((radius - 6.0).abs() < 0.1,
            "radius should be 2*3=6, got {}", radius);
    }

    #[test]
    fn test_dimension_no_cross_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "length L0 5");
        run_ok(&mut ctx, "length L1 3");
        assert_eq!(ctx.sketch.dimensions.len(), 2, "Different entities should have separate dims");
    }

    // -- Autocomplete tests --

    #[test]
    fn test_complete_empty_input() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "").is_empty());
    }

    #[test]
    fn test_complete_first_token_commands() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_");
        assert!(c.contains(&"add_line".to_string()));
        assert!(c.contains(&"add_point".to_string()));
        assert!(c.contains(&"add_circle".to_string()));
        // Should NOT contain entity names
        assert!(!c.iter().any(|s| s.starts_with('L')));
    }

    #[test]
    fn test_complete_list_filters_not_entities() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "list l");
        assert!(c.contains(&"lines".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "list should not offer entity names: {:?}", c);
    }

    #[test]
    fn test_complete_cursor_keywords() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "cursor o");
        assert!(c.contains(&"on".to_string()));
        assert!(c.contains(&"off".to_string()));
    }

    #[test]
    fn test_complete_add_line_cursor() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line curs");
        assert!(c.contains(&"cursor".to_string()));
    }

    #[test]
    fn test_complete_horizontal_lines_only() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "horizontal L");
        assert!(c.contains(&"L0".to_string()));
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.iter().any(|s| is_arc_name(s)), "horizontal should not offer arcs");
        assert!(!c.iter().any(|s| s.starts_with('P')), "horizontal should not offer points");
    }

    #[test]
    fn test_complete_concentric_arcs_only() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "concentric A");
        assert!(c.contains(&"A0".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "concentric should not offer lines");
    }

    #[test]
    fn test_complete_style_values() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "style L0 d");
        assert!(c.contains(&"dashed".to_string()));
        assert!(c.contains(&"dashdot".to_string()));
        assert!(!c.iter().any(|s| s.starts_with("d0")), "style arg2 should not offer dimensions");
    }

    #[test]
    fn test_complete_delete_dim() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "delete d");
        assert!(c.contains(&"d0".to_string()));
    }

    #[test]
    fn test_complete_length_arg2_derived() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "length L0 d");
        assert!(c.contains(&"derived".to_string()));
        // Should offer dimension refs in expression context
        assert!(c.contains(&"d0".to_string()));
        // Should NOT offer lines
        assert!(!c.contains(&"L0".to_string()), "length arg2 should not offer L0");
    }

    #[test]
    fn test_complete_equal_type_matching() {
        let ctx = setup_complete_ctx();
        // After "equal L0", should only offer lines
        let c = completions(&ctx, "equal L0 L");
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.iter().any(|s| is_arc_name(s)), "equal with L0 should not offer arcs");
    }

    #[test]
    fn test_complete_dim_pos() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "dim_pos d0 o");
        assert!(c.contains(&"offset".to_string()));
        let c = completions(&ctx, "dim_pos d0 a");
        assert!(c.contains(&"along".to_string()));
    }

    #[test]
    fn test_complete_no_arg_commands() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "dof x").is_empty());
        assert!(completions(&ctx, "cost x").is_empty());
    }

    #[test]
    fn test_complete_del_param() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "del_param w");
        assert!(c.contains(&"width".to_string()));
    }

    #[test]
    fn test_complete_delete_constraint_types() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "delete L0 h");
        assert!(c.contains(&"horizontal".to_string()));
    }

    #[test]
    fn test_complete_dot_line() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "info L0.");
        assert!(c.contains(&"L0.p1".to_string()));
        assert!(c.contains(&"L0.p2".to_string()));
    }

    #[test]
    fn test_complete_dot_arc() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "info A0.");
        assert!(c.contains(&"A0.center".to_string()));
        assert!(c.contains(&"A0.start".to_string()));
        assert!(c.contains(&"A0.end".to_string()));
    }

    #[test]
    fn test_complete_midpoint_arg2_lines_only() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "midpoint P0 L");
        assert!(c.contains(&"L0".to_string()));
        assert!(!c.iter().any(|s| is_arc_name(s)), "midpoint arg2 should not offer arcs");
    }

    #[test]
    fn test_complete_offset_line_arg1() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "offset L");
        assert!(c.contains(&"L0".to_string()));
        assert!(!c.iter().any(|s| is_arc_name(s)));
    }

    #[test]
    fn test_complete_list_space_shows_options() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "list ");
        assert!(c.contains(&"lines".to_string()));
        assert!(c.contains(&"constraints".to_string()));
    }

    #[test]
    fn test_complete_horizontal_space_shows_lines() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "horizontal ");
        assert!(c.contains(&"L0".to_string()));
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.iter().any(|s| is_arc_name(s)));
    }

    #[test]
    fn test_complete_cursor_space_shows_keywords() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "cursor ");
        assert!(c.contains(&"on".to_string()));
        assert!(c.contains(&"off".to_string()));
    }

    #[test]
    fn test_complete_style_space_shows_entities() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "style ");
        assert!(c.contains(&"L0".to_string()));
    }

    #[test]
    fn test_complete_style_entity_space_shows_values() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "style L0 ");
        assert!(c.contains(&"solid".to_string()));
        assert!(c.contains(&"dashed".to_string()));
    }

    #[test]
    fn test_complete_empty_first_token_no_suggestions() {
        let ctx = setup_complete_ctx();
        // Just a space or empty — no suggestions
        assert!(completions(&ctx, "").is_empty());
    }

    #[test]
    fn test_complete_add_line_after_coords_only_flags() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line 0,0 5,0 ");
        assert!(c.contains(&"noconnect".to_string()));
        assert!(c.contains(&"nocursor".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "Should not offer entities after coords: {:?}", c);
    }

    #[test]
    fn test_complete_add_line_flag_excludes_typed() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line 0,0 5,0 nocursor ");
        assert!(c.contains(&"noconnect".to_string()));
        assert!(!c.contains(&"nocursor".to_string()), "Should not re-offer nocursor");
    }

    #[test]
    fn test_complete_add_line_first_coord() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line curs");
        assert!(c.contains(&"cursor".to_string()));
    }

    #[test]
    fn test_complete_add_point_after_coord() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_point 0,0 ");
        assert!(c.contains(&"nocursor".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "Should not offer entities: {:?}", c);
    }

    #[test]
    fn test_complete_add_circle_radius_position() {
        let ctx = setup_complete_ctx();
        // After center coord, radius is expression context
        let c = completions(&ctx, "add_circle 0,0 w");
        assert!(c.contains(&"width".to_string()));
        assert!(!c.contains(&"cursor".to_string()));
    }

    #[test]
    fn test_complete_add_arc_after_3_coords() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_arc 0,0 5,0 2,3 ");
        assert!(c.contains(&"noconnect".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')));
    }

    #[test]
    fn test_complete_help_full() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "help f");
        assert!(c.contains(&"full".to_string()));
    }

    #[test]
    fn test_complete_list_all_keyword() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "list a");
        assert!(c.contains(&"all".to_string()));
        assert!(c.contains(&"arcs".to_string()));
    }

    #[test]
    fn test_complete_list_no_second_arg() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "list lines ").is_empty());
    }

    #[test]
    fn test_complete_horizontal_excludes_typed() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "horizontal L0 L");
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.contains(&"L0".to_string()), "Should exclude already-typed L0");
    }

    #[test]
    fn test_complete_select_excludes_typed() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "select L0 P0 L");
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.contains(&"L0".to_string()), "Should exclude already-typed L0");
    }

    #[test]
    fn test_complete_cursor_no_second_arg() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "cursor on ").is_empty());
    }

    #[test]
    fn test_complete_parallel_no_third_arg() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "parallel L0 L1 ").is_empty());
    }

    // -- sweep tests --

    #[test]
    fn test_sweep_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        let out = run_ok(&mut ctx, "sweep A0 180");
        assert!(out.contains("Set") || out.contains("sweep"), "Should succeed: {}", out);
        assert!(ctx.sketch.arcs.refs().next().map(|r| ctx.sketch.arcs[r].constraints.has_target_sweep).unwrap_or(false));
        // Solve and check sweep is close to 180 degrees
        ctx.sketch.solve();
        let r = ctx.sketch.arcs.refs().next().unwrap();
        let sweep = (ctx.sketch.arcs[r].end_angle.value - ctx.sketch.arcs[r].start_angle.value).abs().to_degrees();
        assert!((sweep - 180.0).abs() < 1.0, "Sweep should be ~180, got {}", sweep);
    }

    #[test]
    fn test_sweep_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        run_ok(&mut ctx, "sweep A0 180");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "sweep A0 90");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_sweep_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        let out = run_ok(&mut ctx, "sweep A0 derived");
        assert!(out.contains("Derived"), "Should be derived: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
    }

    #[test]
    fn test_sweep_expression() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        run_ok(&mut ctx, "param n 2");
        let out = run_ok(&mut ctx, "sweep A0 \"90*n\"");
        assert!(out.contains("Set") || out.contains("sweep"), "Should succeed: {}", out);
    }

    #[test]
    fn test_sweep_full_circle_rejected() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        let e = run_err(&mut ctx, "sweep A0 180");
        assert!(e.contains("full circle"), "Should reject: {}", e);
    }

    #[test]
    fn test_sweep_remove() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        run_ok(&mut ctx, "sweep A0 180");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let name = ctx.sketch.dimensions[0].name.clone();
        run_ok(&mut ctx, &format!("delete {}", name));
        assert_eq!(ctx.sketch.dimensions.len(), 0);
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!(!ctx.sketch.arcs[r].constraints.has_target_sweep);
    }

    // -- arc derived properties in expressions --

    #[test]
    fn test_print_arc_start_end() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        let out = run_ok(&mut ctx, "print A0.start.x");
        // Should return a number, not an error
        assert!(out.parse::<f64>().is_ok() || out.trim().parse::<f64>().is_ok(),
            "A0.start.x should be a number: {}", out);
        run_ok(&mut ctx, "print A0.start.y");
        run_ok(&mut ctx, "print A0.end.x");
        run_ok(&mut ctx, "print A0.end.y");
        run_ok(&mut ctx, "print A0.sweep");
        run_ok(&mut ctx, "print A0.diameter");
    }

    #[test]
    fn test_geo_functions_in_expressions() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
        // angle() as standalone works
        let out = run_ok(&mut ctx, "print angle(L0,L1)");
        assert!(out.trim().parse::<f64>().is_ok(), "angle(L0,L1) should be numeric: {}", out);
        // angle() inside an expression
        let out = run_ok(&mut ctx, "print angle(L0,L1)+1");
        let val: f64 = out.trim().parse().expect(&format!("should parse: {}", out));
        assert!((val - 91.0).abs() < 1.0, "angle(L0,L1)+1 should be ~91, got {}", val);
        // dist() inside an expression
        let out = run_ok(&mut ctx, "print dist(L0.p1,L0.p2)*2");
        let val: f64 = out.trim().parse().expect(&format!("should parse: {}", out));
        assert!((val - 10.0).abs() < 0.1, "dist*2 should be ~10, got {}", val);
    }

    #[test]
    fn test_inline_comments() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0  # a horizontal line");
        assert_eq!(ctx.sketch.lines.len(), 1);
        run_ok(&mut ctx, "horizontal L0 # make it horizontal");
        assert!(ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].constraints.horizontal);
        // Comment-only line
        let out = run_ok(&mut ctx, "# just a comment");
        assert!(out.is_empty());
        // Quoted strings should not be affected
        run_ok(&mut ctx, "param scale 1");
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 =5*scale # expression dimension");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_variable_assignment() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "s0 = add_line 0,0 5,0; s1 = add_line 5,0 3,4");
        run_ok(&mut ctx, "len = length s0 5");
        assert!(ctx.session_names.contains_key("len"), "len should be set");
        assert_eq!(ctx.session_names["len"], "d0");
        run_ok(&mut ctx, "a = angle s0 s1 60");
        assert!(ctx.session_names.contains_key("a"), "a should be set");
        assert_eq!(ctx.session_names["a"], "d1");
        // Use dimension variable as expression in another dimension
        let out = run_ok(&mut ctx, "print a");
        assert!(out.trim().parse::<f64>().is_ok(), "should resolve: {}", out);
    }

    // -- delete (relational / named constraint) tests --

    #[test]
    fn test_remove_constraint_coincident_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,0; add_point 1,0");
        run_ok(&mut ctx, "coincident P0 P1");
        assert_eq!(ctx.sketch.coincident_pp.len(), 1);
        run_ok(&mut ctx, "delete P0 P1 coincident");
        assert_eq!(ctx.sketch.coincident_pp.len(), 0);
    }

    #[test]
    fn test_remove_constraint_coincident_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
        run_ok(&mut ctx, "coincident L0.p2 L1.p1");
        assert_eq!(ctx.sketch.coincident_ll21.len(), 1);
        run_ok(&mut ctx, "delete L0.p2 L1.p1 coincident");
        assert_eq!(ctx.sketch.coincident_ll21.len(), 0);
    }

    #[test]
    fn test_remove_constraint_coincident_not_found() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
        let e = run_err(&mut ctx, "delete L0.p2 L1.p1 coincident");
        assert!(e.contains("not found"), "{}", e);
    }

    #[test]
    fn test_remove_constraint_point_on_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2,0.5; add_line 0,0 5,0");
        run_ok(&mut ctx, "point_on P0 L0");
        assert_eq!(ctx.sketch.point_on_line.len(), 1);
        run_ok(&mut ctx, "delete P0 L0 point_on");
        assert_eq!(ctx.sketch.point_on_line.len(), 0);
    }

    #[test]
    fn test_remove_constraint_point_on_line_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "point_on L0.p1 L1");
        assert_eq!(ctx.sketch.line_p1_on_line.len(), 1);
        run_ok(&mut ctx, "delete L0.p1 L1 point_on");
        assert_eq!(ctx.sketch.line_p1_on_line.len(), 0);
    }

    #[test]
    fn test_remove_constraint_point_on_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 5,0; add_circle 0,0 5");
        run_ok(&mut ctx, "point_on P0 A0");
        assert_eq!(ctx.sketch.point_on_arc.len(), 1);
        run_ok(&mut ctx, "delete P0 A0 point_on");
        assert_eq!(ctx.sketch.point_on_arc.len(), 0);
    }

    #[test]
    fn test_remove_constraint_point_on_line_arc_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0.5 2; add_line -5,0 5,0");
        run_ok(&mut ctx, "point_on A0.center L0");
        assert!(!ctx.sketch.point_on_line.is_empty());
        run_ok(&mut ctx, "delete A0.center L0 point_on");
        // The point_on_line constraint on the helper should be removed
        // cleanup_helper_points removes orphan helpers
        assert!(ctx.sketch.point_on_line.is_empty() || ctx.sketch.points.refs().all(|p| !ctx.sketch.points[p].helper));
    }

    #[test]
    fn test_remove_constraint_symmetry_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point -3,0; add_line 0,-5 0,5; add_point 3,0");
        run_ok(&mut ctx, "symmetry P0 L0 P1");
        assert_eq!(ctx.sketch.symmetry_pp.len(), 1);
        run_ok(&mut ctx, "delete P0 L0 P1 symmetry");
        assert_eq!(ctx.sketch.symmetry_pp.len(), 0);
    }

    #[test]
    fn test_remove_constraint_symmetry_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line -2,0 -2,3; add_line 0,0 0,5; add_line 2,0 2,3");
        run_ok(&mut ctx, "symmetry L0 L1 L2");
        assert_eq!(ctx.sketch.symmetry_ll.len(), 1);
        run_ok(&mut ctx, "delete L0 L1 L2 symmetry");
        assert_eq!(ctx.sketch.symmetry_ll.len(), 0);
    }

    #[test]
    fn test_remove_constraint_midpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2.5,0.5; add_line 0,0 5,0");
        run_ok(&mut ctx, "midpoint P0 L0");
        assert_eq!(ctx.sketch.midpoint.len(), 1);
        run_ok(&mut ctx, "delete P0 L0 midpoint");
        assert_eq!(ctx.sketch.midpoint.len(), 0);
    }

    #[test]
    fn test_remove_constraint_midpoint_lp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line -5,0 10,0");
        run_ok(&mut ctx, "midpoint L0.p1 L1");
        assert_eq!(ctx.sketch.midpoint_lp1.len(), 1);
        run_ok(&mut ctx, "delete L0.p1 L1 midpoint");
        assert_eq!(ctx.sketch.midpoint_lp1.len(), 0);
    }

    #[test]
    fn test_remove_constraint_equal_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_circle 10,0 3");
        run_ok(&mut ctx, "equal A0 A1");
        assert_eq!(ctx.sketch.equal_radius.len(), 1);
        run_ok(&mut ctx, "delete A0 A1 equal_radius");
        assert_eq!(ctx.sketch.equal_radius.len(), 0);
    }

    #[test]
    fn test_remove_constraint_equal_radius_not_found() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_circle 10,0 3");
        let e = run_err(&mut ctx, "delete A0 A1 equal_radius");
        assert!(e.contains("not found"), "{}", e);
    }

    #[test]
    fn test_remove_constraint_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        run_ok(&mut ctx, "delete L0 horizontal");
        assert!(!ctx.sketch.lines.iter().next().unwrap().constraints.horizontal);
    }

    #[test]
    fn test_remove_constraint_vertical() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,5");
        run_ok(&mut ctx, "vertical L0");
        run_ok(&mut ctx, "delete L0 vertical");
        assert!(!ctx.sketch.lines.iter().next().unwrap().constraints.vertical);
    }

    #[test]
    fn test_remove_constraint_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "parallel L0 L1");
        assert_eq!(ctx.sketch.parallel.len(), 1);
        run_ok(&mut ctx, "delete L0 L1 parallel");
        assert_eq!(ctx.sketch.parallel.len(), 0);
    }

    #[test]
    fn test_remove_constraint_perpendicular() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
        run_ok(&mut ctx, "perpendicular L0 L1");
        assert_eq!(ctx.sketch.perpendicular.len(), 1);
        run_ok(&mut ctx, "delete L0 L1 perpendicular");
        assert_eq!(ctx.sketch.perpendicular.len(), 0);
    }

    #[test]
    fn test_remove_constraint_equal_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "equal L0 L1");
        assert_eq!(ctx.sketch.equal_length.len(), 1);
        run_ok(&mut ctx, "delete L0 L1 equal");
        assert_eq!(ctx.sketch.equal_length.len(), 0);
    }

    #[test]
    fn test_remove_constraint_collinear() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 6,0 10,0");
        run_ok(&mut ctx, "collinear L0 L1");
        assert_eq!(ctx.sketch.collinear.len(), 1);
        run_ok(&mut ctx, "delete L0 L1 collinear");
        assert_eq!(ctx.sketch.collinear.len(), 0);
    }

    #[test]
    fn test_remove_constraint_tangent_la() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,4 5,4; add_circle 2,0 4");
        run_ok(&mut ctx, "tangent L0 A0");
        assert_eq!(ctx.sketch.tangent_la.len(), 1);
        run_ok(&mut ctx, "delete L0 A0 tangent");
        assert_eq!(ctx.sketch.tangent_la.len(), 0);
    }

    #[test]
    fn test_remove_constraint_tangent_aa() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 7,0 4");
        run_ok(&mut ctx, "tangent A0 A1");
        assert_eq!(ctx.sketch.tangent_aa.len(), 1);
        run_ok(&mut ctx, "delete A0 A1 tangent");
        assert_eq!(ctx.sketch.tangent_aa.len(), 0);
    }

    #[test]
    fn test_remove_constraint_concentric() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 1,0 5");
        run_ok(&mut ctx, "concentric A0 A1");
        assert_eq!(ctx.sketch.concentric.len(), 1);
        run_ok(&mut ctx, "delete A0 A1 concentric");
        assert_eq!(ctx.sketch.concentric.len(), 0);
    }

    /// Regression: the circle-to-circle distance dimension must be
    /// placeable on two geometrically-concentric circles without a
    /// pre-existing `Concentric` constraint. The command auto-installs
    /// a Concentric for list-visibility.
    #[test]
    fn test_distance_concentric_no_prior_concentric() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2 noconnect");
        run_ok(&mut ctx, "add_circle 0,0 5 noconnect");
        assert_eq!(ctx.sketch.concentric.len(), 0);
        run_ok(&mut ctx, "distance A0 A1 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert_eq!(ctx.sketch.distance_concentric.len(), 1);
        // A paired Concentric was auto-installed for visibility.
        assert_eq!(ctx.sketch.concentric.len(), 1);
        assert!((ctx.sketch.arcs[ctx.sketch.arcs.refs().nth(1).unwrap()].radius.value
               - ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].radius.value
               - 3.0).abs() < 0.01);
    }

    /// Regression: the dim's self-contained residual keeps the circles
    /// concentric even after the user manually deletes the paired
    /// `Concentric` constraint. Previously the cascade removed the dim
    /// alongside the Concentric; now the dim survives.
    #[test]
    fn test_concentric_distance_survives_concentric_delete() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2 noconnect");
        run_ok(&mut ctx, "add_circle 0,0 5 noconnect");
        run_ok(&mut ctx, "distance A0 A1 3");
        assert_eq!(ctx.sketch.concentric.len(), 1);
        assert_eq!(ctx.sketch.distance_concentric.len(), 1);
        // Delete the paired Concentric by name. The dim (and its
        // backing constraint) must stay.
        let nid = ctx.sketch.concentric[0].nid;
        run_ok(&mut ctx, &format!("delete C{}", nid));
        assert_eq!(ctx.sketch.concentric.len(), 0);
        assert_eq!(ctx.sketch.dimensions.len(), 1,
            "dim must survive manual Concentric deletion");
        assert_eq!(ctx.sketch.distance_concentric.len(), 1,
            "backing DistanceConcentric must survive");
        // Solve must still hold the circles concentric (the dim's own
        // residual enforces it).
        ctx.sketch.solve();
        let ca = ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].center.value;
        let cb = ctx.sketch.arcs[ctx.sketch.arcs.refs().nth(1).unwrap()].center.value;
        assert!((ca.x - cb.x).abs() < 0.01 && (ca.y - cb.y).abs() < 0.01,
            "circles must stay concentric: {:?} vs {:?}", ca, cb);
    }

    /// The dim only accepts circles whose centers currently coincide.
    /// Non-concentric pairs fall through to PointPointDistance or
    /// fail, per the existing grammar.
    #[test]
    fn test_distance_concentric_rejects_non_concentric_circles() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2 noconnect");
        run_ok(&mut ctx, "add_circle 5,0 3 noconnect");
        // Centers are 5 units apart -> not eligible for
        // ConcentricDistance. `distance A0 A1 3` falls through to the
        // endpoint-pair path, which fails to resolve bare arc names as
        // endpoints.
        let out = run_err(&mut ctx, "distance A0 A1 3");
        assert!(out.contains("Cannot parse endpoint"), "{}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 0);
        assert_eq!(ctx.sketch.distance_concentric.len(), 0);
    }

    #[test]
    fn test_remove_constraint_undo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        let dof_with = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "delete L0 horizontal");
        let dof_without = ctx.sketch.dof().unwrap();
        assert!(dof_without > dof_with, "DOF should increase after removing constraint: {} vs {}", dof_without, dof_with);
        run_ok(&mut ctx, "undo");
        let dof_undone = ctx.sketch.dof().unwrap();
        assert_eq!(dof_undone, dof_with, "DOF should restore after undo: {} vs {}", dof_undone, dof_with);
    }

    #[test]
    fn test_remove_constraint_dof_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "parallel L0 L1");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "delete L0 L1 parallel");
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before + 1, "removing parallel should increase DOF by 1: {} -> {}", dof_before, dof_after);
    }

    // -- Multi-segment add_line --

    #[test]
    fn test_add_line_multi_segment_3_points() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 2,1");
        assert_eq!(ctx.sketch.lines.len(), 2);
        // Auto-coincident between L0.p2 and L1.p1
        assert!(!ctx.sketch.coincident_ll21.is_empty() || !ctx.sketch.coincident_ll12.is_empty(),
            "segments should be connected");
    }

    #[test]
    fn test_add_line_multi_segment_5_points() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0 2,1 3,0 4,1");
        assert_eq!(ctx.sketch.lines.len(), 4);
    }

    #[test]
    fn test_add_line_multi_segment_relative() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 @1,0 @0,1");
        assert_eq!(ctx.sketch.lines.len(), 2);
        let l1 = ctx.sketch.lines.iter().nth(1).unwrap().p2.value;
        assert!((l1.x - 1.0).abs() < 0.01 && (l1.y - 1.0).abs() < 0.01,
            "L1.p2 should be (1,1), got ({},{})", l1.x, l1.y);
    }

    #[test]
    fn test_add_line_multi_assignment() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "a, b, c = add_line 0,0 @1,0 @0,1 @-1,0");
        assert_eq!(ctx.sketch.lines.len(), 3);
        assert!(ctx.session_names.contains_key("a"));
        assert!(ctx.session_names.contains_key("b"));
        assert!(ctx.session_names.contains_key("c"));
        // Use alias in constraint
        run_ok(&mut ctx, "horizontal a");
        assert!(ctx.sketch.lines.iter().next().unwrap().constraints.horizontal);
    }

    #[test]
    fn test_add_line_two_points_compat() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.sketch.lines.len(), 1);
    }

    #[test]
    fn test_add_line_multi_segment_dof() {
        let mut ctx = CommandContext::new();
        // 4 points = 3 lines, 2 coincidents
        // 3*4 params - 2*2 coincident = 8 DOF
        run_ok(&mut ctx, "add_line 0,0 1,0 2,1 3,0");
        let dof = ctx.sketch.dof().unwrap();
        assert_eq!(dof, 8, "3 connected lines should have 8 DOF, got {}", dof);
    }

    // -- Angle direct/supplement --

    #[test]
    fn test_angle_default_is_direction_vectors() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        // Default: constrains the angle between p1->p2 direction vectors
        run_ok(&mut ctx, "angle L0 L1 45");
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(!supplement, "default should not be supplement");
        } else {
            panic!("expected angle dimension");
        }
    }

    #[test]
    fn test_angle_supplement_keyword() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        run_ok(&mut ctx, "angle L0 L1 135 supplement");
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(supplement, "should be supplement sector");
        } else {
            panic!("expected angle dimension");
        }
    }

    #[test]
    fn test_angle_closest_keyword() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        // Current angle is ~45. Value 130 is closer to supplement (135) than direct (45).
        run_ok(&mut ctx, "angle L0 L1 130 closest");
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(supplement, "closest should pick supplement for 130 when direct is ~45");
        } else {
            panic!("expected angle dimension");
        }
    }

    #[test]
    fn test_angle_acute_keyword() {
        let mut ctx = CommandContext::new();
        // Lines at ~120 degrees (direct angle > 90, so acute picks supplement)
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 -2,4");
        run_ok(&mut ctx, "angle L0 L1 60 acute");
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(supplement, "acute should pick the smaller sector");
        } else {
            panic!("expected angle dimension");
        }
    }

    #[test]
    fn test_angle_obtuse_keyword() {
        let mut ctx = CommandContext::new();
        // Lines at ~45 degrees (direct is 45, supplement is 135)
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        run_ok(&mut ctx, "angle L0 L1 135 obtuse");
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            // Direct is ~45 (acute), so obtuse picks supplement (135)
            assert!(supplement, "obtuse should pick the larger sector");
        } else {
            panic!("expected angle dimension");
        }
    }

    #[test]
    fn test_angle_negative_value_accepted() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        // Negative value should be accepted (taken as absolute value)
        run_ok(&mut ctx, "angle L0 L1 -45");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_angle_driven_closest() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        let dof_before = ctx.sketch.dof().unwrap();
        // "driven closest" — driven is before sector keyword
        run_ok(&mut ctx, "angle L0 L1 driven closest");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(ctx.sketch.dof().unwrap() < dof_before);
    }

    #[test]
    fn test_angle_driven_supplement() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "angle L0 L1 driven supplement");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(supplement, "should be supplement sector");
        } else {
            panic!("expected angle dimension");
        }
        assert!(ctx.sketch.dof().unwrap() < dof_before);
    }

    #[test]
    fn test_angle_driven_acute() {
        let mut ctx = CommandContext::new();
        // Lines at ~120 degrees so acute picks supplement
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 -2,4");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "angle L0 L1 driven acute");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(supplement, "acute should pick the smaller sector");
        } else {
            panic!("expected angle dimension");
        }
        assert!(ctx.sketch.dof().unwrap() < dof_before);
    }

    #[test]
    fn test_angle_driven_obtuse() {
        let mut ctx = CommandContext::new();
        // Lines at ~45 degrees, so obtuse picks supplement (135)
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "angle L0 L1 driven obtuse");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        if let arael_sketch_solver::DimensionKind::Angle(_, _, supplement) = ctx.sketch.dimensions[0].kind {
            assert!(supplement, "obtuse should pick the larger sector");
        } else {
            panic!("expected angle dimension");
        }
        assert!(ctx.sketch.dof().unwrap() < dof_before);
    }

    #[test]
    fn test_angle_closest_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        let dof_before = ctx.sketch.dof().unwrap();
        // Reverse order: sector keyword before driven
        run_ok(&mut ctx, "angle L0 L1 closest driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(ctx.sketch.dof().unwrap() < dof_before);
    }

    #[test]
    fn test_angle_value_closest_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        // With explicit value + sector + driven
        run_ok(&mut ctx, "angle L0 L1 45 closest driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
    }

    #[test]
    fn test_angle_value_driven_closest() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        // Reverse order with explicit value
        run_ok(&mut ctx, "angle L0 L1 45 driven closest");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
    }

    // -- chamfer --

    #[test]
    fn test_chamfer_two_lines() {
        let mut ctx = CommandContext::new();
        // Right-angle corner at (5,0).
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 5,3");
        let out = run_ok(&mut ctx, "chamfer L0 L1 0.5");
        assert!(out.contains("L0 L1 -> "), "got: {}", out);
        // A new line (the bevel) and a new point (corner anchor).
        assert_eq!(ctx.sketch.lines.refs().count(), 3);
        assert_eq!(ctx.sketch.points.refs().filter(|r| !ctx.sketch.points[*r].helper).count(), 1);
        // Two distance dims.
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        // Secondary dim tracks the primary by name.
        let primary_name = ctx.sketch.dimensions[0].name.clone();
        assert_eq!(ctx.sketch.dimensions[1].expr_str.as_deref(), Some(primary_name.as_str()));
        // L0 trimmed to x=4.5, L1 trimmed to y=0.5.
        let l0 = ctx.sketch.lines.iter().next().unwrap();
        assert!((l0.p2.value.x - 4.5).abs() < 1e-6);
        let l1 = ctx.sketch.lines.iter().nth(1).unwrap();
        assert!((l1.p1.value.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chamfer_endpoint_form() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 5,3");
        let out = run_ok(&mut ctx, "chamfer L0.p2 0.5");
        assert!(out.contains("L0.p2 -> "), "got: {}", out);
    }

    #[test]
    fn test_chamfer_parametric_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0");
        run_ok(&mut ctx, "add_line 10,0 10,10");
        run_ok(&mut ctx, "length L0 10");
        // Live expression on the chamfer distance -- must stick as
        // expr_str on the primary distance dim so the whole chamfer
        // tracks when d0 moves.
        run_ok(&mut ctx, "chamfer L0 L1 d0*0.1");
        // Dim layout: d0 (length), d1 (chamfer primary), d2 (secondary).
        let d1 = &ctx.sketch.dimensions[1];
        assert_eq!(d1.expr_str.as_deref(), Some("d0*0.1"));
        let d2 = &ctx.sketch.dimensions[2];
        assert_eq!(d2.expr_str.as_deref(), Some(d1.name.as_str()));
        run_ok(&mut ctx, "length L0 20");
        let d1 = &ctx.sketch.dimensions[1];
        assert!((d1.value - 2.0).abs() < 1e-3,
            "chamfer primary should track to 2.0, got {}", d1.value);
    }

    #[test]
    fn test_chamfer_rejects_collinear() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 10,0");
        let r = execute_one(&mut ctx, "chamfer L0 L1 0.5");
        assert!(r.is_error);
        assert!(r.output.contains("collinear"), "got: {}", r.output);
    }

    #[test]
    fn test_chamfer_rejects_too_short() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0");
        run_ok(&mut ctx, "add_line 1,0 1,1");
        let r = execute_one(&mut ctx, "chamfer L0 L1 2");
        assert!(r.is_error);
        assert!(r.output.contains("too short"), "got: {}", r.output);
    }

    // -- fillet --

    #[test]
    fn test_fillet_two_lines() {
        let mut ctx = CommandContext::new();
        // Right-angle corner at (5,0).
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 5,3");
        let out = run_ok(&mut ctx, "fillet L0 L1 0.5");
        assert!(out.contains("L0 L1 -> A0"), "got: {}", out);
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
        assert_eq!(ctx.sketch.tangent_la.len(), 2);
        // Radius dimension added.
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        // Lines trimmed: L0.p2 was at x=5, now at x=4.5.
        let l0 = ctx.sketch.lines.iter().next().unwrap();
        assert!((l0.p2.value.x - 4.5).abs() < 1e-6,
            "L0.p2.x should be 4.5, got {}", l0.p2.value.x);
        let l1 = ctx.sketch.lines.iter().nth(1).unwrap();
        assert!((l1.p1.value.y - 0.5).abs() < 1e-6,
            "L1.p1.y should be 0.5, got {}", l1.p1.value.y);
    }

    #[test]
    fn test_fillet_endpoint_form() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 5,3");
        let out = run_ok(&mut ctx, "fillet L0.p2 0.5");
        assert!(out.contains("L0.p2 -> "), "got: {}", out);
    }

    #[test]
    fn test_fillet_variadic_all_rect_corners() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect 0,0 5,3 hv");
        // All four corners in one command, last token is radius.
        // Every non-primary fillet references the primary dim name
        // so the four radii track one source.
        let out = run_ok(&mut ctx, "fillet L0 L1 L1 L2 L2 L3 L3 L0 0.5");
        assert!(out.contains("Filleted 4 of 4"), "got: {}", out);
        assert_eq!(ctx.sketch.arcs.refs().count(), 4);
        // Exactly one radius dim that's a literal; three that
        // reference the primary by name.
        let dims: Vec<_> = ctx.sketch.dimensions.iter().collect();
        assert_eq!(dims.len(), 4);
        let primary = &dims[0];
        assert!(primary.expr_str.is_none());
        for d in &dims[1..] {
            assert_eq!(d.expr_str.as_deref(), Some(primary.name.as_str()));
        }
    }

    #[test]
    fn test_fillet_variadic_partial_failure() {
        // First corner valid, second corner too short -- whole command
        // still succeeds with one fillet reported and the other as
        // FAILED in the per-corner report.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 5,3");
        // L2 is only 0.1 long -- fillet radius 0.5 is too short.
        run_ok(&mut ctx, "add_line 5,3 5.1,3");
        let out = run_ok(&mut ctx, "fillet L0 L1 L1 L2 0.5");
        assert!(out.contains("Filleted 1 of 2"), "got: {}", out);
        assert!(out.contains("FAILED"), "got: {}", out);
    }

    #[test]
    fn test_fillet_notangent_noradius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 5,3");
        run_ok(&mut ctx, "fillet L0 L1 0.5 notangent noradius");
        assert_eq!(ctx.sketch.tangent_la.len(), 0);
        assert_eq!(ctx.sketch.dimensions.len(), 0);
        // Arc still exists.
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    }

    #[test]
    fn test_fillet_parametric_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0");
        run_ok(&mut ctx, "add_line 10,0 10,10");
        run_ok(&mut ctx, "length L0 10");
        // Radius expressed as a live expression referencing the length
        // dim: the fillet dim must store the expression so later edits
        // to d0 propagate through.
        run_ok(&mut ctx, "fillet L0 L1 d0*0.1");
        // Radius dim is the second dim (d1). Verify it captured the
        // expression, not the numeric snapshot.
        let r_dim = &ctx.sketch.dimensions[1];
        assert_eq!(r_dim.expr_str.as_deref(), Some("d0*0.1"),
            "expected live expr, got {:?}", r_dim.expr_str);
        // Change the source; fillet radius must follow.
        run_ok(&mut ctx, "length L0 20");
        let r_dim = &ctx.sketch.dimensions[1];
        assert!((r_dim.value - 2.0).abs() < 1e-3,
            "radius should track to 2.0 after length=20, got {}", r_dim.value);
    }

    #[test]
    fn test_fillet_rejects_collinear() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,0 10,0");
        let r = execute_one(&mut ctx, "fillet L0 L1 0.5");
        assert!(r.is_error, "collinear fillet must fail");
        assert!(r.output.contains("collinear"), "got: {}", r.output);
    }

    #[test]
    fn test_fillet_rejects_too_short() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0");
        run_ok(&mut ctx, "add_line 1,0 1,1");
        let r = execute_one(&mut ctx, "fillet L0 L1 2");
        assert!(r.is_error);
        assert!(r.output.contains("too short"), "got: {}", r.output);
    }

    #[test]
    fn test_fillet_rejects_disconnected() {
        let mut ctx = CommandContext::new();
        // Not touching.
        run_ok(&mut ctx, "add_line 0,0 1,0");
        run_ok(&mut ctx, "add_line 2,0 2,1");
        let r = execute_one(&mut ctx, "fillet L0 L1 0.2");
        assert!(r.is_error);
        assert!(r.output.contains("not connected"), "got: {}", r.output);
    }

    // -- add_rect / add_rect3 / add_rectcenter --

    #[test]
    fn test_add_rect_basic() {
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "add_rect 0,0 5,3");
        assert_eq!(ctx.sketch.lines.refs().count(), 4);
        assert_eq!(ctx.sketch.perpendicular.len(), 1);
        assert_eq!(ctx.sketch.parallel.len(), 2);
        assert!(out.contains("perpendicular"), "should list perpendicular: {}", out);
        assert!(out.contains("parallel"), "should list parallel: {}", out);
    }

    #[test]
    fn test_add_rect_hv() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect 0,0 5,3 hv");
        assert_eq!(ctx.sketch.lines.refs().count(), 4);
        // hv: horizontal on 2 lines, vertical on 2 lines
        let h_count = ctx.sketch.lines.refs().filter(|r| ctx.sketch.lines[*r].constraints.horizontal).count();
        let v_count = ctx.sketch.lines.refs().filter(|r| ctx.sketch.lines[*r].constraints.vertical).count();
        assert_eq!(h_count, 2);
        assert_eq!(v_count, 2);
        assert_eq!(ctx.sketch.perpendicular.len(), 0);
        assert_eq!(ctx.sketch.parallel.len(), 0);
    }

    #[test]
    fn test_add_rect_noconstraint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect 0,0 5,3 noconstraint");
        assert_eq!(ctx.sketch.lines.refs().count(), 4);
        assert_eq!(ctx.sketch.perpendicular.len(), 0);
        assert_eq!(ctx.sketch.parallel.len(), 0);
    }

    #[test]
    fn test_add_rect_driven() {
        let mut ctx = CommandContext::new();
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "add_rect 0,0 5,3 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(!ctx.sketch.dimensions[1].derived);
        assert!(ctx.sketch.dof().unwrap() < dof_before + 8); // 4 lines = +8 DOF, constraints + dims reduce
    }

    #[test]
    fn test_add_rect_noconstraint_conflicts() {
        let mut ctx = CommandContext::new();
        let r1 = execute_one(&mut ctx, "add_rect 0,0 5,3 noconstraint hv");
        assert!(r1.is_error);
        let r2 = execute_one(&mut ctx, "add_rect 0,0 5,3 noconstraint driven");
        assert!(r2.is_error);
        let r3 = execute_one(&mut ctx, "add_rect 0,0 5,3 noconstraint strict");
        assert!(r3.is_error);
    }

    #[test]
    fn test_add_rect_relative_coords() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect 0,0 @5,3");
        assert_eq!(ctx.sketch.lines.refs().count(), 4);
        let l0 = ctx.sketch.lines.refs().next().unwrap();
        let l = &ctx.sketch.lines[l0];
        assert!(near(l.p2.value.x - l.p1.value.x, 5.0));
    }

    #[test]
    fn test_add_rect_session_names() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect 0,0 5,3");
        assert_eq!(ctx.session_names.get("_0").map(|s| s.as_str()), Some("L0"));
        assert_eq!(ctx.session_names.get("_1").map(|s| s.as_str()), Some("L1"));
        assert_eq!(ctx.session_names.get("_2").map(|s| s.as_str()), Some("L2"));
        assert_eq!(ctx.session_names.get("_3").map(|s| s.as_str()), Some("L3"));
    }

    #[test]
    fn test_add_rect_noconnect() {
        let mut ctx = CommandContext::new();
        // Place a line at a rect corner
        run_ok(&mut ctx, "add_line 0,0 10,0");
        let coinc_before = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
            + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
        run_ok(&mut ctx, "add_rect 0,0 5,3 noconnect");
        let coinc_after = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
            + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
        // No new coincident constraints (not even internal ones)
        assert_eq!(coinc_after, coinc_before);
    }

    #[test]
    fn test_add_rect_non_strict_warns() {
        let mut ctx = CommandContext::new();
        // Fully constrain a rect
        run_ok(&mut ctx, "add_rect 0,0 5,3 hv driven");
        // Overlapping rect: constraints will be redundant, but non-strict should succeed with warnings
        let result = run_ok(&mut ctx, "add_rect 0,0 5,3");
        assert!(result.contains("warning"), "should contain warnings: {}", result);
    }

    #[test]
    fn test_add_rect3_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect3 0,0 5,0 5,3");
        assert_eq!(ctx.sketch.lines.refs().count(), 4);
        assert_eq!(ctx.sketch.perpendicular.len(), 1);
        assert_eq!(ctx.sketch.parallel.len(), 2);
    }

    #[test]
    fn test_add_rect3_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rect3 0,0 5,0 5,3 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(!ctx.sketch.dimensions[1].derived);
    }

    #[test]
    fn test_add_rect3_hv_rejected() {
        let mut ctx = CommandContext::new();
        let r = execute_one(&mut ctx, "add_rect3 0,0 5,0 5,3 hv");
        assert!(r.is_error);
    }

    #[test]
    fn test_add_rect3_collinear_rejected() {
        let mut ctx = CommandContext::new();
        let r = execute_one(&mut ctx, "add_rect3 1,1 2,3 3,5");
        assert!(r.is_error, "collinear points should be rejected: {}", r.output);
        assert!(r.output.contains("collinear"), "error should mention collinear: {}", r.output);
        assert_eq!(ctx.sketch.lines.refs().count(), 0, "no geometry should be created");
    }

    #[test]
    fn test_add_rectcenter_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rectcenter 2.5,1.5 0,0");
        assert_eq!(ctx.sketch.lines.refs().count(), 4);
        assert_eq!(ctx.sketch.perpendicular.len(), 1);
        assert_eq!(ctx.sketch.parallel.len(), 2);
        // Verify corners: bl=(0,0), br=(5,0), tr=(5,3), tl=(0,3)
        let refs: Vec<_> = ctx.sketch.lines.refs().collect();
        let l0 = &ctx.sketch.lines[refs[0]];
        assert!(near(l0.p1.value.x, 0.0) && near(l0.p1.value.y, 0.0));
        assert!(near(l0.p2.value.x, 5.0) && near(l0.p2.value.y, 0.0));
    }

    #[test]
    fn test_add_rectcenter_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_rectcenter 2.5,1.5 0,0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        assert!(!ctx.sketch.dimensions[0].derived);
    }

    // -- add_line / add_circle driven + new circle tools --

    #[test]
    fn test_short_line_does_not_break_solver() {
        // A very short line should not prevent constraints on other entities.
        // Previously, the length drift's sqrt had a singularity at zero length.
        // Note: truly zero-length lines are now softly penalized (Heaviside minimum
        // length), so we use a short-but-nonzero line instead.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,0.1; add_line 3,3 @0.02,0");
        run_ok(&mut ctx, "horizontal L0");
        let l = &ctx.sketch.lines[resolve_line(&ctx.sketch, "L0").unwrap()];
        assert!((l.p1.value.y - l.p2.value.y).abs() < 0.01);
    }

    #[test]
    fn test_add_line_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 5.0));
    }

    #[test]
    fn test_add_line_multi_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0 5,3 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        assert!(near(ctx.sketch.dimensions[0].value, 5.0));
        assert!(near(ctx.sketch.dimensions[1].value, 3.0));
    }

    #[test]
    fn test_add_circle_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 3.0));
    }

    #[test]
    fn test_add_arc_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc 0,0 5,0 2.5,2.5 driven noconnect");
        // Should have 2 dimensions: radius and sweep
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(!ctx.sketch.dimensions[1].derived);
        // Radius > 0
        assert!(ctx.sketch.dimensions[0].value > 0.0);
        // Sweep > 0
        assert!(ctx.sketch.dimensions[1].value > 0.0);
    }

    #[test]
    fn test_add_arc_driven_variable_assignment() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "l = add_line 0,0 1,0");
        run_ok(&mut ctx, "a = add_arc 1,0 3,1 2,0 driven noconnect");
        // Variable 'a' should resolve to the arc, not the dimension
        run_ok(&mut ctx, "tangent l a");
        assert_eq!(ctx.sketch.tangent_la.len(), 1);
    }

    #[test]
    fn test_add_circle2_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle2 0,0 10,0");
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let arc = &ctx.sketch.arcs[arc_ref];
        assert!(near(arc.center.value.x, 5.0));
        assert!(near(arc.center.value.y, 0.0));
        assert!(near(arc.radius.value, 5.0));
    }

    #[test]
    fn test_add_circle3_basic() {
        let mut ctx = CommandContext::new();
        // Points on a circle of radius 5 centered at origin
        run_ok(&mut ctx, "add_circle3 5,0 -5,0 0,5");
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let arc = &ctx.sketch.arcs[arc_ref];
        assert!(near(arc.center.value.x, 0.0));
        assert!(near(arc.center.value.y, 0.0));
        assert!(near(arc.radius.value, 5.0));
    }

    #[test]
    fn test_add_circle3_collinear_error() {
        let mut ctx = CommandContext::new();
        let r = execute_one(&mut ctx, "add_circle3 0,0 5,0 10,0");
        assert!(r.is_error);
    }

    #[test]
    fn test_add_circle2t_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
        let out = run_ok(&mut ctx, "add_circle2t L0 L1 2");
        assert!(out.contains("tangent L0"), "should list tangent L0: {}", out);
        assert!(out.contains("tangent L1"), "should list tangent L1: {}", out);
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let arc = &ctx.sketch.arcs[arc_ref];
        assert!(near(arc.center.value.x, 2.0));
        assert!(near(arc.center.value.y, 2.0));
        assert!(near(arc.radius.value, 2.0));
        assert_eq!(ctx.sketch.tangent_la.len(), 2);
    }

    #[test]
    fn test_add_circle2t_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
        run_ok(&mut ctx, "add_circle2t L0 L1 2 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 2.0));
    }

    #[test]
    fn test_add_circle2t_noconstraint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
        run_ok(&mut ctx, "add_circle2t L0 L1 2 noconstraint");
        assert_eq!(ctx.sketch.tangent_la.len(), 0);
    }

    #[test]
    fn test_add_circle3t_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 10,0 5,8; add_line 5,8 0,0");
        run_ok(&mut ctx, "add_circle3t L0 L1 L2");
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
        assert_eq!(ctx.sketch.tangent_la.len(), 3);
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let arc = &ctx.sketch.arcs[arc_ref];
        // Incircle should be inside the triangle
        assert!(arc.center.value.x > 0.0 && arc.center.value.x < 10.0);
        assert!(arc.center.value.y > 0.0 && arc.center.value.y < 8.0);
    }

    #[test]
    fn test_add_circle3t_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 10,0 5,8; add_line 5,8 0,0");
        run_ok(&mut ctx, "add_circle3t L0 L1 L2 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(!ctx.sketch.dimensions[0].derived);
    }

    #[test]
    fn test_add_circle2t_segment_disambiguation() {
        let mut ctx = CommandContext::new();
        // Two perpendicular rays from origin: only 1 of 4 sectors has both tangent points on segments
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,0 0,10");
        run_ok(&mut ctx, "add_circle2t L0 L1 1");
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let arc = &ctx.sketch.arcs[arc_ref];
        // Must be in the +x,+y quadrant (interior of the angle)
        assert!(arc.center.value.x > 0.0, "center.x should be positive: {}", arc.center.value.x);
        assert!(arc.center.value.y > 0.0, "center.y should be positive: {}", arc.center.value.y);
    }

    #[test]
    fn test_add_circle2t_no_touching() {
        let mut ctx = CommandContext::new();
        // Two parallel segments far apart: no tangent circle of r=1 touches both
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 0,100 10,100");
        let r = execute_one(&mut ctx, "add_circle2t L0 L1 1");
        assert!(r.is_error, "should fail: {}", r.output);
    }

    #[test]
    fn test_add_circle3t_segment_touches() {
        let mut ctx = CommandContext::new();
        // Closed triangle: exactly 1 incircle touches all 3 segments
        run_ok(&mut ctx, "add_line 0,0 10,0; add_line 10,0 5,8; add_line 5,8 0,0");
        run_ok(&mut ctx, "add_circle3t L0 L1 L2");
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let arc = &ctx.sketch.arcs[arc_ref];
        // Incircle center must be inside the triangle
        assert!(arc.center.value.x > 0.0 && arc.center.value.x < 10.0);
        assert!(arc.center.value.y > 0.0 && arc.center.value.y < 8.0);
        assert_eq!(ctx.sketch.tangent_la.len(), 3);
    }

    // -- mirror --

    #[test]
    fn test_mirror_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
        run_ok(&mut ctx, "mirror L1 about L0");
        assert_eq!(ctx.sketch.lines.refs().count(), 3);
        let refs: Vec<_> = ctx.sketch.lines.refs().collect();
        let mirrored = &ctx.sketch.lines[refs[2]];
        assert!(near(mirrored.p1.value.x, 3.0));
        assert!(near(mirrored.p2.value.x, 3.0));
        assert_eq!(ctx.sketch.symmetry_pp.len(), 2);
    }

    #[test]
    fn test_mirror_circle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_circle -5,3 2");
        run_ok(&mut ctx, "mirror A0 about L0");
        assert_eq!(ctx.sketch.arcs.refs().count(), 2);
        let refs: Vec<_> = ctx.sketch.arcs.refs().collect();
        let mirrored = &ctx.sketch.arcs[refs[1]];
        assert!(near(mirrored.center.value.x, 5.0));
        assert!(near(mirrored.center.value.y, 3.0));
        assert!(near(mirrored.radius.value, 2.0));
        assert_eq!(ctx.sketch.symmetry_pp.len(), 1); // center only for circles
    }

    #[test]
    fn test_mirror_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_point -4,5");
        run_ok(&mut ctx, "mirror P0 about L0");
        assert_eq!(ctx.sketch.points.refs().count(), 2);
        let refs: Vec<_> = ctx.sketch.points.refs().collect();
        let mirrored = &ctx.sketch.points[refs[1]];
        assert!(near(mirrored.pos.value.x, 4.0));
        assert!(near(mirrored.pos.value.y, 5.0));
        assert_eq!(ctx.sketch.symmetry_pp.len(), 1);
    }

    #[test]
    fn test_mirror_multiple() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -5,1 -5,4; add_line -5,4 -2,7");
        run_ok(&mut ctx, "mirror L1 L2 about L0");
        assert_eq!(ctx.sketch.lines.refs().count(), 5);
    }

    #[test]
    fn test_mirror_selection() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
        run_ok(&mut ctx, "select L1");
        run_ok(&mut ctx, "mirror selection about L0");
        assert_eq!(ctx.sketch.lines.refs().count(), 3);
    }

    #[test]
    fn test_mirror_session_names() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8; add_line -3,8 -6,5");
        run_ok(&mut ctx, "mirror L1 L2 about L0");
        assert_eq!(ctx.session_names.get("_0").map(|s| s.as_str()), Some("L3"));
        assert_eq!(ctx.session_names.get("_1").map(|s| s.as_str()), Some("L4"));
    }

    #[test]
    fn test_mirror_noconstraint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8; add_line -3,8 -6,5");
        run_ok(&mut ctx, "mirror L1 L2 about L0 noconstraint");
        assert_eq!(ctx.sketch.symmetry_pp.len(), 0);
        // No coincident recreation either
        let ll_coinc = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
            + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
        // Only the original L1.p2=L2.p1 coincident, no mirrored copy
        assert_eq!(ll_coinc, 1);
    }

    #[test]
    fn test_mirror_coincident_recreation() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -5,0 -2,3; add_line -2,3 -5,6");
        // L2.p1 = L1.p2 via auto-coincident (stored as LL12: a=L2, b=L1)
        let coinc_before = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
            + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
        run_ok(&mut ctx, "mirror L1 L2 about L0");
        let coinc_after = ctx.sketch.coincident_ll11.len() + ctx.sketch.coincident_ll12.len()
            + ctx.sketch.coincident_ll21.len() + ctx.sketch.coincident_ll22.len();
        assert_eq!(coinc_after, coinc_before + 1, "should recreate coincident among mirrored lines");
    }

    #[test]
    fn test_mirror_symmetry_dedup() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -5,0 -2,3; add_line -2,3 -5,6");
        run_ok(&mut ctx, "mirror L1 L2 about L0");
        // 4 endpoints total, but L1.p2=L2.p1 is shared, so 3 unique positions -> 3 symmetry constraints
        assert_eq!(ctx.sketch.symmetry_pp.len(), 3);
    }

    #[test]
    fn test_mirror_missing_about() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
        let r = execute_one(&mut ctx, "mirror L1 L0");
        assert!(r.is_error, "should require 'about' keyword: {}", r.output);
    }

    #[test]
    fn test_mirror_output_lists_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,10; add_line -3,2 -3,8");
        let out = run_ok(&mut ctx, "mirror L1 about L0");
        assert!(out.contains("symmetry"), "should list symmetry: {}", out);
        assert!(out.contains("Mirrored L1"), "should list mirrored entity: {}", out);
    }

    // -- add_ellipse --

    #[test]
    fn test_add_ellipse_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 45");
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let a = &ctx.sketch.arcs[arc_ref];
        assert!(a.is_ellipse);
        assert!(a.closed);
        assert!(near(a.radius.value, 5.0));
        assert!(near(a.radius_b.value, 3.0));
        assert!(near(a.rotation.value.to_degrees(), 45.0));
    }

    #[test]
    fn test_add_ellipse_dof() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 0");
        // DOF: center(2) + rx(1) + ry(1) + rotation(1) = 5
        assert_eq!(ctx.sketch.dof().unwrap(), 5);
    }

    #[test]
    fn test_add_ellipse_list_output() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 30");
        let out = run_ok(&mut ctx, "list");
        assert!(out.contains("[ellipse]"), "should show [ellipse]: {}", out);
        assert!(out.contains("rx="), "should show rx: {}", out);
        assert!(out.contains("ry="), "should show ry: {}", out);
    }

    #[test]
    fn test_add_ellipse_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 0 driven");
        assert_eq!(ctx.sketch.dimensions.len(), 2);
        assert!(near(ctx.sketch.dimensions[0].value, 5.0));
        assert!(near(ctx.sketch.dimensions[1].value, 3.0));
    }

    #[test]
    fn test_ellipse_print_params() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 45");
        let out = run_ok(&mut ctx, "print EA0.rotation");
        let val: f64 = out.trim().parse().unwrap();
        assert!(near(val, std::f64::consts::PI / 4.0));
        let out = run_ok(&mut ctx, "print EA0.radius_b");
        let val: f64 = out.trim().parse().unwrap();
        assert!(near(val, 3.0));
    }

    #[test]
    fn test_ellipse_radius_b_command() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 0");
        run_ok(&mut ctx, "radius_b EA0 4");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(near(ctx.sketch.dimensions[0].value, 4.0));
        assert!(near(ctx.sketch.arcs.refs().next().map(|r| ctx.sketch.arcs[r].radius_b.value).unwrap(), 4.0));
    }

    #[test]
    fn test_radius_b_rejects_non_ellipse() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        let r = execute_one(&mut ctx, "radius_b A0 3");
        assert!(r.is_error, "radius_b should reject non-ellipse: {}", r.output);
    }

    #[test]
    fn test_ellipse_measure() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse 0,0 5 3 30");
        let out = run_ok(&mut ctx, "measure EA0");
        assert!(out.contains("rx="), "should show rx: {}", out);
        assert!(out.contains("ry="), "should show ry: {}", out);
        assert!(out.contains("rotation="), "should show rotation: {}", out);
    }

    #[test]
    fn test_arc_unchanged_by_ellipse_fields() {
        // Verify that existing arcs/circles still work identically
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let a = &ctx.sketch.arcs[arc_ref];
        assert!(!a.is_ellipse);
        assert!(near(a.radius_b.value, 5.0)); // radius_b == radius
        assert!(near(a.rotation.value, 0.0));
        assert!(a.radius_b.optimize); // optimizable (equality constraint keeps it = radius)
        assert!(!a.rotation.optimize); // fixed
    }

    #[test]
    fn test_ellipse_start_end_pos() {
        // Verify ellipse point formula works for arc_start_pos/arc_end_pos
        let mut ctx = CommandContext::new();
        // Unrotated ellipse: rx=4, ry=2. At angle 0, point = (cx+4, cy)
        run_ok(&mut ctx, "add_ellipse 0,0 4 2 0");
        let arc_ref = ctx.sketch.arcs.refs().next().unwrap();
        let sp = crate::geometry::arc_start_pos(&ctx.sketch.arcs[arc_ref]);
        assert!(near(sp.x, 4.0));
        assert!(near(sp.y, 0.0));
    }

    // -- Measure --

    #[test]
    fn test_measure_single_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        let out = run_ok(&mut ctx, "measure L0");
        assert!(out.contains("length=5.0000"), "should show length: {}", out);
    }

    #[test]
    fn test_measure_two_parallel_lines() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        let out = run_ok(&mut ctx, "measure L0 L1");
        assert!(out.contains("parallel"), "should detect parallel: {}", out);
        assert!(out.contains("3.0000"), "should show distance: {}", out);
    }

    #[test]
    fn test_measure_two_lines_angle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,3");
        let out = run_ok(&mut ctx, "measure L0 L1");
        assert!(out.contains("45.0000"), "should show 45 deg: {}", out);
        assert!(out.contains("135.0000"), "should show supplement: {}", out);
    }

    #[test]
    fn test_measure_two_points() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        let out = run_ok(&mut ctx, "measure L0.p1 L0.p2");
        assert!(out.contains("5.0000"), "should show distance 5: {}", out);
    }

    #[test]
    fn test_measure_point_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_point 2,3");
        let out = run_ok(&mut ctx, "measure P0 L0");
        assert!(out.contains("3.0000"), "should show perp distance 3: {}", out);
    }

    #[test]
    fn test_measure_single_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        let out = run_ok(&mut ctx, "measure A0");
        assert!(out.contains("radius=5.0000"), "should show radius: {}", out);
    }

    // -- Arc-arc symmetry --

    #[test]
    fn test_symmetry_aa_command() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 4,1 2");
        let dof_before = ctx.sketch.dof().unwrap();
        run_ok(&mut ctx, "symmetry A0 L0 A1");
        let dof_after = ctx.sketch.dof().unwrap();
        assert_eq!(dof_after, dof_before - 3, "arc symmetry should remove 3 DOF: {} -> {}", dof_before, dof_after);
        assert_eq!(ctx.sketch.symmetry_aa.len(), 1);
    }

    #[test]
    fn test_symmetry_aa_equal_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 3,0 2");
        run_ok(&mut ctx, "symmetry A0 L0 A1");
        let r0 = ctx.sketch.arcs.iter().next().unwrap().radius.value;
        let r1 = ctx.sketch.arcs.iter().nth(1).unwrap().radius.value;
        assert!((r0 - r1).abs() < 0.01, "radii should be equal: {} vs {}", r0, r1);
    }

    #[test]
    fn test_symmetry_aa_remove() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 3,0 1");
        run_ok(&mut ctx, "symmetry A0 L0 A1");
        assert_eq!(ctx.sketch.symmetry_aa.len(), 1);
        run_ok(&mut ctx, "delete A0 L0 A1 symmetry");
        assert_eq!(ctx.sketch.symmetry_aa.len(), 0);
    }

    #[test]
    fn test_symmetry_aa_duplicate() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,-5 0,5; add_circle -3,0 1; add_circle 3,0 1");
        run_ok(&mut ctx, "symmetry A0 L0 A1");
        let e = run_err(&mut ctx, "symmetry A0 L0 A1");
        assert!(e.contains("already exists"), "{}", e);
    }

    #[test]
    fn test_symmetry_aa_ellipse() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,-5 0,5; add_ellipse -3,0 2 1 30; add_ellipse 4,1 3 2 0");
        run_ok(&mut ctx, "symmetry EA0 L0 EA1");
        let a0 = ctx.sketch.arcs.refs().nth(0).unwrap();
        let a1 = ctx.sketch.arcs.refs().nth(1).unwrap();
        let r0 = ctx.sketch.arcs[a0].radius.value;
        let r1 = ctx.sketch.arcs[a1].radius.value;
        assert!((r0 - r1).abs() < 0.01, "radii should be equal: {} vs {}", r0, r1);
        let rb0 = ctx.sketch.arcs[a0].radius_b.value;
        let rb1 = ctx.sketch.arcs[a1].radius_b.value;
        assert!((rb0 - rb1).abs() < 0.01, "radius_b should be equal: {} vs {}", rb0, rb1);
    }

    #[test]
    fn test_tangent_aa_ellipse() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_ellipse -5,0 3 2 0; add_ellipse 5,0 3 2 0");
        run_ok(&mut ctx, "tangent EA0 EA1");
        assert_eq!(ctx.sketch.tangent_aa.len(), 1);
        // Verify constraint is satisfied: dist = r_eff_a + r_eff_b
        let a0 = ctx.sketch.arcs.refs().nth(0).unwrap();
        let a1 = ctx.sketch.arcs.refs().nth(1).unwrap();
        let aa = &ctx.sketch.arcs[a0];
        let bb = &ctx.sketch.arcs[a1];
        let dx = aa.center.value.x - bb.center.value.x;
        let dy = aa.center.value.y - bb.center.value.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let nx = dx / dist;
        let ny = dy / dist;
        // Effective radius of a
        let cra = aa.rotation.value.cos();
        let sra = aa.rotation.value.sin();
        let nxa = nx * cra + ny * sra;
        let nya = -nx * sra + ny * cra;
        let r_eff_a = (nxa * nxa * aa.radius.value * aa.radius.value + nya * nya * aa.radius_b.value * aa.radius_b.value).sqrt();
        // Effective radius of b
        let crb = bb.rotation.value.cos();
        let srb = bb.rotation.value.sin();
        let nxb = -nx * crb - ny * srb;
        let nyb = nx * srb - ny * crb;
        let r_eff_b = (nxb * nxb * bb.radius.value * bb.radius.value + nyb * nyb * bb.radius_b.value * bb.radius_b.value).sqrt();
        let residual = (dist - r_eff_a - r_eff_b).abs();
        assert!(residual < 0.01, "tangent should be satisfied: dist={}, r_eff_a={}, r_eff_b={}, residual={}", dist, r_eff_a, r_eff_b, residual);
    }

    // -- List constraint filtering --

    #[test]
    fn test_list_filter_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "horizontal L0; horizontal L1");
        let out = run_ok(&mut ctx, "list horizontal");
        assert!(out.contains("horizontal L0"), "should list L0: {}", out);
        assert!(out.contains("horizontal L1"), "should list L1: {}", out);
        assert!(!out.contains("coincident"), "should not include other types: {}", out);
    }

    #[test]
    fn test_list_filter_empty() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "list parallel");
        assert_eq!(out, "(empty)");
    }

    #[test]
    fn test_list_filter_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line @0,3");
        let out = run_ok(&mut ctx, "list coincident");
        assert!(out.contains("coincident"), "should show coincident: {}", out);
        assert!(!out.contains("L0:"), "should not include entity listing: {}", out);
    }

    // -- Auto-assigned constraint names (C<n>, CL0H) --

    #[test]
    fn test_rc_by_numeric_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "parallel L0 L1");
        assert_eq!(ctx.sketch.parallel.len(), 1);
        let nid = ctx.sketch.parallel[0].nid;
        let out = run_ok(&mut ctx, &format!("delete C{}", nid));
        assert!(out.contains(&format!("C{}", nid)), "output should mention name: {}", out);
        assert!(out.contains("parallel"), "output should describe constraint: {}", out);
        assert_eq!(ctx.sketch.parallel.len(), 0);
    }

    #[test]
    fn test_rc_by_flag_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 0,7");
        run_ok(&mut ctx, "horizontal L0; vertical L1");
        assert!(ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].constraints.horizontal);
        let out = run_ok(&mut ctx, "delete CL0H");
        assert!(out.contains("CL0H"), "output should mention name: {}", out);
        assert!(out.contains("horizontal L0"), "output should describe constraint: {}", out);
        let l0 = ctx.sketch.lines.refs().next().unwrap();
        assert!(!ctx.sketch.lines[l0].constraints.horizontal);
        // CL1V still set until removed.
        let l1 = ctx.sketch.lines.refs().nth(1).unwrap();
        assert!(ctx.sketch.lines[l1].constraints.vertical);
        run_ok(&mut ctx, "delete CL1V");
        assert!(!ctx.sketch.lines[l1].constraints.vertical);
    }

    #[test]
    fn test_rc_unknown_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let e = run_err(&mut ctx, "delete C999");
        assert!(e.contains("Unknown"), "{}", e);
        let e = run_err(&mut ctx, "delete CL0H");
        assert!(e.contains("Unknown"), "{}", e);
    }

    #[test]
    fn test_info_by_constraint_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "parallel L0 L1");
        let nid = ctx.sketch.parallel[0].nid;
        let out = run_ok(&mut ctx, &format!("info C{}", nid));
        assert!(out.contains("parallel"), "info output: {}", out);
        assert!(out.contains(&format!("C{}:", nid)), "info output: {}", out);
    }

    #[test]
    fn test_info_by_flag_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        let out = run_ok(&mut ctx, "info CL0H");
        assert!(out.contains("horizontal L0"), "info output: {}", out);
    }

    #[test]
    fn test_list_includes_constraint_names() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "horizontal L0; parallel L0 L1");
        let out = run_ok(&mut ctx, "list");
        assert!(out.contains("CL0H: horizontal L0"), "list output: {}", out);
        assert!(out.contains(": parallel L0 L1"), "list output: {}", out);
    }

    #[test]
    fn test_rc_entity_syntax_still_works() {
        // Existing entity-pair + type dispatch keeps working alongside name lookup.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "parallel L0 L1");
        assert_eq!(ctx.sketch.parallel.len(), 1);
        run_ok(&mut ctx, "delete L0 L1 parallel");
        assert_eq!(ctx.sketch.parallel.len(), 0);
    }

    #[test]
    fn test_list_and_info_show_range_bound() {
        // Range dimensions must be identifiable from script output (was
        // previously indistinguishable from a numeric dim in both `list
        // dims` and `info d0`).
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 3 to 7");
        let out = run_ok(&mut ctx, "list dims");
        assert!(out.contains("3 to 7"), "list dims should show range: {}", out);
        let out = run_ok(&mut ctx, "info d0");
        assert!(out.contains("range=3 to 7"), "info should mark range: {}", out);
        assert!(!out.contains("expr=(numeric)"), "info should not say numeric: {}", out);

        // One-sided and live-expression forms stay recognisable too.
        run_ok(&mut ctx, "add_line 0,3 5,3");
        run_ok(&mut ctx, "length L1 >= 2");
        let out = run_ok(&mut ctx, "info d1");
        assert!(out.contains("range=>= 2"), "one-sided: {}", out);

        run_ok(&mut ctx, "param lo 2");
        run_ok(&mut ctx, "length L0 lo to 8");
        let out = run_ok(&mut ctx, "list dims");
        assert!(out.contains("lo to 8"), "live range: {}", out);
    }

    #[test]
    fn test_constraint_names_survive_undo_redo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,2 5,2");
        run_ok(&mut ctx, "parallel L0 L1");
        let nid_before = ctx.sketch.parallel[0].nid;
        run_ok(&mut ctx, "undo");
        assert_eq!(ctx.sketch.parallel.len(), 0);
        run_ok(&mut ctx, "redo");
        assert_eq!(ctx.sketch.parallel.len(), 1);
        assert_eq!(ctx.sketch.parallel[0].nid, nid_before);
    }

    #[test]
    fn test_add_earc() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc 0,0 5,0 3 1 0 noconnect");
        assert_eq!(ctx.sketch.arcs.len(), 1);
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        assert!(!a.closed);
        let sp = crate::geometry::arc_start_pos(a);
        let ep = crate::geometry::arc_end_pos(a);
        assert!((sp.x - 0.0).abs() < 0.1, "start x: {}", sp.x);
        assert!((sp.y - 0.0).abs() < 0.1, "start y: {}", sp.y);
        assert!((ep.x - 5.0).abs() < 0.1, "end x: {}", ep.x);
        assert!((ep.y - 0.0).abs() < 0.1, "end y: {}", ep.y);
    }

    #[test]
    fn test_add_earc_large() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc 0,0 5,0 3 1 0 large noconnect");
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        let sweep = (a.end_angle.value - a.start_angle.value).abs();
        assert!(sweep > std::f64::consts::PI, "sweep {:.2} should be > pi for large arc", sweep);
    }

    #[test]
    fn test_add_earc_cw() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc 0,0 5,0 3 1 0 cw noconnect");
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(!a.ccw, "should be clockwise");
    }

    #[test]
    fn test_add_earc_center() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc_center 0,0 3 1 45 0 90 noconnect");
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        assert!(!a.closed);
        assert!((a.center.value.x).abs() < 0.01);
        assert!((a.center.value.y).abs() < 0.01);
        assert!((a.radius.value - 3.0).abs() < 0.01);
        assert!((a.radius_b.value - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_add_earc3() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc3 0,0 5,0 2,2 3 1 noconnect");
        assert_eq!(ctx.sketch.arcs.len(), 1);
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        assert!(!a.closed);
    }

    #[test]
    fn test_add_earc_driven() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc 0,0 5,0 3 1 0 driven noconnect");
        assert_eq!(ctx.sketch.dimensions.len(), 2);
    }

    // --- Auto-tangent tests ---

    #[test]
    fn test_auto_tangent_line_arc() {
        let mut ctx = CommandContext::new();
        // Horizontal line, then arc tangent at p2
        run(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_arc 5,0 5,5 7.5,2.5");
        assert!(out.contains("tangent"), "expected auto-tangent: {}", out);
        assert_eq!(ctx.sketch.tangent_la.len(), 1);
    }

    #[test]
    fn test_auto_tangent_not_applied_when_not_tangent() {
        let mut ctx = CommandContext::new();
        // Horizontal line, then arc clearly not tangent (goes straight up)
        run(&mut ctx, "add_line 0,0 5,0");
        run(&mut ctx, "add_arc 5,0 5,5 4,2.5");
        assert_eq!(ctx.sketch.tangent_la.len(), 0, "should not auto-tangent non-tangent geometry");
    }

    #[test]
    fn test_auto_tangent_notangent_keyword() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_arc 5,0 5,5 7.5,2.5 notangent");
        assert!(!out.contains("tangent"), "notangent should suppress: {}", out);
        assert_eq!(ctx.sketch.tangent_la.len(), 0);
    }

    #[test]
    fn test_auto_tangent_noconnect_implies_notangent() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0");
        run(&mut ctx, "add_arc 5,0 5,5 7.5,2.5 noconnect");
        assert_eq!(ctx.sketch.tangent_la.len(), 0);
    }

    #[test]
    fn test_auto_tangent_arc_arc() {
        let mut ctx = CommandContext::new();
        // Two arcs sharing an endpoint, tangent at the junction
        // First arc: semicircle from (0,0) to (2,0) with center (1,1)
        run(&mut ctx, "add_arc 0,0 2,0 1,1 noconnect");
        // Second arc: continues tangent from (2,0)
        let out = run_ok(&mut ctx, "add_arc 2,0 4,0 3,1");
        assert!(out.contains("tangent"), "expected arc-arc auto-tangent: {}", out);
        assert_eq!(ctx.sketch.tangent_aa.len(), 1);
    }

    #[test]
    fn test_tangent_aa_shared_endpoint() {
        // Manual tangent command with shared endpoint should use cross-product formula
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_earc 377.953,0 204.989,2.559 2017.12 2017.12 0 noconnect");
        run(&mut ctx, "add_earc 204.989,2.559 142.625,0.378 4123.85 4123.85 0 noconnect notangent");
        run_ok(&mut ctx, "tangent EA0 EA1");
        assert_eq!(ctx.sketch.tangent_aa.len(), 1);
        assert_ne!(ctx.sketch.tangent_aa[0].shared, SharedEndpoint::None);
    }

    // --- Quiet mode tests ---

    #[test]
    fn test_quiet_keyword_line() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 quiet noconnect");
        let r = ctx.sketch.lines.refs().next().unwrap();
        assert!(ctx.sketch.lines[r].quiet);
    }

    #[test]
    fn test_quiet_keyword_arc() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_circle 0,0 5 quiet noconnect");
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!(ctx.sketch.arcs[r].quiet);
    }

    #[test]
    fn test_quiet_toggle() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        let r = ctx.sketch.lines.refs().next().unwrap();
        assert!(!ctx.sketch.lines[r].quiet);
        run_ok(&mut ctx, "quiet L0");
        assert!(ctx.sketch.lines[r].quiet);
        run_ok(&mut ctx, "quiet L0");
        assert!(!ctx.sketch.lines[r].quiet);
    }

    #[test]
    fn test_quiet_on_off() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        let r = ctx.sketch.lines.refs().next().unwrap();
        run_ok(&mut ctx, "quiet L0 on");
        assert!(ctx.sketch.lines[r].quiet);
        run_ok(&mut ctx, "quiet L0 off");
        assert!(!ctx.sketch.lines[r].quiet);
    }

    #[test]
    fn test_quiet_info() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 quiet noconnect");
        let out = run_ok(&mut ctx, "info L0");
        assert!(out.contains("[quiet]"), "info should show [quiet]: {}", out);
    }

    // --- add_earc_tangent tests ---

    #[test]
    fn test_add_earc_tangent_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_earc_tangent 0,0 1,0 5,5 0,1 noconnect");
        assert_eq!(ctx.sketch.arcs.len(), 1);
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        assert!(!a.closed);
    }

    #[test]
    fn test_add_earc_tangent_bulge() {
        let mut ctx = CommandContext::new();
        // Semicircle-like: opposing tangents, bulge=1
        run_ok(&mut ctx, "add_earc_tangent 0,0 0,1 10,0 0,-1 1 noconnect");
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        // bulge=1 with symmetric tangents should be near-circular
        assert!((a.radius.value - a.radius_b.value).abs() < 0.5,
            "expected near-circular: rx={:.3} ry={:.3}", a.radius.value, a.radius_b.value);
    }

    #[test]
    fn test_cursor_tangent_from_line() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        assert!(ctx.cursor_tangent.is_some(), "tangent should be set after add_line");
        let t = ctx.cursor_tangent.unwrap();
        assert!((t.x - 1.0).abs() < 0.01, "tangent x: {}", t.x);
        assert!(t.y.abs() < 0.01, "tangent y: {}", t.y);
    }

    #[test]
    fn test_cursor_tangent_chaining() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        run_ok(&mut ctx, "add_earc_tangent @cursor @tangent 10,3 0,1 noconnect");
        // Cursor should now be at (10,3)
        assert!(ctx.cursor.is_some());
        let c = ctx.cursor.unwrap();
        assert!((c.x - 10.0).abs() < 0.1, "cursor x: {}", c.x);
        assert!((c.y - 3.0).abs() < 0.1, "cursor y: {}", c.y);
        // Tangent should be set from arc end
        assert!(ctx.cursor_tangent.is_some(), "tangent should be set after earc_tangent");
    }

    #[test]
    fn test_cursor_tangent_from_arc() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_arc 0,0 5,0 2.5,2.5 noconnect");
        assert!(ctx.cursor_tangent.is_some(), "tangent should be set after add_arc");
    }

    // --- add_earc_rtangent tests ---

    #[test]
    fn test_add_earc_rtangent_basic() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        run_ok(&mut ctx, "add_earc_rtangent 10,3 0,1 0.5 noconnect");
        assert_eq!(ctx.sketch.arcs.len(), 1);
        let a = &ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()];
        assert!(a.is_ellipse);
        // Start should be at cursor (5,0)
        let sp = crate::geometry::arc_start_pos(a);
        assert!((sp.x - 5.0).abs() < 0.01, "start x: {:.4}", sp.x);
        assert!((sp.y - 0.0).abs() < 0.01, "start y: {:.4}", sp.y);
    }

    #[test]
    fn test_add_earc_rtangent_chaining() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        run_ok(&mut ctx, "add_earc_rtangent 10,3 0,1 0.5 noconnect");
        // Cursor should now be at (10,3) and tangent set
        let c = ctx.cursor.unwrap();
        assert!((c.x - 10.0).abs() < 0.1, "cursor x: {:.4}", c.x);
        assert!((c.y - 3.0).abs() < 0.1, "cursor y: {:.4}", c.y);
        assert!(ctx.cursor_tangent.is_some());
        // Chain another
        run_ok(&mut ctx, "add_earc_rtangent 15,0 1,0 0.5 noconnect");
        assert_eq!(ctx.sketch.arcs.len(), 2);
    }

    #[test]
    fn test_add_earc_rtangent_no_cursor() {
        let mut ctx = CommandContext::new();
        // No line/arc created yet — should fail
        let results = crate::commands::execute(&mut ctx, "add_earc_rtangent 10,3 0,1 0.5");
        assert!(results[0].is_error, "should fail without cursor");
    }

    // --- Construction tests ---

    #[test]
    fn test_constr_keyword_line() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 constr noconnect");
        let r = ctx.sketch.lines.refs().next().unwrap();
        assert!(ctx.sketch.lines[r].construction);
        assert_eq!(ctx.sketch.lines[r].style, arael_sketch_solver::LineStyle::DashDot);
    }

    #[test]
    fn test_constr_keyword_arc() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_circle 0,0 5 constr noconnect");
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!(ctx.sketch.arcs[r].construction);
        assert_eq!(ctx.sketch.arcs[r].style, arael_sketch_solver::LineStyle::DashDot);
    }

    #[test]
    fn test_constr_toggle() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        let r = ctx.sketch.lines.refs().next().unwrap();
        assert!(!ctx.sketch.lines[r].construction);
        run_ok(&mut ctx, "constr L0");
        assert!(ctx.sketch.lines[r].construction);
        assert_eq!(ctx.sketch.lines[r].style, arael_sketch_solver::LineStyle::DashDot);
        run_ok(&mut ctx, "constr L0");
        assert!(!ctx.sketch.lines[r].construction);
        assert_eq!(ctx.sketch.lines[r].style, arael_sketch_solver::LineStyle::Solid);
    }

    #[test]
    fn test_constr_info() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 constr noconnect");
        let out = run_ok(&mut ctx, "info L0");
        assert!(out.contains("[constr]"), "info should show [constr]: {}", out);
    }

    #[test]
    fn test_list_constr_filter() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 constr noconnect");
        run(&mut ctx, "add_line 1,0 6,0 noconnect");
        let out = run_ok(&mut ctx, "list constr");
        assert!(out.contains("L0"), "should list L0: {}", out);
        assert!(!out.contains("L1"), "should not list L1: {}", out);
    }

    // --- Drag tests ---

    #[test]
    fn test_drag_line_endpoint() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        let out = run_ok(&mut ctx, "drag L0.p2 5,3");
        assert!(out.contains("Dragged"), "{}", out);
        let p2 = ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].p2.value;
        assert!((p2.x - 5.0).abs() < 0.1, "p2.x={:.4}", p2.x);
        assert!((p2.y - 3.0).abs() < 0.1, "p2.y={:.4}", p2.y);
    }

    #[test]
    fn test_drag_relative() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        run_ok(&mut ctx, "drag L0.p2 @0,3");
        let p2 = ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()].p2.value;
        assert!((p2.x - 5.0).abs() < 0.1, "p2.x={:.4}", p2.x);
        assert!((p2.y - 3.0).abs() < 0.1, "p2.y={:.4}", p2.y);
    }

    #[test]
    fn test_drag_point() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_point 0,0");
        run_ok(&mut ctx, "drag P0 3,3");
        let p = ctx.sketch.points[ctx.sketch.points.refs().next().unwrap()].pos.value;
        assert!((p.x - 3.0).abs() < 0.1, "x={:.4}", p.x);
        assert!((p.y - 3.0).abs() < 0.1, "y={:.4}", p.y);
    }

    #[test]
    fn test_drag_constrained() {
        let mut ctx = CommandContext::new();
        run(&mut ctx, "add_line 0,0 5,0 noconnect");
        run_ok(&mut ctx, "horizontal L0");
        // Drag p2 to (5,3) — horizontal should keep y equal
        run_ok(&mut ctx, "drag L0.p2 5,3");
        let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
        assert!((l.p1.value.y - l.p2.value.y).abs() < 0.1, "should stay horizontal: p1.y={:.4} p2.y={:.4}", l.p1.value.y, l.p2.value.y);
    }

    /// Soft drag (default): dragging L0.p2 toward an infeasible target
    /// must leave the sketch at its relaxed cost ~ 0 state, with the
    /// dragged endpoint lagging at the nearest feasible point rather
    /// than forcing the sketch to deform.
    #[test]
    fn test_drag_soft_respects_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "lock L0.p1");
        run_ok(&mut ctx, "length L0 5");
        // Target (10, 0) is outside the reachable circle (radius 5).
        run_ok(&mut ctx, "drag L0.p2 10,0");
        let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
        let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
        assert!((len - 5.0).abs() < 0.01, "length constraint must hold: {:.4}", len);
        // p1 stays locked at origin.
        assert!(l.p1.value.x.abs() < 0.01 && l.p1.value.y.abs() < 0.01,
            "p1 must stay locked: {:?}", l.p1.value);
        // p2 lands at (5, 0), the nearest feasible point toward (10, 0).
        assert!((l.p2.value.x - 5.0).abs() < 0.1 && l.p2.value.y.abs() < 0.1,
            "p2 should relax to nearest feasible point: {:?}", l.p2.value);
    }

    /// Feasible soft drag must still land exactly on the cursor target.
    /// Verifies the drag-pull attractor dominates background drift so
    /// unconstrained DOFs track the cursor rather than compromising
    /// between cursor and starting position.
    #[test]
    fn test_drag_soft_feasible_hits_target() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "drag L0.p2 3,4");
        let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
        assert!((l.p2.value.x - 3.0).abs() < 0.01 && (l.p2.value.y - 4.0).abs() < 0.01,
            "drag should land at cursor: {:?}", l.p2.value);
    }

    // -- Range-dimension transitions (regression tests) --

    /// Regression: updating a range dim with a bare numeric value must
    /// clear `dim.range` so the barrier residual stops firing in
    /// `rebuild_expr_constraints`. Previously the old range stuck and
    /// the geometry couldn't leave the band.
    #[test]
    fn test_range_to_numeric_transition() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 3 to 6");
        // Currently 5, inside the [3, 6] band.
        assert!(near(line_len(&ctx, "L0"), 5.0));
        // Set to 2 (outside the band). With the bug, length would stay
        // clamped to 3 because the old barrier still applied.
        run_ok(&mut ctx, "length L0 2");
        assert!(near(line_len(&ctx, "L0"), 2.0),
            "expected length 2.0 after range->numeric, got {:.4}", line_len(&ctx, "L0"));
    }

    /// Range dimensions must not contribute to the reported DOF --
    /// their Jacobian row is zero inside the feasible band and
    /// non-zero at/outside the bound, so including them would make
    /// DOF swing by one as the user drags geometry across the bound.
    /// Reported DOF should be the geometric DOF, stable regardless
    /// of whether the barrier is currently active.
    #[test]
    fn test_range_dim_does_not_affect_dof() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 3 to 6");
        let dof_inside = ctx.sketch.dof().unwrap();
        // Push onto the lower bound: barrier would be active there.
        run_ok(&mut ctx, "length L0 1 to 6");
        let dof_at_lower = ctx.sketch.dof().unwrap();
        // Push onto the upper bound.
        run_ok(&mut ctx, "length L0 3 to 4");
        let dof_at_upper = ctx.sketch.dof().unwrap();
        assert_eq!(dof_inside, dof_at_lower,
            "DOF must be stable; inside={}, at lower={}", dof_inside, dof_at_lower);
        assert_eq!(dof_inside, dof_at_upper,
            "DOF must be stable; inside={}, at upper={}", dof_inside, dof_at_upper);
    }

    /// Regression: updating a numeric dim to a range must drop the old
    /// per-kind equality constraint (e.g. `has_length = 5`) so the
    /// barrier can drive the parameter into the band unopposed.
    /// Previously both constraints stayed and the equality won.
    #[test]
    fn test_numeric_to_range_transition() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 5");
        assert!(near(line_len(&ctx, "L0"), 5.0));
        // Upper bound 3: barrier must clamp length down from 5 to 3.
        run_ok(&mut ctx, "length L0 2 to 3");
        assert!(near(line_len(&ctx, "L0"), 3.0),
            "expected length 3.0 after numeric->range clamp, got {:.4}", line_len(&ctx, "L0"));
    }

    #[test]
    fn test_bare_expr_is_live() {
        // Bare parameter names in a dimension value are live by
        // default: `length L0 w` creates an expression dim that
        // tracks `w`. Snapshot form `=w` bakes the current value
        // in as a literal.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "param w 5");
        run_ok(&mut ctx, "length L0 w");
        run_ok(&mut ctx, "param w 8");
        assert!(near(line_len(&ctx, "L0"), 8.0),
            "bare `w` must be live; len after `param w 8`: {:.4}",
            line_len(&ctx, "L0"));
    }

    #[test]
    fn test_eq_prefix_is_snapshot() {
        // `=w` snapshots: length is baked as a literal at command
        // time and does not track later param changes.
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "param w 5");
        run_ok(&mut ctx, "length L0 =w");
        run_ok(&mut ctx, "param w 8");
        assert!(near(line_len(&ctx, "L0"), 5.0),
            "`=w` must snapshot; len after `param w 8`: {:.4}",
            line_len(&ctx, "L0"));
    }

    #[test]
    fn test_range_bare_expr_is_live() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        run_ok(&mut ctx, "param lo 2");
        run_ok(&mut ctx, "param hi 6");
        run_ok(&mut ctx, "length L0 lo to hi");
        // Current 4 is inside [2, 6], bound inactive.
        assert!(near(line_len(&ctx, "L0"), 4.0));
        // Shrink the band by moving `hi` below the current length;
        // the barrier activates and clamps.
        run_ok(&mut ctx, "param hi 3");
        assert!(near(line_len(&ctx, "L0"), 3.0),
            "range must track `hi`; len after `param hi 3`: {:.4}",
            line_len(&ctx, "L0"));
    }

    #[test]
    fn test_xangle_range() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");  // xangle 0
        run_ok(&mut ctx, "xangle L0 30 to 60");
        let l = &ctx.sketch.lines[ctx.sketch.lines.refs().next().unwrap()];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let ang = dy.atan2(dx).to_degrees();
        assert!((30.0..=60.0).contains(&ang) || ang.abs() < 0.1,
            "xangle {:.4} not in [30, 60] after Between(30, 60)", ang);
        // The solver should have rotated the line to at least reach the band.
        // (The lower bound activates; exact landing is near 30 given the initial 0.)
        assert!(ang >= 30.0 - 0.1,
            "xangle lower bound: {:.4} should be >= 30", ang);
    }

    #[test]
    fn test_radius_range() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        // Lower bound activates: radius grows from 5 to 7.
        run_ok(&mut ctx, "radius A0 >= 7");
        let r = ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].radius.value;
        assert!(near(r, 7.0), "radius after Min(7): {:.4}", r);
        // Transition to a two-sided range clamping down.
        run_ok(&mut ctx, "radius A0 2 to 4");
        let r = ctx.sketch.arcs[ctx.sketch.arcs.refs().next().unwrap()].radius.value;
        assert!(near(r, 4.0), "radius after 2..4: {:.4}", r);
    }

    /// Round-trip: numeric -> range -> numeric -> range exercises both
    /// transition directions through the same command path.
    #[test]
    fn test_range_numeric_roundtrip() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 5");
        assert!(near(line_len(&ctx, "L0"), 5.0));
        run_ok(&mut ctx, "length L0 2 to 3");
        assert!(near(line_len(&ctx, "L0"), 3.0));
        run_ok(&mut ctx, "length L0 4");
        assert!(near(line_len(&ctx, "L0"), 4.0),
            "range->numeric: got {:.4}", line_len(&ctx, "L0"));
        run_ok(&mut ctx, "length L0 1 to 2");
        assert!(near(line_len(&ctx, "L0"), 2.0),
            "numeric->range clamp: got {:.4}", line_len(&ctx, "L0"));
    }
}
