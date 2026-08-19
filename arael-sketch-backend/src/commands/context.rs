use super::*;

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
    /// Notices the sketch raised while actions were applied (an offset
    /// dropped because its result was edited, say). Appended to the next
    /// command result, then cleared.
    pub notices: Vec<String>,
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
            notices: Vec::new(),
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
            notices: Vec::new(),
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
pub(crate) fn dimension_rejection_hint(sketch: &Sketch, action: &Action) -> String {
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
pub(crate) fn dim_endpoint_pos_from_sketch(sketch: &Sketch, ep: &DimensionEndpoint) -> vect2d {
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
pub(crate) fn prune_selection(ctx: &mut CommandContext) {
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
        Selection::Meta(mid) => sketch.meta_index(mid).is_some(),
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
pub(crate) fn blocker_hint_for_rejection(sketch: &mut Sketch, pre_snap: &[u8]) -> (String, Vec<String>) {
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
pub(crate) fn build_cid_name_map(
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
pub(crate) fn build_cid_display_map(
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

pub(crate) fn format_blocker_report(
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

        let created = if action.is_constraint_action() {
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
        };
        let notices = self.sketch.mutate_values(|s| s.take_notices());
        self.notices.extend(notices);
        created
    }

    /// Apply a split plan through the action machinery, returning the
    /// outcome report `exec` cannot carry. Mirrors the non-constraint
    /// `exec` path: apply, name, solve, dedup, record in history --
    /// the recorded `Action::SplitEntity` replays the same plan.
    pub fn exec_split(&mut self, plan: crate::split::SplitPlan)
        -> Result<crate::split::SplitOutcome, String>
    {
        self.status_error = None;
        self.status_blocker_names = None;
        let outcome = {
            let s = self.sketch.get_mut();
            let o = crate::split::apply_split(s, &plan)?;
            s.assign_constraint_names();
            s.solve();
            s.update_expr_dim_values();
            o
        };
        self.sketch.get_mut().dedup_constraints();
        self.history.push(
            Action::SplitEntity { plan },
            &self.sketch,
            CursorState { pos: self.cursor, tangent: self.cursor_tangent },
        );
        let notices = self.sketch.mutate_values(|s| s.take_notices());
        self.notices.extend(notices);
        Ok(outcome)
    }
}

pub struct CommandResult {
    pub output: String,
    pub is_error: bool,
    pub no_echo: bool,
    pub markdown: bool,
}

/// Handler result: `Ok` carries the success output, `Err` the error
/// message that `execute_one` turns into an error `CommandResult`.
pub(crate) type CmdResult = Result<CommandResult, String>;

pub(crate) fn ok(msg: impl Into<String>) -> CommandResult {
    CommandResult { output: msg.into(), is_error: false, no_echo: false, markdown: false }
}

/// Return ok or the status_error if the last exec was rejected.
/// Also resets skip_dof_check to false.
pub(crate) fn ok_or_status(ctx: &mut CommandContext, msg: impl Into<String>) -> CommandResult {
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
pub(crate) fn applied_msg(sketch: &Sketch, name: &str, fallback: &str) -> String {
    match sketch.find_constraint_description(name) {
        Some(desc) => format!("{}: {}", name, desc),
        None => fallback.to_string(),
    }
}

pub(crate) fn err(msg: impl Into<String>) -> CommandResult {
    CommandResult { output: msg.into(), is_error: true, no_echo: false, markdown: false }
}

/// Name of the last dimension in the sketch -- the one just added by
/// the caller's `Action::AddDimension`. Used by the dim-creation
/// commands so their success messages surface the newly-minted
/// `d<N>` identifier.
pub(crate) fn last_dim_name(ctx: &CommandContext) -> String {
    ctx.sketch.dimensions.last().map(|d| d.name.clone()).unwrap_or_default()
}

/// Peel recognized trailing keywords off the token list, any order,
/// any count. Returns one flag per `keys` entry, in order.
pub(crate) fn peel_keywords<const N: usize>(tokens: &mut Vec<&str>, keys: [&str; N]) -> [bool; N] {
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
pub(crate) fn driven_dim_fragment(ctx: &mut CommandContext, action: Action, label: &str, value: f64) -> String {
    ctx.exec(action);
    match ctx.status_error.take() {
        Some(e) => format!(" [driven {} rejected: {}]", label, e),
        None => format!(" [driven {} {}={:.4}]", last_dim_name(ctx), label, value),
    }
}

// ---------------------------------------------------------------------------
// Entity resolution
// ---------------------------------------------------------------------------

