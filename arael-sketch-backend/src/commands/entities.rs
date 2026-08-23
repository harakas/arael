use super::*;

pub(crate) const SNAP_THRESHOLD: f64 = 1e-3;

pub(crate) fn snap_near(a: vect2d, b: vect2d) -> bool {
    (a.x - b.x).abs() < SNAP_THRESHOLD && (a.y - b.y).abs() < SNAP_THRESHOLD
}


/// The (own slot, target) of an auto-connect candidate action.
/// `own_is_line`: in auto_coincident_line the line side of the shared
/// line-arc variants is ours; in auto_coincident_arc the arc side is.
fn connect_slots(a: &Action, own_is_line: bool) -> Option<(u8, Selection)> {
    use Action::*;
    Some(match *a {
        ApplyCoincidentLL11 { b, .. } => (0, Selection::LineP1(b)),
        ApplyCoincidentLL12 { b, .. } => (0, Selection::LineP2(b)),
        ApplyCoincidentLL21 { b, .. } => (1, Selection::LineP1(b)),
        ApplyCoincidentLL22 { b, .. } => (1, Selection::LineP2(b)),
        ApplyCoincidentLP1 { point, .. } => (0, Selection::Point(point)),
        ApplyCoincidentLP2 { point, .. } => (1, Selection::Point(point)),
        ApplyCoincidentLP1ArcCenter { line, arc } =>
            if own_is_line { (0, Selection::ArcCenter(arc)) } else { (0, Selection::LineP1(line)) },
        ApplyCoincidentLP2ArcCenter { line, arc } =>
            if own_is_line { (1, Selection::ArcCenter(arc)) } else { (0, Selection::LineP2(line)) },
        ApplyCoincidentLP1ArcStart { line, arc } =>
            if own_is_line { (0, Selection::ArcStart(arc)) } else { (1, Selection::LineP1(line)) },
        ApplyCoincidentLP2ArcStart { line, arc } =>
            if own_is_line { (1, Selection::ArcStart(arc)) } else { (1, Selection::LineP2(line)) },
        ApplyCoincidentLP1ArcEnd { line, arc } =>
            if own_is_line { (0, Selection::ArcEnd(arc)) } else { (2, Selection::LineP1(line)) },
        ApplyCoincidentLP2ArcEnd { line, arc } =>
            if own_is_line { (1, Selection::ArcEnd(arc)) } else { (2, Selection::LineP2(line)) },
        ApplyConcentric { b, .. } => (0, Selection::ArcCenter(b)),
        ApplyCoincidentArcCenterStart { b, .. } => (0, Selection::ArcStart(b)),
        ApplyCoincidentArcCenterEnd { b, .. } => (0, Selection::ArcEnd(b)),
        ApplyCoincidentArcStartCenter { b, .. } => (1, Selection::ArcCenter(b)),
        ApplyCoincidentArcStartStart { b, .. } => (1, Selection::ArcStart(b)),
        ApplyCoincidentArcStartEnd { b, .. } => (1, Selection::ArcEnd(b)),
        ApplyCoincidentArcEndCenter { b, .. } => (2, Selection::ArcCenter(b)),
        ApplyCoincidentArcEndStart { b, .. } => (2, Selection::ArcStart(b)),
        ApplyCoincidentArcEndEnd { b, .. } => (2, Selection::ArcEnd(b)),
        ApplyCoincidentArcCenter { point, .. } => (0, Selection::Point(point)),
        ApplyCoincidentArcStart { point, .. } => (1, Selection::Point(point)),
        ApplyCoincidentArcEnd { point, .. } => (2, Selection::Point(point)),
        _ => return None,
    })
}

/// Keep one auto-connect per (own slot, target cluster): the members
/// of an already-coincident cluster are one point, so a second
/// connect to the same cluster would be a redundant constraint (and
/// would end the incremental DOF window for nothing).
fn dedup_connects(
    sketch: &Sketch,
    own_is_line: bool,
    actions: Vec<(Action, String)>,
) -> Vec<(Action, String)> {
    if actions.len() < 2 {
        return actions;
    }
    let mut groups = crate::coincide::CoincidenceGroups::build(sketch);
    let mut claimed: std::collections::HashSet<(u8, usize)> = std::collections::HashSet::new();
    actions
        .into_iter()
        .filter(|(a, _)| match connect_slots(a, own_is_line) {
            Some((side, target)) => match groups.selection_id(target) {
                Some(id) => {
                    let root = groups.find(id);
                    claimed.insert((side, root))
                }
                None => true,
            },
            None => true,
        })
        .collect()
}

/// Auto-connect endpoints of the last created line to nearby existing endpoints.
pub(crate) fn auto_coincident_line(ctx: &mut CommandContext, line_ref: Ref<Line>) -> Vec<String> {
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
    let actions = dedup_connects(&ctx.sketch, true, actions);
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
pub(crate) fn auto_coincident_arc(ctx: &mut CommandContext, arc_ref: Ref<Arc>, center_only: bool) -> Vec<String> {
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

    let actions = dedup_connects(&ctx.sketch, false, actions);
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
pub(crate) fn pop_tangent(sketch: &mut Sketch, action: &Action) {
    match action {
        Action::ApplyTangentLA { .. } => { sketch.tangent_la.pop(); }
        Action::ApplyTangentAA { .. } => { sketch.tangent_aa.pop(); }
        _ => {}
    }
}

/// Check if two direction vectors are nearly parallel (within ~1 degree).
pub(crate) fn nearly_tangent(d1: vect2d, d2: vect2d) -> bool {
    let len1 = (d1.x * d1.x + d1.y * d1.y).sqrt();
    let len2 = (d2.x * d2.x + d2.y * d2.y).sqrt();
    if len1 < 1e-12 || len2 < 1e-12 { return false; }
    let cross = (d1.x * d2.y - d1.y * d2.x).abs() / (len1 * len2);
    cross < 0.018 // sin(1 deg) ~ 0.01745
}

/// Try to apply auto-tangent constraints for a newly created line.
/// Checks if the line's endpoints are coincident with arc endpoints
/// and the geometry is already tangent.
pub(crate) fn auto_tangent_line(ctx: &mut CommandContext, line_ref: Ref<Line>) -> Vec<String> {
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
pub(crate) fn auto_tangent_arc(ctx: &mut CommandContext, arc_ref: Ref<Arc>) -> Vec<String> {
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

pub(crate) fn cmd_add_line(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, notangent, driven, quiet, constr] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "notangent", "driven", "quiet", "constr"]);

    // Parse all coordinate tokens
    let points: Vec<vect2d> = if tokens.len() >= 2 {
        let mut pts = Vec::new();
        let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
        pts.push(p1);
        for i in 1..tokens.len() {
            let prev = *pts.last().unwrap();
            let p = parse_coord(ctx, tokens[i], Some(prev))?;
            pts.push(p);
        }
        pts
    } else if tokens.len() == 1 {
        let prev = match ctx.cursor {
            Some(p) => p,
            None => return Err("No previous point. Use: add_line x1,y1 x2,y2".into()),
        };
        let p2 = parse_coord(ctx, tokens[0], Some(prev))?;
        vec![prev, p2]
    } else {
        return Err("Usage: add_line x1,y1 x2,y2 [x3,y3 ...] [noconnect] [notangent] [nocursor] [driven]".into());
    };

    ctx.begin_group();
    let mut msgs = Vec::new();
    let n_segments = points.len() - 1;
    for i in 0..n_segments {
        let p1 = points[i];
        let p2 = points[i + 1];
        let Some(line_ref) = ctx.exec(Action::AddLine { p1, p2 }).line() else {
            return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msgs.join("\n")))
}

/// Set cursor_tangent from an arc's end angle.
pub(crate) fn set_cursor_tangent_from_arc(ctx: &mut CommandContext, arc_ref: arael::refs::Ref<arael_sketch_solver::Arc>) {
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
pub(crate) fn constraint_names_since(ctx: &CommandContext, watermark: u32) -> Vec<String> {
    let mut nids: Vec<u32> = ctx.sketch.constraint_nid_cid_pairs().into_iter()
        .map(|(nid, _)| nid).filter(|&n| n >= watermark).collect();
    nids.sort_unstable();
    nids.dedup();
    nids.into_iter().map(|n| format!("C{}", n)).collect()
}

pub(crate) fn rect_exec(ctx: &mut CommandContext, action: Action, strict: bool, desc: &str, applied: &mut Vec<String>, warnings: &mut Vec<String>) -> Result<(), String> {
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
pub(crate) fn build_rect(
    ctx: &mut CommandContext,
    corners: [vect2d; 4],
    noconnect: bool,
    noconstraint: bool,
    hv: bool,
    driven: bool,
    strict: bool,
) -> CmdResult {
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
            return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
                return Err(e.into());
            }
            let desc = format!("vertical {} {}", line_names[1], line_names[3]);
            if let Err(e) = rect_exec(ctx, Action::ApplyVertical { lines: vec![line_refs[1], line_refs[3]] }, strict, &desc, &mut applied, &mut warnings) {
                return Err(e.into());
            }
        } else {
            let desc = format!("perpendicular {} {}", line_names[0], line_names[1]);
            if let Err(e) = rect_exec(ctx, Action::ApplyPerpendicular { a: line_refs[0], b: line_refs[1] }, strict, &desc, &mut applied, &mut warnings) {
                return Err(e.into());
            }
            let desc = format!("parallel {} {}", line_names[0], line_names[2]);
            if let Err(e) = rect_exec(ctx, Action::ApplyParallel { a: line_refs[0], b: line_refs[2] }, strict, &desc, &mut applied, &mut warnings) {
                return Err(e.into());
            }
            let desc = format!("parallel {} {}", line_names[1], line_names[3]);
            if let Err(e) = rect_exec(ctx, Action::ApplyParallel { a: line_refs[1], b: line_refs[3] }, strict, &desc, &mut applied, &mut warnings) {
                return Err(e.into());
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
                return Err(e.into());
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
    Ok(ok(msg))
}

pub(crate) struct RectKeywords {
    noconnect: bool,
    noconstraint: bool,
    hv: bool,
    driven: bool,
    strict: bool,
}

/// Parse trailing rect keywords. Returns error string on conflict.
pub(crate) fn parse_rect_keywords(tokens: &mut Vec<&str>, allow_hv: bool) -> Result<RectKeywords, String> {
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

pub(crate) fn cmd_add_rect(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let kw = parse_rect_keywords(&mut tokens, true)?;
    if tokens.len() != 2 {
        return Err("Usage: add_rect x1,y1 x2,y2 [hv] [noconnect] [noconstraint] [driven] [strict]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], Some(p1))?;
    let bl = p1;
    let br = vect2d::new(p2.x, p1.y);
    let tr = p2;
    let tl = vect2d::new(p1.x, p2.y);
    build_rect(ctx, [bl, br, tr, tl], kw.noconnect, kw.noconstraint, kw.hv, kw.driven, kw.strict)
}

pub(crate) fn cmd_add_rect3(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let kw = parse_rect_keywords(&mut tokens, false)?;
    if tokens.len() != 3 {
        return Err("Usage: add_rect3 p1 p2 p3 [noconnect] [noconstraint] [driven] [strict]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], Some(p1))?;
    let p3 = parse_coord(ctx, tokens[2], Some(p2))?;
    // Reject collinear points (cross product of p1->p2 and p2->p3 ~ 0)
    let cross = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);
    if cross.abs() < 1e-9 {
        return Err("Points are collinear, cannot form a rectangle".into());
    }
    // p4 = p1 + (p3 - p2)
    let p4 = vect2d::new(p1.x + (p3.x - p2.x), p1.y + (p3.y - p2.y));
    build_rect(ctx, [p1, p2, p3, p4], kw.noconnect, kw.noconstraint, kw.hv, kw.driven, kw.strict)
}

pub(crate) fn cmd_add_rectcenter(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let kw = parse_rect_keywords(&mut tokens, true)?;
    if tokens.len() != 2 {
        return Err("Usage: add_rectcenter cx,cy px,py [hv] [noconnect] [noconstraint] [driven] [strict]".into());
    }
    let center = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let corner = parse_coord(ctx, tokens[1], Some(center))?;
    // Axis-aligned: bl=corner, tr=opposite corner (reflected through center)
    let bl = corner;
    let tr = vect2d::new(2.0 * center.x - corner.x, 2.0 * center.y - corner.y);
    let br = vect2d::new(tr.x, bl.y);
    let tl = vect2d::new(bl.x, tr.y);
    build_rect(ctx, [bl, br, tr, tl], kw.noconnect, kw.noconstraint, kw.hv, kw.driven, kw.strict)
}

pub(crate) fn cmd_add_point(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let nocursor = tokens.last() == Some(&"nocursor");
    if nocursor { tokens.pop(); }
    if tokens.len() != 1 { return Err("Usage: add_point x,y [nocursor]".into()); }
    let pos = parse_coord(ctx, tokens[0], ctx.cursor)?;
    ctx.begin_group();
    ctx.exec(Action::AddPoint { pos });
    let name = ctx.sketch.points.refs().last().map(|r| ctx.sketch.points[r].name.clone()).unwrap_or_default();
    if !nocursor { ctx.cursor = Some(pos); }
    ctx.session_names.insert("_".into(), name.clone());
    Ok(ok(format!("Added {}: ({:.2},{:.2})", name, pos.x, pos.y)))
}

pub(crate) fn cmd_add_circle(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 2 {
        return Err("Usage: add_circle cx,cy radius [noconnect] [nocursor] [driven]".into());
    }
    let center = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let r = eval_expr(&ctx.sketch, tokens[1])?;
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_circle2(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 2 {
        return Err("Usage: add_circle2 p1 p2 [noconnect] [nocursor] [driven]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], Some(p1))?;
    let center = vect2d::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
    let r = ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt() / 2.0;
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_circle3(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 3 {
        return Err("Usage: add_circle3 p1 p2 p3 [noconnect] [nocursor] [driven]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], Some(p1))?;
    let p3 = parse_coord(ctx, tokens[2], Some(p2))?;
    let (center, r, _, _, _) = match crate::geometry::circumscribed_arc(p1, p2, p3) {
        Some(v) => v,
        None => return Err("Points are collinear, cannot define a circle".into()),
    };
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_ellipse(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "driven"]);
    if tokens.len() != 4 {
        return Err("Usage: add_ellipse cx,cy rx ry rotation [noconnect] [nocursor] [driven]".into());
    }
    let center = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let rx = eval_expr(&ctx.sketch, tokens[1])?;
    let ry = eval_expr(&ctx.sketch, tokens[2])?;
    let rot = eval_expr(&ctx.sketch, tokens[3])?;
    let rot_rad = arael::utils::deg2rad(rot);
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipse { center, rx, ry, rotation: rot_rad }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_earc(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven, large, cw] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven", "large", "cw"]);
    if tokens.len() != 5 {
        return Err("Usage: add_earc p1 p2 rx ry rot_deg [large] [cw] [noconnect] [notangent] [nocursor] [driven]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], ctx.cursor)?;
    let rx = eval_expr(&ctx.sketch, tokens[2])?;
    let ry = eval_expr(&ctx.sketch, tokens[3])?;
    let rot_deg = eval_expr(&ctx.sketch, tokens[4])?;
    let rot = arael::utils::deg2rad(rot_deg);
    let sweep = !cw; // SVG sweep-flag: true = CCW
    let result = crate::geometry::svg_arc_to_center(p1, p2, rx, ry, rot, large, sweep);
    let (center, sa, ea, rx, ry) = match result {
        Some(v) => v,
        None => return Err("Cannot compute elliptic arc (degenerate or zero radii)".into()),
    };
    let ccw = !cw;
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start: sa, end: ea, ccw }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_earc3(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven"]);
    if tokens.len() != 5 {
        return Err("Usage: add_earc3 p1 p2 pmid rx ry [noconnect] [notangent] [nocursor] [driven]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], ctx.cursor)?;
    let pmid = parse_coord(ctx, tokens[2], ctx.cursor)?;
    let rx = eval_expr(&ctx.sketch, tokens[3])?;
    let ry = eval_expr(&ctx.sketch, tokens[4])?;
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
                let mid_pt = ellipse_point(center, rx_out, ry_out, rot, mid_angle);
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
        None => return Err("Cannot compute elliptic arc from given points and radii".into()),
    };
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start: sa, end: ea, ccw }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_earc_center(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven, cw] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven", "cw"]);
    if tokens.len() != 6 {
        return Err("Usage: add_earc_center cx,cy rx ry rot_deg start_deg end_deg [cw] [noconnect] [notangent] [nocursor] [driven]".into());
    }
    let center = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let rx = eval_expr(&ctx.sketch, tokens[1])?;
    let ry = eval_expr(&ctx.sketch, tokens[2])?;
    let rot_deg = eval_expr(&ctx.sketch, tokens[3])?;
    let start_deg = eval_expr(&ctx.sketch, tokens[4])?;
    let end_deg = eval_expr(&ctx.sketch, tokens[5])?;
    let rot = arael::utils::deg2rad(rot_deg);
    let start = arael::utils::deg2rad(start_deg);
    let end = arael::utils::deg2rad(end_deg);
    let ccw = !cw;
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start, end, ccw }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_earc_tangent(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, notangent, driven, quiet, constr] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "notangent", "driven", "quiet", "constr"]);
    // Syntax: add_earc_tangent p1 t1 p2 t2 [bulge]
    if tokens.len() < 4 || tokens.len() > 5 {
        return Err("Usage: add_earc_tangent p1 t1 p2 t2 [bulge] [noconnect] [notangent] [nocursor] [quiet] [driven]".into());
    }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let t1 = parse_coord(ctx, tokens[1], None)?;
    let p2 = parse_coord(ctx, tokens[2], Some(p1))?;
    let t2 = parse_coord(ctx, tokens[3], None)?;
    let w = if tokens.len() == 5 {
        eval_expr(&ctx.sketch, tokens[4])?
    } else { 1.0 };

    let result = crate::earc_fit::fit_earc_tangent(p1, t1, p2, t2, w);
    let (center, rx, ry, rot, sa, ea, ccw) = match result {
        Some(v) => v,
        None => return Err("Cannot fit elliptic arc (degenerate tangent configuration)".into()),
    };
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddEllipticArc { center, rx, ry, rotation: rot, start: sa, end: ea, ccw }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
    Ok(ok(msg))
}

pub(crate) fn cmd_add_earc_rtangent(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let p1 = match ctx.cursor {
        Some(p) => p,
        None => return Err("No cursor position (add a line or arc first)".into()),
    };
    let t1 = match ctx.cursor_tangent {
        Some(t) => t,
        None => return Err("No tangent direction (add a line or arc first)".into()),
    };
    // Prepend p1 and t1 as coordinate strings and delegate
    let p1_str = format!("{},{}", p1.x, p1.y);
    let t1_str = format!("{},{}", t1.x, t1.y);
    let new_args = format!("{} {} {}", p1_str, t1_str, args);
    cmd_add_earc_tangent(ctx, &new_args)
}

/// Check if the perpendicular projection of `center` onto the line segment (p1,p2)
/// falls within the segment (0 <= t <= 1).
pub(crate) fn tangent_touches_segment(center: vect2d, p1: vect2d, p2: vect2d) -> bool {
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
pub(crate) fn circle_tangent_2lines(sketch: &Sketch, la: Ref<Line>, lb: Ref<Line>, radius: f64) -> Result<vect2d, String> {
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
pub(crate) fn circle_tangent_3lines(sketch: &Sketch, la: Ref<Line>, lb: Ref<Line>, lc: Ref<Line>) -> Result<(vect2d, f64), String> {
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
pub(crate) fn line_line_intersect(p1: vect2d, d1: vect2d, p2: vect2d, d2: vect2d) -> Option<vect2d> {
    let cross = d1.x * d2.y - d1.y * d2.x;
    if cross.abs() < 1e-12 { return None; }
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let t = (dx * d2.y - dy * d2.x) / cross;
    Some(vect2d::new(p1.x + t * d1.x, p1.y + t * d1.y))
}

pub(crate) fn cmd_add_circle2t(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [noconnect, quiet, constr, noconstraint, driven, strict] = peel_keywords(&mut tokens,
        ["noconnect", "quiet", "constr", "noconstraint", "driven", "strict"]);
    if noconstraint && (driven || strict) {
        return Err("noconstraint conflicts with driven and strict".into());
    }
    if tokens.len() != 3 {
        return Err("Usage: add_circle2t L0 L1 radius [noconnect] [noconstraint] [driven] [strict]".into());
    }
    let la = resolve_line(&ctx.sketch, tokens[0])?;
    let lb = resolve_line(&ctx.sketch, tokens[1])?;
    let r = eval_expr(&ctx.sketch, tokens[2])?;
    let center = circle_tangent_2lines(&ctx.sketch, la, lb, r)?;
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let mut warnings = Vec::new();
    let mut applied = Vec::new();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
            return Err(e.into());
        }
        let desc = format!("tangent {} {}", tokens[1], name);
        if let Err(e) = rect_exec(ctx, Action::ApplyTangentLA { line: lb, arc: arc_ref }, strict, &desc, &mut applied, &mut warnings) {
            return Err(e.into());
        }
    }
    if driven {
        let desc = format!("driven radius {} = {:.4}", name, r);
        if let Err(e) = rect_exec(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, strict, &desc, &mut applied, &mut warnings) {
            return Err(e.into());
        }
    }
    for a in &applied {
        msg += &format!("\n  {}", a);
    }
    for w in &warnings {
        msg += &format!("\n  warning: {}", w);
    }
    Ok(ok(msg))
}

pub(crate) fn cmd_add_circle3t(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [noconnect, quiet, constr, noconstraint, driven, strict] = peel_keywords(&mut tokens,
        ["noconnect", "quiet", "constr", "noconstraint", "driven", "strict"]);
    if noconstraint && (driven || strict) {
        return Err("noconstraint conflicts with driven and strict".into());
    }
    if tokens.len() != 3 {
        return Err("Usage: add_circle3t L0 L1 L2 [noconnect] [noconstraint] [driven] [strict]".into());
    }
    let la = resolve_line(&ctx.sketch, tokens[0])?;
    let lb = resolve_line(&ctx.sketch, tokens[1])?;
    let lc = resolve_line(&ctx.sketch, tokens[2])?;
    let (center, r) = circle_tangent_3lines(&ctx.sketch, la, lb, lc)?;
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    let mut warnings = Vec::new();
    let mut applied_list = Vec::new();
    let Some(arc_ref) = ctx.exec(Action::AddCircle { center, edge }).arc() else {
        return Err(ctx.status_error.take().unwrap_or_else(|| "Internal: creation action added no entity".into()).into());
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
                return Err(e.into());
            }
        }
    }
    if driven {
        let desc = format!("driven radius {} = {:.4}", name, r);
        if let Err(e) = rect_exec(ctx, Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: r, expr: None, derived: false, range: None,
        }, strict, &desc, &mut applied_list, &mut warnings) {
            return Err(e.into());
        }
    }
    for a in &applied_list {
        msg += &format!("\n  {}", a);
    }
    for w in &warnings {
        msg += &format!("\n  warning: {}", w);
    }
    Ok(ok(msg))
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
pub(crate) fn cmd_delete(ctx: &mut CommandContext, args: &str) -> CmdResult {
    use crate::ids::find_constraint_by_name;
    let cleaned = args.trim();
    if cleaned.is_empty() {
        return Err("Usage: delete L0 | delete P0 | delete A0 | delete C3 | delete CL0H | delete d0 | delete L0 L1 parallel".into());
    }
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();

    // The whole selection as one batch, like the GUI's Backspace.
    if tokens.len() == 1 && tokens[0] == "selection" {
        return delete_selection(ctx);
    }

    // Meta-constraint: `delete M0` dissolves it (the result stays),
    // `delete M0 all` deletes the result too.
    if tokens[0].starts_with('M') && let Some(i) = ctx.sketch.find_meta(tokens[0]) {
        let name = ctx.sketch.metas[i].name.clone();
        let mid = ctx.sketch.metas[i].mid;
        return match tokens.get(1) {
            None => {
                crate::meta::dissolve(ctx, mid)?;
                Ok(ok_or_status(ctx, format!("Dissolved {} (its geometry stays)", name)))
            }
            Some(&"all") => {
                let names = crate::meta::delete_with_result(ctx, mid)?;
                Ok(ok_or_status(ctx, format!("Deleted {} and {}", name, names.join(", "))))
            }
            Some(other) => Err(format!("delete {}: unknown option '{}' (use 'all' to delete the result too)", name, other)),
        };
    }

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
        return Ok(ok(format!("Deleted {}", desc)));
    }

    // 2) Dimension: d<n>.
    if let Some(idx) = ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        ctx.begin_group();
        ctx.exec(Action::RemoveDimension { did: ctx.sketch.dimensions[idx].did });
        return Ok(ok(format!("Deleted dimension {}", name)));
    }

    // 3) Entity: L0, P0, A0, EA0.
    let action = if name.starts_with('L') && !name.starts_with("LineAngle") {
        // Resolve as line.
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => Some(("line", Action::DeleteLine { line: r })),
            Err(e) => return Err(e),
        }
    } else if name.starts_with('P') {
        match resolve_point(&ctx.sketch, name) {
            Ok(r) => Some(("point", Action::DeletePoint { point: r })),
            Err(e) => return Err(e),
        }
    } else if is_arc_name(name) {
        match resolve_arc(&ctx.sketch, name) {
            Ok(r) => Some(("arc", Action::DeleteArc { arc: r })),
            Err(e) => return Err(e),
        }
    } else {
        None
    };

    let Some((kind, action)) = action else {
        return Err(format!("Unknown name: {}", name).into());
    };

    // Snapshot both views of the current state so we can report
    // what got cascade-removed: list_constraints() covers the
    // constraints addressable by their own name, the dims listing
    // covers d<n>-named dimensions (each with its own constraint).
    let before_constraints: std::collections::BTreeSet<String> =
        ctx.sketch.list_constraints().into_iter().collect();
    let before_dims: Vec<(String, String)> = ctx.sketch.dimensions.iter()
        .map(|d| (d.name.clone(), dim_line(&ctx.sketch, d)))
        .collect();

    ctx.begin_group();
    ctx.exec(action);

    let after_constraints: std::collections::BTreeSet<String> =
        ctx.sketch.list_constraints().into_iter().collect();
    let after_dim_names: std::collections::BTreeSet<String> =
        ctx.sketch.dimensions.iter().map(|d| d.name.clone()).collect();

    let removed_constraints: Vec<String> = before_constraints
        .difference(&after_constraints).cloned().collect();
    let removed_dims: Vec<&String> = before_dims.iter()
        .filter(|(n, _)| !after_dim_names.contains(n))
        .map(|(_, line)| line)
        .collect();

    let mut msg = format!("Deleted {} {}", kind, name);
    if !removed_constraints.is_empty() || !removed_dims.is_empty() {
        msg.push_str("\n  cascade:");
        for c in &removed_constraints {
            msg.push_str(&format!("\n    {}", c));
        }
        for line in &removed_dims {
            msg.push_str(&format!("\n    {}", line));
        }
    }
    Ok(ok(msg))
}

/// `delete selection`: every selected entity, constraint and dimension
/// in one batch (one history entry); selected meta-constraints
/// dissolve. Reports what was deleted and what cascaded away.
fn delete_selection(ctx: &mut CommandContext) -> CmdResult {
    use crate::ids::constraint_id_name;
    let mut direct: Vec<String> = Vec::new();
    let mut dissolved: Vec<String> = Vec::new();
    for sel in &ctx.selection {
        match *sel {
            Selection::Line(r) => { if let Some(l) = ctx.sketch.lines.get(r) { direct.push(l.name.clone()); } }
            Selection::Point(r) => { if let Some(p) = ctx.sketch.points.get(r) { direct.push(p.name.clone()); } }
            Selection::Arc(r) => { if let Some(a) = ctx.sketch.arcs.get(r) { direct.push(a.name.clone()); } }
            Selection::Constraint(id) => { if let Some(n) = constraint_id_name(&ctx.sketch, id) { direct.push(n); } }
            Selection::Dimension(did) => {
                if let Some(i) = ctx.sketch.dimension_index_by_did(did) {
                    direct.push(ctx.sketch.dimensions[i].name.clone());
                }
            }
            Selection::Meta(mid) => {
                if let Some(i) = ctx.sketch.meta_index(mid) {
                    dissolved.push(ctx.sketch.metas[i].name.clone());
                }
            }
            _ => {} // endpoints aren't deletable on their own
        }
    }
    let acts = crate::actions::delete_selection_actions(&ctx.selection);
    if acts.is_empty() {
        return Err("Nothing deletable selected (endpoints cannot be deleted on their own)".into());
    }

    let before_constraints: std::collections::BTreeSet<String> =
        ctx.sketch.list_constraints().into_iter().collect();
    let before_dims: Vec<(String, String)> = ctx.sketch.dimensions.iter()
        .map(|d| (d.name.clone(), dim_line(&ctx.sketch, d)))
        .collect();

    ctx.begin_group();
    ctx.exec(Action::Batch { label: "Delete selection".into(), actions: acts });
    if let Some(e) = ctx.status_error.take() {
        return Err(e);
    }
    ctx.selection.clear();

    let after_constraints: std::collections::BTreeSet<String> =
        ctx.sketch.list_constraints().into_iter().collect();
    let after_dim_names: std::collections::BTreeSet<String> =
        ctx.sketch.dimensions.iter().map(|d| d.name.clone()).collect();
    let direct_set: std::collections::BTreeSet<&str> =
        direct.iter().map(|n| n.as_str()).collect();
    let cascade_constraints: Vec<&String> = before_constraints
        .difference(&after_constraints)
        .filter(|l| l.split(':').next().is_none_or(|n| !direct_set.contains(n)))
        .collect();
    let cascade_dims: Vec<&String> = before_dims.iter()
        .filter(|(n, _)| !after_dim_names.contains(n) && !direct_set.contains(n.as_str()))
        .map(|(_, line)| line)
        .collect();

    let mut parts: Vec<String> = Vec::new();
    if !direct.is_empty() {
        parts.push(format!("Deleted {}", direct.join(" ")));
    }
    if !dissolved.is_empty() {
        parts.push(format!("Dissolved {} (the geometry stays)", dissolved.join(" ")));
    }
    let mut msg = parts.join("\n");
    if !cascade_constraints.is_empty() || !cascade_dims.is_empty() {
        msg.push_str("\n  cascade:");
        for c in &cascade_constraints {
            msg.push_str(&format!("\n    {}", c));
        }
        for line in &cascade_dims {
            msg.push_str(&format!("\n    {}", line));
        }
    }
    Ok(ok_or_status(ctx, msg))
}


// ---------------------------------------------------------------------------
// Scale
// ---------------------------------------------------------------------------

pub(crate) fn cmd_scale(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    let usage = "Usage: scale <entity|selection>... about <center> <factor>";
    let about_pos = match tokens.iter().position(|&t| t == "about") {
        Some(p) => p,
        None => return Err(usage.into()),
    };
    if about_pos == 0 {
        return Err("No entities to scale".into());
    }
    let after = &tokens[about_pos + 1..];
    if after.len() != 2 {
        return Err(usage.into());
    }
    // Center: a point-like endpoint name or a coordinate.
    let center = resolve_endpoint_pos(&ctx.sketch, after[0])
        .or_else(|_| parse_coord(ctx, after[0], None))
        .map_err(|e| format!("scale center '{}': {}", after[0], e))?;
    let factor = eval_expr(&ctx.sketch, after[1])?;
    if !factor.is_finite() || factor <= 0.0 {
        return Err(format!("Scale factor must be positive, got {}", after[1]));
    }

    // Entities: explicit names or the current selection.
    let mut lines: Vec<Ref<Line>> = Vec::new();
    let mut arcs: Vec<Ref<Arc>> = Vec::new();
    let mut points: Vec<Ref<Point>> = Vec::new();
    let source_tokens = &tokens[..about_pos];
    if source_tokens.len() == 1 && source_tokens[0] == "selection" {
        for sel in &ctx.selection {
            match sel {
                Selection::Line(r) => if !lines.contains(r) { lines.push(*r); },
                Selection::Arc(r) => if !arcs.contains(r) { arcs.push(*r); },
                Selection::Point(r) => if !points.contains(r) { points.push(*r); },
                _ => {} // endpoint/constraint/dimension selections skipped
            }
        }
        if lines.is_empty() && arcs.is_empty() && points.is_empty() {
            return Err("No lines, arcs, or points in selection".into());
        }
    } else {
        for &name in source_tokens {
            if name.starts_with('P') {
                let r = resolve_point(&ctx.sketch, name)?;
                if !points.contains(&r) { points.push(r); }
            } else if is_arc_name(name) {
                let r = resolve_arc(&ctx.sketch, name)?;
                if !arcs.contains(&r) { arcs.push(r); }
            } else {
                let r = resolve_line(&ctx.sketch, name)?;
                if !lines.contains(&r) { lines.push(r); }
            }
        }
    }

    // Dimension classification, for the report (the action applies
    // the same classification).
    let (_, report) = crate::scale::classify_scale_dims(&ctx.sketch, &lines, &arcs, &points);
    let mut names: Vec<String> = Vec::new();
    for r in &lines { names.push(ctx.sketch.lines[*r].name.clone()); }
    for r in &arcs { names.push(ctx.sketch.arcs[*r].name.clone()); }
    for r in &points { names.push(ctx.sketch.points[*r].name.clone()); }

    ctx.begin_group();
    ctx.exec(Action::Scale { lines, arcs, points, center, factor });
    if let Some(e) = ctx.status_error.take() {
        return Err(e);
    }
    let mut msg = format!(
        "Scaled {} about ({:.3},{:.3}) x{}",
        names.join(" "), center.x, center.y, factor
    );
    if !report.scaled.is_empty() {
        msg += &format!("\n  dims scaled: {}", report.scaled.join(" "));
    }
    if !report.left.is_empty() {
        let left: Vec<String> = report.left.iter()
            .map(|(n, why)| format!("{} ({})", n, why))
            .collect();
        msg += &format!("\n  dims left: {}", left.join("; "));
    }
    Ok(ok(msg))
}
