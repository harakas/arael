use super::*;

pub(crate) fn cmd_freeze(ctx: &mut CommandContext, args: &str) -> CmdResult {
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
                    Err(e) => return Err(e),
                }
            } else if is_arc_name(name) {
                match resolve_arc(&ctx.sketch, name) {
                    Ok(r) => arc_refs.push(r),
                    Err(e) => return Err(e),
                }
            } else {
                return Err(format!("freeze applies to lines and arcs: {}", name).into());
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
        Ok(ok("Nothing to freeze"))
    } else {
        Ok(ok(lines.join("\n")))
    }
}

/// Find the helper point associated with an arc endpoint ref (center/start/end).
/// Returns None for non-arc endpoints or if no helper point exists.
pub(crate) fn resolve_endpoint_as_point(sketch: &Sketch, ep: EndpointRef) -> Option<Ref<Point>> {
    match ep {
        EndpointRef::Point(p) => Some(p),
        EndpointRef::ArcCenter(arc) => sketch.coincident_arc_center.iter().find(|c| c.arc == arc).map(|c| c.point),
        EndpointRef::ArcStart(arc) => sketch.coincident_arc_start.iter().find(|c| c.arc == arc).map(|c| c.point),
        EndpointRef::ArcEnd(arc) => sketch.coincident_arc_end.iter().find(|c| c.arc == arc).map(|c| c.point),
        _ => None,
    }
}

pub(crate) fn find_coincident_id(sketch: &Sketch, a: EndpointRef, b: EndpointRef) -> Option<crate::ids::ConstraintId> {
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

pub(crate) fn find_point_on_line_id(sketch: &Sketch, ep: EndpointRef, line: Ref<Line>) -> Option<crate::ids::ConstraintId> {
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

pub(crate) fn find_point_on_arc_id(sketch: &Sketch, ep: EndpointRef, arc: Ref<Arc>) -> Option<crate::ids::ConstraintId> {
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

pub(crate) fn find_midpoint_id(sketch: &Sketch, ep: EndpointRef, target_name: &str) -> Option<crate::ids::ConstraintId> {
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
pub(crate) fn delete_relational(ctx: &mut CommandContext, args: &str) -> CmdResult {
    use crate::ids::ConstraintId;
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err("Usage: delete L0 horizontal | delete L0 L1 parallel".into());
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
            let r = resolve_line(sketch, tokens[0])?;
            if sketch.lines[r].constraints.horizontal { Some(ConstraintId::Horizontal(r)) } else { None }
        }
        "vertical" => {
            let r = resolve_line(sketch, tokens[0])?;
            if sketch.lines[r].constraints.vertical { Some(ConstraintId::Vertical(r)) } else { None }
        }
        "parallel" if tokens.len() >= 3 => {
            let a = resolve_line(sketch, tokens[0])?;
            let b = resolve_line(sketch, tokens[1])?;
            find_ab!(sketch.parallel, a, b).map(|i| ConstraintId::Numbered(sketch.parallel[i].nid))
        }
        "perpendicular" | "perp" if tokens.len() >= 3 => {
            let a = resolve_line(sketch, tokens[0])?;
            let b = resolve_line(sketch, tokens[1])?;
            find_ab!(sketch.perpendicular, a, b).map(|i| ConstraintId::Numbered(sketch.perpendicular[i].nid))
        }
        "equal" | "equal_length" if tokens.len() >= 3 => {
            let a = resolve_line(sketch, tokens[0])?;
            let b = resolve_line(sketch, tokens[1])?;
            find_ab!(sketch.equal_length, a, b).map(|i| ConstraintId::Numbered(sketch.equal_length[i].nid))
        }
        "collinear" if tokens.len() >= 3 => {
            let a = resolve_line(sketch, tokens[0])?;
            let b = resolve_line(sketch, tokens[1])?;
            find_ab!(sketch.collinear, a, b).map(|i| ConstraintId::Numbered(sketch.collinear[i].nid))
        }
        "tangent" if tokens.len() >= 3 => {
            if tokens[0].starts_with('L') && is_arc_name(tokens[1]) {
                let line = resolve_line(sketch, tokens[0])?;
                let arc = resolve_arc(sketch, tokens[1])?;
                sketch.tangent_la.iter().find(|c| c.line == line && c.arc == arc).map(|c| ConstraintId::Numbered(c.nid))
            } else if is_arc_name(tokens[0]) && is_arc_name(tokens[1]) {
                let a = resolve_arc(sketch, tokens[0])?;
                let b = resolve_arc(sketch, tokens[1])?;
                find_ab!(sketch.tangent_aa, a, b).map(|i| ConstraintId::Numbered(sketch.tangent_aa[i].nid))
            } else { None }
        }
        "concentric" if tokens.len() >= 3 => {
            let a = resolve_arc(sketch, tokens[0])?;
            let b = resolve_arc(sketch, tokens[1])?;
            find_ab!(sketch.concentric, a, b).map(|i| ConstraintId::Numbered(sketch.concentric[i].nid))
        }
        "on_normal" if tokens.len() >= 3 => {
            let placed = resolve_endpoint_ref(sketch, tokens[0])?;
            let reference = resolve_endpoint_ref(sketch, tokens[1])?;
            match (placed, reference) {
                (EndpointRef::LineP1(a) | EndpointRef::LineP2(a), EndpointRef::LineP1(b) | EndpointRef::LineP2(b)) => {
                    let (pe, re) = (matches!(placed, EndpointRef::LineP2(_)), matches!(reference, EndpointRef::LineP2(_)));
                    sketch.on_normal_ll.iter()
                        .find(|c| c.a == a && c.b == b && c.placed_end == pe && c.ref_end == re)
                        .map(|c| ConstraintId::Numbered(c.nid))
                }
                (EndpointRef::ArcStart(a) | EndpointRef::ArcEnd(a), EndpointRef::ArcStart(b) | EndpointRef::ArcEnd(b)) => {
                    let (pe, re) = (matches!(placed, EndpointRef::ArcEnd(_)), matches!(reference, EndpointRef::ArcEnd(_)));
                    sketch.on_normal_aa.iter()
                        .find(|c| c.a == a && c.b == b && c.placed_end == pe && c.ref_end == re)
                        .map(|c| ConstraintId::Numbered(c.nid))
                }
                _ => None,
            }
        }
        "lock" => {
            // Locks are entity flags, not a ConstraintId: route through
            // the unlock actions so undo restores them.
            let ep = resolve_endpoint_ref(sketch, tokens[0])?;
            let action = match ep {
                EndpointRef::Point(p) => Some(Action::UnlockPoint { point: p }),
                EndpointRef::LineP1(l) => Some(Action::UnlockLineP1 { line: l }),
                EndpointRef::LineP2(l) => Some(Action::UnlockLineP2 { line: l }),
                EndpointRef::ArcCenter(a) => Some(Action::UnlockArcCenter { arc: a }),
                _ => None,
            };
            let Some(action) = action else { return Err("Constraint not found".to_string().into()); };
            ctx.begin_group();
            ctx.exec(action);
            return Ok(ok(format!("Removed lock on {}", tokens[0])));
        }
        "equal_radius" if tokens.len() >= 3 => {
            let a = resolve_arc(sketch, tokens[0])?;
            let b = resolve_arc(sketch, tokens[1])?;
            find_ab!(sketch.equal_radius, a, b).map(|i| ConstraintId::Numbered(sketch.equal_radius[i].nid))
        }
        "coincident" if tokens.len() >= 3 => {
            let a = resolve_endpoint_ref(sketch, tokens[0])?;
            let b = resolve_endpoint_ref(sketch, tokens[1])?;
            find_coincident_id(sketch, a, b)
        }
        "point_on" if tokens.len() >= 3 => {
            let ep = resolve_endpoint_ref(sketch, tokens[0])?;
            let target = tokens[1];
            let found = if target.starts_with('L') || target.starts_with('l') {
                let line = resolve_line(sketch, target)?;
                find_point_on_line_id(sketch, ep, line)
            } else if is_arc_name(target) || target.starts_with('a') {
                let arc = resolve_arc(sketch, target)?;
                find_point_on_arc_id(sketch, ep, arc)
            } else { None };
            // Arc endpoints use helper points -- fall back to the
            // helper's own point_on entry if the direct lookup missed.
            if found.is_none()
                && let Some(p) = resolve_endpoint_as_point(&ctx.sketch, ep) {
                    if target.starts_with('L') || target.starts_with('l') {
                        let line = resolve_line(&ctx.sketch, target)?;
                        ctx.sketch.point_on_line.iter()
                            .position(|c| c.point == p && c.line == line)
                            .map(|i| ConstraintId::Numbered(ctx.sketch.point_on_line[i].nid))
                    } else if is_arc_name(target) || target.starts_with('a') {
                        let arc = resolve_arc(&ctx.sketch, target)?;
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
                    let line = resolve_line(sketch, tokens[1])?;
                    sketch.symmetry_pp.iter().find(|s| (s.a == a && s.c == c && s.line == line) || (s.a == c && s.c == a && s.line == line))
                        .map(|s| ConstraintId::Numbered(s.nid))
                } else {
                    let a = resolve_line(sketch, tokens[0])?;
                    let b = resolve_line(sketch, tokens[1])?;
                    let c = resolve_line(sketch, tokens[2])?;
                    sketch.symmetry_ll.iter().find(|s| s.b == b && ((s.a == a && s.c == c) || (s.a == c && s.c == a)))
                        .map(|s| ConstraintId::Numbered(s.nid))
                }
            }
        }
        "midpoint" if tokens.len() >= 3 => {
            let ep = resolve_endpoint_ref(sketch, tokens[0])?;
            find_midpoint_id(sketch, ep, tokens[1])
        }
        _ => { return Err(format!("Unknown constraint type: {}. Use: horizontal, vertical, parallel, perpendicular, equal, equal_radius, collinear, tangent, concentric, coincident, point_on, symmetry, midpoint, lock", ctype).into()); }
    };

    if let Some(id) = id {
        // Name it before the delete removes it.
        let id_name = crate::ids::constraint_id_name(&ctx.sketch, id);
        ctx.begin_group();
        ctx.exec(Action::DeleteConstraint { id });
        match id_name {
            Some(n) => Ok(ok(format!("Removed {} constraint {}", ctype, n))),
            None => Ok(ok(format!("Removed {} constraint", ctype))),
        }
    } else {
        Err("Constraint not found".to_string().into())
    }
}

// ---------------------------------------------------------------------------
// Rename param
// ---------------------------------------------------------------------------

pub(crate) fn cmd_rename_param(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: rename_param old_name new_name".into()); }
    let old = tokens[0];
    let new = tokens[1];
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == old) {
        if let Err(e) = ctx.sketch.validate_param_name(new, Some(idx)) {
            return Err(e.into());
        }
        let expr = ctx.sketch.user_params[idx].expr_str.clone();
        ctx.begin_group();
        ctx.exec(Action::UpdateUserParam { index: idx, name: new.to_string(), expr_str: expr });
        Ok(ok(format!("Renamed {} -> {}", old, new)))
    } else {
        Err(format!("Unknown parameter: {}", old).into())
    }
}

// ---------------------------------------------------------------------------
// Find
// ---------------------------------------------------------------------------

pub(crate) fn cmd_find(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return Err("Usage: find x,y [radius]".into()); }
    let pos = parse_coord(ctx, tokens[0], None)?;
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
        Ok(ok(format!("Nothing found within {:.1} of ({:.2},{:.2})", radius, pos.x, pos.y)))
    } else {
        Ok(ok(found.join(", ")))
    }
}

// ---------------------------------------------------------------------------
// Goto history position
// ---------------------------------------------------------------------------

pub(crate) fn cmd_goto(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let target_group: usize = match args.trim().parse() {
        Ok(v) => v,
        Err(_) => return Err("Usage: goto <group_number> (see 'history')".into()),
    };
    let groups = ctx.history.group_list();
    let target_pos = if target_group == 0 {
        0
    } else if target_group <= groups.len() {
        groups[target_group - 1].1
    } else {
        return Err(format!("Group {} does not exist (max {})", target_group, groups.len()).into());
    };
    if let Some((s, c)) = ctx.history.goto(target_pos) {
        ctx.sketch = s.into();
        ctx.cursor = c.pos;
        ctx.cursor_tangent = c.tangent;
    }
    Ok(ok(format!("Moved to group {} (position {})", target_group, ctx.history.cursor)))
}

// ---------------------------------------------------------------------------
// Session variables
// ---------------------------------------------------------------------------

pub(crate) fn cmd_let(ctx: &mut CommandContext, args: &str) -> CmdResult {
    // let name = expr
    let args = args.trim();
    let (name, expr) = match args.split_once('=') {
        Some((n, e)) => (n.trim(), e.trim()),
        None => return Err("Usage: let name = expression".into()),
    };
    if name.is_empty() { return Err("Variable name cannot be empty".into()); }
    // Try as coordinate (geo function or endpoint ref)
    if let Some(result) = eval_geo_coord(&ctx.sketch, expr) {
        match result {
            Ok(v) => {
                ctx.session_vecs.insert(name.to_string(), v);
                ctx.session_vars.insert(format!("{}.x", name), v.x);
                ctx.session_vars.insert(format!("{}.y", name), v.y);
                return Ok(ok(format!("{} = ({:.4}, {:.4})", name, v.x, v.y)));
            }
            Err(e) => return Err(e),
        }
    }
    if let Ok(pos) = resolve_endpoint_pos(&ctx.sketch, expr) {
        ctx.session_vecs.insert(name.to_string(), pos);
        ctx.session_vars.insert(format!("{}.x", name), pos.x);
        ctx.session_vars.insert(format!("{}.y", name), pos.y);
        return Ok(ok(format!("{} = ({:.4}, {:.4})", name, pos.x, pos.y)));
    }
    // Try as scalar
    if let Some(result) = eval_geo_scalar(&ctx.sketch, expr) {
        match result {
            Ok(v) => { ctx.session_vars.insert(name.to_string(), v); return Ok(ok(format!("{} = {:.6}", name, v))); }
            Err(e) => return Err(e),
        }
    }
    match eval_expr_with(&ctx.sketch, expr, &ctx.session_vars) {
        Ok(v) => { ctx.session_vars.insert(name.to_string(), v); Ok(ok(format!("{} = {:.6}", name, v))) }
        Err(e) => Err(format!("Eval error: {}", e).into()),
    }
}

// ---------------------------------------------------------------------------
// File operations
// ---------------------------------------------------------------------------

pub(crate) fn cmd_save(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let path = args.trim();
    if path.is_empty() { return Err("Usage: save path.json".into()); }
    match serde_json::to_string_pretty(&ctx.sketch) {
        Ok(json) => match std::fs::write(path, &json) {
            Ok(_) => Ok(ok(format!("Saved to {}", path))),
            Err(e) => Err(format!("Write error: {}", e).into()),
        },
        Err(e) => Err(format!("Serialize error: {}", e).into()),
    }
}

pub(crate) fn cmd_load(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let path = args.trim();
    if path.is_empty() { return Err("Usage: load path.json".into()); }
    match std::fs::read_to_string(path) {
        Ok(json) => match serde_json::from_str::<Sketch>(&json) {
            Ok(mut sketch) => {
                sketch.assign_constraint_names();
                sketch.solve();
                ctx.history = crate::history::History::new(&sketch);
                ctx.sketch = sketch.into();
                Ok(ok(format!("Loaded {}", path)))
            }
            Err(e) => Err(format!("Parse error: {}", e).into()),
        },
        Err(e) => Err(format!("Read error: {}", e).into()),
    }
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

pub(crate) fn cmd_cursor(ctx: &mut CommandContext, args: &str) -> CmdResult {
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
        return Ok(ok(format!("Cursor: {} tangent: {}", pos_str, tan_str)));
    }
    match args {
        "off" | "hide" => { ctx.cursor = None; return Ok(ok("Cursor hidden")); }
        "on" | "show" => {
            if ctx.cursor.is_none() { ctx.cursor = Some(vect2d::new(0.0, 0.0)); }
            let p = ctx.cursor.unwrap();
            return Ok(ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y)));
        }
        _ => {}
    }
    // Set cursor to coordinate (absolute, relative, endpoint ref, etc.)
    match parse_coord(ctx, args, ctx.cursor) {
        Ok(p) => { ctx.cursor = Some(p); Ok(ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y))) }
        Err(e) => Err(e.into()),
    }
}

// ---------------------------------------------------------------------------
// Dimension text position
// ---------------------------------------------------------------------------

pub(crate) fn cmd_dim_pos(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 3 { return Err("Usage: dim_pos d0 offset 1.5  or  dim_pos d0 along 0.3".into()); }
    let dim_name = tokens[0];
    let field = tokens[1];
    let val_str = tokens[2];
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == dim_name) {
        Some(i) => i,
        None => return Err(format!("Unknown dimension: {}", dim_name).into()),
    };
    let is_relative = val_str.starts_with('@');
    let val_str = val_str.strip_prefix('@').unwrap_or(val_str);
    let val = eval_expr(&ctx.sketch, val_str)?;
    let mut offset = ctx.sketch.dimensions[idx].offset;
    let mut text_along = ctx.sketch.dimensions[idx].text_along;
    match field {
        "offset" => {
            offset.y = if is_relative { offset.y + val } else { val };
            ctx.begin_group();
            ctx.exec(Action::MoveDimension { did: ctx.sketch.dimensions[idx].did, offset, text_along });
            Ok(ok(format!("{} offset = {:.4}", dim_name, ctx.sketch.dimensions[idx].offset.y)))
        }
        "along" => {
            text_along = if is_relative { text_along + val } else { val };
            ctx.begin_group();
            ctx.exec(Action::MoveDimension { did: ctx.sketch.dimensions[idx].did, offset, text_along });
            Ok(ok(format!("{} along = {:.4}", dim_name, ctx.sketch.dimensions[idx].text_along)))
        }
        _ => Err(format!("Unknown field '{}'. Use: offset, along", field).into()),
    }
}

// ---------------------------------------------------------------------------
// Derived dimensions
// ---------------------------------------------------------------------------

pub(crate) fn cmd_set_derived(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let name = args.trim();
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        Some(i) => i,
        None => return Err(format!("Unknown dimension: {}", name).into()),
    };
    if ctx.sketch.dimensions[idx].derived {
        return Ok(ok(format!("{} is already derived", name)));
    }
    let did = ctx.sketch.dimensions[idx].did;
    ctx.begin_group();
    ctx.exec(Action::ConvertDimension { did, derived: true, value: None });
    Ok(ok_or_status(ctx, format!("{} is now derived (reference only)", name)))
}

pub(crate) fn cmd_set_driven(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
    if tokens.is_empty() { return Err("Usage: set_driven d0 [value]".into()); }
    let name = tokens[0].trim();
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        Some(i) => i,
        None => return Err(format!("Unknown dimension: {}", name).into()),
    };
    if !ctx.sketch.dimensions[idx].derived {
        return Ok(ok(format!("{} is already driven (constraining)", name)));
    }
    let (new_value, new_expr) = if tokens.len() > 1 {
        let val_str = tokens[1].trim().trim_matches('"');
        if let Ok(v) = val_str.parse::<f64>() {
            (v, None)
        } else {
            // Expression: evaluate for initial value, store as expr
            let v = eval_expr(&ctx.sketch, val_str)?;
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
    Ok(ok_or_status(ctx, msg))
}

// ---------------------------------------------------------------------------
// DOF analysis
// ---------------------------------------------------------------------------


/// `timing on|off`: print each command line's wall time (enter to
/// prompt return) after it completes; bare `timing` reports the state.
pub(crate) fn cmd_timing(ctx: &mut CommandContext, args: &str) -> CmdResult {
    match args.trim() {
        "on" => {
            ctx.timing = true;
            Ok(ok("timing on"))
        }
        "off" => {
            ctx.timing = false;
            Ok(ok("timing off"))
        }
        "" => Ok(ok(format!("timing is {}", if ctx.timing { "on" } else { "off" }))),
        other => Err(format!("timing: expected on or off, got '{}'", other)),
    }
}

/// `settings [name [on|off]]`: runtime toggles. Bare `settings` lists
/// them; a name alone reports its state.
pub(crate) fn cmd_settings(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let state = |on: bool| if on { "on" } else { "off" };
    let tokens: Vec<&str> = args.split_whitespace().collect();
    match tokens.as_slice() {
        [] => Ok(ok(format!(
            "structural_dof: {}",
            state(ctx.sketch.structural_dof_enabled())
        ))),
        ["structural_dof"] => Ok(ok(format!(
            "structural_dof is {}",
            state(ctx.sketch.structural_dof_enabled())
        ))),
        ["structural_dof", v @ ("on" | "off")] => {
            let on = *v == "on";
            ctx.sketch.mutate_values(|s| s.set_structural_dof(on));
            Ok(ok(format!("structural_dof {}", v)))
        }
        _ => Err("Usage: settings [structural_dof [on|off]]".into()),
    }
}
