use super::*;

pub(crate) fn cmd_lock(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return Err("Usage: lock P0  or  lock L0.p1  or  lock L0.p1 x,y".into()); }
    let ep = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
    let pos = if tokens.len() > 1 {
        parse_coord(ctx, tokens[1], None)?
    } else {
        resolve_endpoint_pos(&ctx.sketch, tokens[0])?
    };
    let action = match ep {
        EndpointRef::Point(p) => Action::LockPoint { point: p, pos },
        EndpointRef::LineP1(l) => Action::LockLineP1 { line: l, pos },
        EndpointRef::LineP2(l) => Action::LockLineP2 { line: l, pos },
        EndpointRef::ArcCenter(a) => Action::LockArcCenter { arc: a, pos },
        _ => return Err("Can only lock points, line endpoints, and arc centers".into()),
    };
    ctx.begin_group();
    ctx.exec(action);
    Ok(ok(format!("Locked {} at ({:.2},{:.2})", tokens[0], pos.x, pos.y)))
}

pub(crate) fn cmd_unlock(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let name = args.trim();
    let ep = resolve_endpoint_ref(&ctx.sketch, name)?;
    let action = match ep {
        EndpointRef::Point(p) => Action::UnlockPoint { point: p },
        EndpointRef::LineP1(l) => Action::UnlockLineP1 { line: l },
        EndpointRef::LineP2(l) => Action::UnlockLineP2 { line: l },
        EndpointRef::ArcCenter(a) => Action::UnlockArcCenter { arc: a },
        _ => return Err("Can only unlock points, line endpoints, and arc centers".into()),
    };
    ctx.begin_group();
    ctx.exec(action);
    Ok(ok(format!("Unlocked {}", name)))
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

pub(crate) fn cmd_param(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
    if tokens.len() != 2 { return Err("Usage: param name value".into()); }
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
            return Err(e.into());
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
                return Err("Parameter change rejected: could not satisfy all constraints".into());
            }
    let val = ctx.sketch.user_params.iter().find(|p| p.name == name).map(|p| p.value).unwrap_or(0.0);
    Ok(ok(format!("{} {} = {} ({:.4})", if is_update { "Updated" } else { "Added" }, name, expr, val)))
}

pub(crate) fn cmd_del_param(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let name = args.trim();
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == name) {
        ctx.begin_group();
        ctx.exec(Action::RemoveUserParam { index: idx });
        Ok(ok(format!("Deleted parameter {}", name)))
    } else {
        Err(format!("Unknown parameter: {}", name).into())
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

pub(crate) fn cmd_style(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return Err("Usage: style L0 [solid|dashed|dashdot]".into()); }
    let name = tokens[0];
    if tokens.len() == 1 {
        // Query
        if name.starts_with('L') {
            let r = resolve_line(&ctx.sketch, name)?;
            return Ok(ok(format!("{}: {}", name, ctx.sketch.lines[r].style.name())));
        } else if is_arc_name(name) {
            let r = resolve_arc(&ctx.sketch, name)?;
            return Ok(ok(format!("{}: {}", name, ctx.sketch.arcs[r].style.name())));
        }
        return Err("Style applies to lines and arcs".into());
    }
    let style = match LineStyle::from_name(tokens[1]) {
        Some(s) => s,
        None => return Err(format!("Unknown style '{}'. Use: solid, dashed, dashdot", tokens[1]).into()),
    };
    if name.starts_with('L') {
        let r = resolve_line(&ctx.sketch, name)?;
        ctx.begin_group();
        ctx.exec(Action::SetStyleLine { line: r, style });
        Ok(ok(format!("{}: {}", name, style.name())))
    } else if is_arc_name(name) {
        let r = resolve_arc(&ctx.sketch, name)?;
        ctx.begin_group();
        ctx.exec(Action::SetStyleArc { arc: r, style });
        Ok(ok(format!("{}: {}", name, style.name())))
    } else {
        Err("Style applies to lines and arcs".into())
    }
}

pub(crate) fn cmd_quiet(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return Err("Usage: quiet L0 [on|off] | quiet A0 EA0 ...".into()); }
    // Check for explicit on/off
    let explicit = if tokens.last() == Some(&"on") { Some(true) }
        else if tokens.last() == Some(&"off") { Some(false) }
        else { None };
    let names = if explicit.is_some() { &tokens[..tokens.len()-1] } else { &tokens[..] };
    ctx.begin_group();
    let mut msgs = Vec::new();
    for name in names {
        if name.starts_with('P') {
            let r = resolve_point(&ctx.sketch, name)?;
            let q = explicit.unwrap_or(!ctx.sketch.points[r].quiet);
            ctx.exec(Action::SetQuietPoint { point: r, on: q });
            msgs.push(format!("{}: quiet={}", name, q));
        } else if name.starts_with('L') {
            let r = resolve_line(&ctx.sketch, name)?;
            let q = explicit.unwrap_or(!ctx.sketch.lines[r].quiet);
            ctx.exec(Action::SetQuietLine { line: r, on: q });
            msgs.push(format!("{}: quiet={}", name, q));
        } else if is_arc_name(name) {
            let r = resolve_arc(&ctx.sketch, name)?;
            let q = explicit.unwrap_or(!ctx.sketch.arcs[r].quiet);
            ctx.exec(Action::SetQuietArc { arc: r, on: q });
            msgs.push(format!("{}: quiet={}", name, q));
        } else {
            return Err(format!("Unknown entity '{}'", name).into());
        }
    }
    Ok(ok(msgs.join(", ")))
}

pub(crate) fn cmd_constr(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return Err("Usage: constr L0 [on|off] | constr A0 EA0 ...".into()); }
    let explicit = if tokens.last() == Some(&"on") { Some(true) }
        else if tokens.last() == Some(&"off") { Some(false) }
        else { None };
    let names = if explicit.is_some() { &tokens[..tokens.len()-1] } else { &tokens[..] };
    ctx.begin_group();
    let mut msgs = Vec::new();
    for name in names {
        if name.starts_with('L') {
            let r = resolve_line(&ctx.sketch, name)?;
            let c = explicit.unwrap_or(!ctx.sketch.lines[r].construction);
            ctx.exec(Action::SetConstructionLine { line: r, on: c });
            msgs.push(format!("{}: construction={}", name, c));
        } else if is_arc_name(name) {
            let r = resolve_arc(&ctx.sketch, name)?;
            let c = explicit.unwrap_or(!ctx.sketch.arcs[r].construction);
            ctx.exec(Action::SetConstructionArc { arc: r, on: c });
            msgs.push(format!("{}: construction={}", name, c));
        } else {
            return Err(format!("constr applies to lines and arcs, not '{}'", name).into());
        }
    }
    Ok(ok(msgs.join(", ")))
}

pub(crate) fn cmd_drag(ctx: &mut CommandContext, args: &str) -> CmdResult {

    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 {
        return Err("Usage: drag L0.p1 x,y | drag L0 @dx,dy | drag P0 x,y | drag A0.center x,y".into());
    }
    let entity_spec = tokens[0];

    // Resolve current position of the drag target
    use arael_sketch_solver::DragTarget;

    let (target, current_pos) = if let Some((ent, field)) = entity_spec.split_once('.') {
        if ent.starts_with('L') {
            let r = resolve_line(&ctx.sketch, ent)?;
            match field {
                "p1" => (DragTarget::LineP1(r), ctx.sketch.lines[r].p1.value),
                "p2" => (DragTarget::LineP2(r), ctx.sketch.lines[r].p2.value),
                _ => return Err(format!("Unknown line field '{}'. Use p1 or p2", field).into()),
            }
        } else if is_arc_name(ent) {
            let r = resolve_arc(&ctx.sketch, ent)?;
            match field {
                "center" => (DragTarget::ArcCenter(r), ctx.sketch.arcs[r].center.value),
                "start" => (DragTarget::ArcStart(r), crate::geometry::arc_start_pos(&ctx.sketch.arcs[r])),
                "end" => (DragTarget::ArcEnd(r), crate::geometry::arc_end_pos(&ctx.sketch.arcs[r])),
                _ => return Err(format!("Unknown arc field '{}'. Use center, start, or end", field).into()),
            }
        } else {
            return Err(format!("Unknown entity '{}' in drag target", ent).into());
        }
    } else if entity_spec.starts_with('P') {
        let r = resolve_point(&ctx.sketch, entity_spec)?;
        (DragTarget::Point(r), ctx.sketch.points[r].pos.value)
    } else if entity_spec.starts_with('L') {
        let r = resolve_line(&ctx.sketch, entity_spec)?;
        let mid = vect2d::new(
            (ctx.sketch.lines[r].p1.value.x + ctx.sketch.lines[r].p2.value.x) / 2.0,
            (ctx.sketch.lines[r].p1.value.y + ctx.sketch.lines[r].p2.value.y) / 2.0,
        );
        (DragTarget::LineBody(r), mid)
    } else if is_arc_name(entity_spec) {
        let r = resolve_arc(&ctx.sketch, entity_spec)?;
        (DragTarget::ArcBody(r), ctx.sketch.arcs[r].center.value)
    } else {
        return Err(format!("Unknown entity '{}'. Use P0, L0, L0.p1, A0.center, etc.", entity_spec).into());
    };

    // Parse target coordinate (absolute or relative)
    let target_pos = parse_coord(ctx, tokens[1], Some(current_pos))?;

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
                return Err("Drag failed: could not satisfy constraints".into());
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
    Ok(ok(format!("Dragged {} to ({:.4}, {:.4})", entity_spec, new_pos.x, new_pos.y)))
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

