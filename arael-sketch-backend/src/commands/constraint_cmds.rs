use super::*;

/// Shared tail for constraint-applying commands: duplicate rejection
/// (naming the blocking constraint), grouped action application, and
/// the applied-message carrying the minted C-id.
fn apply_constraint(
    ctx: &mut CommandContext,
    exists: Option<u32>,
    what: &str,
    action: Action,
    last_nid: impl Fn(&Sketch) -> u32,
    applied: &str,
) -> CmdResult {
    if let Some(nid) = exists {
        return Err(format!("{} constraint already exists (C{})", what, nid));
    }
    ctx.begin_group();
    ctx.exec(action);
    let nid = last_nid(&ctx.sketch);
    let msg = applied_msg(&ctx.sketch, &format!("C{}", nid), &format!("Applied {}", applied));
    Ok(ok_or_status(ctx, msg))
}

/// Closure reading the most recent nid off one constraint collection.
macro_rules! last_nid {
    ($table:ident) => { |s: &Sketch| s.$table.last().map(|c| c.nid).unwrap_or(0) }
}

/// Closure for actions that fan out over several collections: the most
/// recently assigned nid is next_constraint_id - 1.
fn last_minted_nid(s: &Sketch) -> u32 {
    s.next_constraint_id.saturating_sub(1)
}

pub(crate) fn cmd_horizontal(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut lines = Vec::new();
    for name in args.split_whitespace() {
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => lines.push(r),
            Err(e) => return Err(e),
        }
    }
    if lines.is_empty() { return Err("Usage: horizontal L0 [L1 ...]".into()); }
    for &r in &lines {
        if ctx.sketch.lines[r].constraints.horizontal {
            return Err(format!("{} is already horizontal", ctx.sketch.lines[r].name).into());
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
    Ok(ok_or_status(ctx, parts.join(", ")))
}

pub(crate) fn cmd_vertical(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut lines = Vec::new();
    for name in args.split_whitespace() {
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => lines.push(r),
            Err(e) => return Err(e),
        }
    }
    if lines.is_empty() { return Err("Usage: vertical L0 [L1 ...]".into()); }
    for &r in &lines {
        if ctx.sketch.lines[r].constraints.vertical {
            return Err(format!("{} is already vertical", ctx.sketch.lines[r].name).into());
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
    Ok(ok_or_status(ctx, parts.join(", ")))
}

/// Classify a `parallel` argument: may be a line, an ellipse major
/// axis (only ellipses have an optimisable rotation), or fail. A
/// circular arc (non-ellipse) is rejected with a clear message.
pub(crate) enum ParallelArg {
    Line(Ref<Line>),
    Ellipse(Ref<Arc>),
}

pub(crate) fn resolve_parallel_arg(sketch: &Sketch, name: &str) -> Result<ParallelArg, String> {
    if let Ok(arc) = resolve_arc(sketch, name) {
        if !sketch.arcs[arc].is_ellipse {
            return Err(format!("parallel on {name}: only ellipses have an optimisable major-axis rotation; circular arcs have no orientation to align"));
        }
        return Ok(ParallelArg::Ellipse(arc));
    }
    let line = resolve_line(sketch, name)?;
    Ok(ParallelArg::Line(line))
}

pub(crate) fn cmd_parallel(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: parallel <L|EA> <L|EA>".into()); }
    let a = resolve_parallel_arg(&ctx.sketch, tokens[0])?;
    let b = resolve_parallel_arg(&ctx.sketch, tokens[1])?;
    match (a, b) {
        (ParallelArg::Line(la), ParallelArg::Line(lb)) => {
            if la == lb { return Err("Cannot constrain a line parallel to itself".into()); }
            let exists = ctx.sketch.parallel.iter().find(|c| (c.a == la && c.b == lb) || (c.a == lb && c.b == la)).map(|c| c.nid);
            apply_constraint(ctx, exists, "Parallel", Action::ApplyParallel { a: la, b: lb },
                             last_nid!(parallel), "parallel")
        }
        (ParallelArg::Ellipse(ea), ParallelArg::Ellipse(eb)) => {
            if ea == eb { return Err("Cannot constrain an ellipse parallel to itself".into()); }
            let exists = ctx.sketch.arc_arc_parallel.iter().find(|c| (c.a == ea && c.b == eb) || (c.a == eb && c.b == ea)).map(|c| c.nid);
            apply_constraint(ctx, exists, "Parallel", Action::ApplyArcArcParallel { a: ea, b: eb },
                             last_nid!(arc_arc_parallel), "parallel")
        }
        (ParallelArg::Ellipse(ea), ParallelArg::Line(lb))
        | (ParallelArg::Line(lb), ParallelArg::Ellipse(ea)) => {
            let exists = ctx.sketch.arc_line_parallel.iter().find(|c| c.arc == ea && c.line == lb).map(|c| c.nid);
            apply_constraint(ctx, exists, "Parallel", Action::ApplyArcLineParallel { arc: ea, line: lb },
                             last_nid!(arc_line_parallel), "parallel")
        }
    }
}

pub(crate) fn cmd_perpendicular(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: perpendicular L0 L1".into()); }
    let a = resolve_line(&ctx.sketch, tokens[0])?;
    let b = resolve_line(&ctx.sketch, tokens[1])?;
    if a == b { return Err("Cannot constrain a line perpendicular to itself".into()); }
    let exists = ctx.sketch.perpendicular.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid);
    apply_constraint(ctx, exists, "Perpendicular", Action::ApplyPerpendicular { a, b },
                     last_nid!(perpendicular), "perpendicular")
}

pub(crate) fn cmd_equal(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: equal L0 L1  or  equal A0 A1".into()); }
    if tokens[0].starts_with('L') && tokens[1].starts_with('L') {
        let a = resolve_line(&ctx.sketch, tokens[0])?;
        let b = resolve_line(&ctx.sketch, tokens[1])?;
        if a == b { return Err("Cannot constrain a line equal to itself".into()); }
        let exists = ctx.sketch.equal_length.iter().find(|c|
            (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid);
        apply_constraint(ctx, exists, "Equal length", Action::ApplyEqualLength { a, b },
                         last_nid!(equal_length), "equal length")
    } else if is_arc_name(tokens[0]) && is_arc_name(tokens[1]) {
        let a = resolve_arc(&ctx.sketch, tokens[0])?;
        let b = resolve_arc(&ctx.sketch, tokens[1])?;
        if a == b { return Err("Cannot constrain an arc equal to itself".into()); }
        let exists = ctx.sketch.equal_radius.iter().find(|c|
            (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid);
        apply_constraint(ctx, exists, "Equal radius", Action::ApplyEqualRadius { a, b },
                         last_nid!(equal_radius), "equal radius")
    } else {
        Err("equal needs two lines or two arcs".into())
    }
}

pub(crate) fn cmd_collinear(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: collinear L0 L1".into()); }
    let a = resolve_line(&ctx.sketch, tokens[0])?;
    let b = resolve_line(&ctx.sketch, tokens[1])?;
    if a == b { return Err("Cannot constrain a line collinear with itself".into()); }
    let exists = ctx.sketch.collinear.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid);
    apply_constraint(ctx, exists, "Collinear", Action::ApplyCollinear { a, b },
                     last_nid!(collinear), "collinear")
}

pub(crate) fn cmd_tangent(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: tangent L0 A0  or  tangent A0 A1".into()); }
    if tokens[0].starts_with('L') && is_arc_name(tokens[1]) {
        let line = resolve_line(&ctx.sketch, tokens[0])?;
        let arc = resolve_arc(&ctx.sketch, tokens[1])?;
        let exists = ctx.sketch.tangent_la.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid);
        apply_constraint(ctx, exists, "Tangent", Action::ApplyTangentLA { line, arc },
                         last_nid!(tangent_la), "tangent")
    } else if is_arc_name(tokens[0]) && is_arc_name(tokens[1]) {
        let a = resolve_arc(&ctx.sketch, tokens[0])?;
        let b = resolve_arc(&ctx.sketch, tokens[1])?;
        if a == b { return Err("Cannot constrain an arc tangent to itself".into()); }
        let exists = ctx.sketch.tangent_aa.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid);
        apply_constraint(ctx, exists, "Tangent", Action::ApplyTangentAA { a, b },
                         last_nid!(tangent_aa), "tangent")
    } else {
        Err("tangent needs line+arc or arc+arc".into())
    }
}

pub(crate) fn cmd_coincident(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: coincident L0.p2 L1.p1".into()); }
    let a = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
    let b = resolve_endpoint_ref(&ctx.sketch, tokens[1])?;
    if a == b { return Err("Cannot constrain an endpoint coincident with itself".into()); }
    use EndpointRef::*;
    let s = &ctx.sketch;
    // Check for existing equivalent coincident constraint
    let exists = match (a, b) {
        (Point(a), Point(b)) => s.coincident_pp.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => s.coincident_lp1.iter().find(|c| c.line == l && c.point == p).map(|c| c.nid),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => s.coincident_lp2.iter().find(|c| c.line == l && c.point == p).map(|c| c.nid),
        (LineP1(a), LineP1(b)) => s.coincident_ll11.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid),
        (LineP1(a), LineP2(b)) => s.coincident_ll12.iter().find(|c| c.a == a && c.b == b).map(|c| c.nid)
            .or_else(|| s.coincident_ll21.iter().find(|c| c.a == b && c.b == a).map(|c| c.nid)),
        (LineP2(a), LineP1(b)) => s.coincident_ll21.iter().find(|c| c.a == a && c.b == b).map(|c| c.nid)
            .or_else(|| s.coincident_ll12.iter().find(|c| c.a == b && c.b == a).map(|c| c.nid)),
        (LineP2(a), LineP2(b)) => s.coincident_ll22.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid),
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => s.coincident_arc_center.iter().find(|c| c.point == p && c.arc == arc).map(|c| c.nid),
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => s.coincident_arc_start.iter().find(|c| c.point == p && c.arc == arc).map(|c| c.nid),
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => s.coincident_arc_end.iter().find(|c| c.point == p && c.arc == arc).map(|c| c.nid),
        (LineP1(line), ArcCenter(arc)) | (ArcCenter(arc), LineP1(line)) => s.coincident_lp1_arc_center.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid),
        (LineP2(line), ArcCenter(arc)) | (ArcCenter(arc), LineP2(line)) => s.coincident_lp2_arc_center.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid),
        (LineP1(line), ArcStart(arc)) | (ArcStart(arc), LineP1(line)) => s.coincident_lp1_arc_start.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid),
        (LineP2(line), ArcStart(arc)) | (ArcStart(arc), LineP2(line)) => s.coincident_lp2_arc_start.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid),
        (LineP1(line), ArcEnd(arc)) | (ArcEnd(arc), LineP1(line)) => s.coincident_lp1_arc_end.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid),
        (LineP2(line), ArcEnd(arc)) | (ArcEnd(arc), LineP2(line)) => s.coincident_lp2_arc_end.iter().find(|c| c.line == line && c.arc == arc).map(|c| c.nid),
        _ => None,
    };
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
        _ => return Err("Unsupported coincident combination".into()),
    };
    apply_constraint(ctx, exists, "Coincident", action, last_minted_nid, "coincident")
}

pub(crate) fn cmd_concentric(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: concentric A0 A1".into()); }
    let a = resolve_arc(&ctx.sketch, tokens[0])?;
    let b = resolve_arc(&ctx.sketch, tokens[1])?;
    if a == b { return Err("Cannot constrain an arc concentric with itself".into()); }
    let exists = ctx.sketch.concentric.iter().find(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)).map(|c| c.nid);
    apply_constraint(ctx, exists, "Concentric", Action::ApplyConcentric { a, b },
                     last_nid!(concentric), "concentric")
}

/// `on_normal <placed endpoint> <reference endpoint>`: the placed endpoint
/// lies on the normal of the reference curve at the reference endpoint.
pub(crate) fn cmd_on_normal(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 {
        return Err("Usage: on_normal L1.p2 L0.p2 | on_normal A1.start A0.start".into());
    }
    let to_dim = |ep: EndpointRef| -> Result<DimensionEndpoint, String> {
        match ep {
            EndpointRef::LineP1(l) => Ok(DimensionEndpoint::LineP1(l)),
            EndpointRef::LineP2(l) => Ok(DimensionEndpoint::LineP2(l)),
            EndpointRef::ArcStart(a) => Ok(DimensionEndpoint::ArcStart(a)),
            EndpointRef::ArcEnd(a) => Ok(DimensionEndpoint::ArcEnd(a)),
            _ => Err("on_normal takes line endpoints (p1/p2) or arc endpoints (start/end)".into()),
        }
    };
    let placed = to_dim(resolve_endpoint_ref(&ctx.sketch, tokens[0])?)?;
    let reference = to_dim(resolve_endpoint_ref(&ctx.sketch, tokens[1])?)?;
    // Duplicates and unsupported pairs are rejected by validate_action.
    apply_constraint(ctx, None, "On-normal", Action::ApplyOnNormal { placed, reference },
                     last_minted_nid, "on_normal")
}

// ---------------------------------------------------------------------------
// Dimension commands
// ---------------------------------------------------------------------------


pub(crate) fn cmd_midpoint(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: midpoint P0 L0 | midpoint L0.p1 A0".into()); }
    let ep = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
    let target = tokens[1];
    // Target is a line or arc
    if let Ok(line) = resolve_line(&ctx.sketch, target) {
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.midpoint.iter().find(|c| c.point == p && c.line == line).map(|c| c.nid),
            EndpointRef::LineP1(l) => s.midpoint_lp1.iter().find(|c| c.line == l && c.target == line).map(|c| c.nid),
            EndpointRef::LineP2(l) => s.midpoint_lp2.iter().find(|c| c.line == l && c.target == line).map(|c| c.nid),
            EndpointRef::ArcStart(a) => s.midpoint_arc_start.iter().find(|c| c.arc == a && c.line == line).map(|c| c.nid),
            EndpointRef::ArcEnd(a) => s.midpoint_arc_end.iter().find(|c| c.arc == a && c.line == line).map(|c| c.nid),
            _ => None,
        };
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyMidpoint { point: p, line },
            EndpointRef::LineP1(l) => Action::ApplyMidpointLP1 { line: l, target: line },
            EndpointRef::LineP2(l) => Action::ApplyMidpointLP2 { line: l, target: line },
            EndpointRef::ArcStart(a) => Action::ApplyMidpointArcStart { arc: a, line },
            EndpointRef::ArcEnd(a) => Action::ApplyMidpointArcEnd { arc: a, line },
            _ => return Err("First arg must be a point or endpoint".into()),
        };
        apply_constraint(ctx, exists, "Midpoint", action, last_minted_nid, "midpoint")
    } else if let Ok(arc) = resolve_arc(&ctx.sketch, target) {
        if ctx.sketch.arcs[arc].closed { return Err("Cannot use midpoint on a full circle".into()); }
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.midpoint_arc_point.iter().find(|c| c.point == p && c.arc == arc).map(|c| c.nid),
            EndpointRef::LineP1(l) => s.midpoint_lp1_arc.iter().find(|c| c.line == l && c.arc == arc).map(|c| c.nid),
            EndpointRef::LineP2(l) => s.midpoint_lp2_arc.iter().find(|c| c.line == l && c.arc == arc).map(|c| c.nid),
            EndpointRef::ArcStart(a) => s.midpoint_arc_start_arc.iter().find(|c| c.a == a && c.b == arc).map(|c| c.nid),
            EndpointRef::ArcEnd(a) => s.midpoint_arc_end_arc.iter().find(|c| c.a == a && c.b == arc).map(|c| c.nid),
            _ => None,
        };
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyMidpointArcPoint { point: p, arc },
            EndpointRef::LineP1(l) => Action::ApplyMidpointLP1Arc { line: l, arc },
            EndpointRef::LineP2(l) => Action::ApplyMidpointLP2Arc { line: l, arc },
            EndpointRef::ArcStart(a) => Action::ApplyMidpointArcStartArc { a, b: arc },
            EndpointRef::ArcEnd(a) => Action::ApplyMidpointArcEndArc { a, b: arc },
            _ => return Err("First arg must be a point or endpoint".into()),
        };
        apply_constraint(ctx, exists, "Midpoint", action, last_minted_nid, "midpoint")
    } else {
        Err("Second arg must be a line (L0) or arc (A0)".into())
    }
}

/// EndpointRef -> DimensionEndpoint, the actions' endpoint currency.
pub(crate) fn to_dim_endpoint(ep: EndpointRef) -> DimensionEndpoint {
    match ep {
        EndpointRef::Point(p) => DimensionEndpoint::Point(p),
        EndpointRef::LineP1(l) => DimensionEndpoint::LineP1(l),
        EndpointRef::LineP2(l) => DimensionEndpoint::LineP2(l),
        EndpointRef::ArcCenter(a) => DimensionEndpoint::ArcCenter(a),
        EndpointRef::ArcStart(a) => DimensionEndpoint::ArcStart(a),
        EndpointRef::ArcEnd(a) => DimensionEndpoint::ArcEnd(a),
    }
}


pub(crate) fn cmd_symmetry(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 3 { return Err("Usage: symmetry L0 L1 L2 | symmetry P0 L0 P1 | symmetry A0 L0 A1".into()); }
    // Try arc + line + arc symmetry
    if is_arc_name(tokens[0]) && is_arc_name(tokens[2])
        && let (Ok(a), Ok(line), Ok(c)) = (resolve_arc(&ctx.sketch, tokens[0]),
            resolve_line(&ctx.sketch, tokens[1]),
            resolve_arc(&ctx.sketch, tokens[2]))
        {
            if a == c { return Err("Cannot constrain an arc symmetric with itself".into()); }
            let exists = ctx.sketch.symmetry_aa.iter().find(|s|
                s.line == line && ((s.a == a && s.c == c) || (s.a == c && s.c == a))).map(|c| c.nid);
            return apply_constraint(ctx, exists, "Symmetry", Action::ApplySymmetryAA { a, line, c },
                                    last_nid!(symmetry_aa), "arc symmetry");
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
        let a = to_dim_endpoint(resolve_endpoint_ref(&ctx.sketch, tokens[0])?);
        let line = resolve_line(&ctx.sketch, tokens[1]).unwrap();
        let c = to_dim_endpoint(resolve_endpoint_ref(&ctx.sketch, tokens[2])?);
        return apply_constraint(ctx, None, "Symmetry", Action::ApplySymmetryPP { a, line, c },
                                last_nid!(symmetry_pp), "point symmetry");
    }
    // Fall back to line-line-line symmetry
    let a = resolve_line(&ctx.sketch, tokens[0])?;
    let b = resolve_line(&ctx.sketch, tokens[1])?;
    let c = resolve_line(&ctx.sketch, tokens[2])?;
    let exists = ctx.sketch.symmetry_ll.iter().find(|s|
        s.b == b && ((s.a == a && s.c == c) || (s.a == c && s.c == a))).map(|c| c.nid);
    apply_constraint(ctx, exists, "Symmetry", Action::ApplySymmetryLL { a, b, c },
                     last_nid!(symmetry_ll), "symmetry")
}

/// Reflect a point across a line defined by two points.
pub(crate) fn mirror_point_across(pt: vect2d, lp1: vect2d, lp2: vect2d) -> vect2d {
    let dx = lp2.x - lp1.x;
    let dy = lp2.y - lp1.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-24 { return pt; }
    let t = ((pt.x - lp1.x) * dx + (pt.y - lp1.y) * dy) / len2;
    let proj = vect2d::new(lp1.x + t * dx, lp1.y + t * dy);
    vect2d::new(2.0 * proj.x - pt.x, 2.0 * proj.y - pt.y)
}

/// Source entity to be mirrored.
pub(crate) enum MirrorSource {
    Line(Ref<Line>),
    Point(Ref<Point>),
    Arc(Ref<Arc>),
}

pub(crate) fn cmd_mirror(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    // Find "about" keyword
    let about_pos = match tokens.iter().position(|&t| t == "about") {
        Some(p) => p,
        None => return Err("Usage: mirror L0 L1 ... about L_axis [noconstraint] [strict]".into()),
    };
    if about_pos == 0 {
        return Err("No entities to mirror".into());
    }

    // Parse trailing keywords after the mirror line
    let mut after_about: Vec<&str> = tokens[about_pos + 1..].to_vec();
    let [noconstraint, strict] = peel_keywords(&mut after_about, ["noconstraint", "strict"]);
    if noconstraint && strict {
        return Err("noconstraint conflicts with strict".into());
    }
    if after_about.len() != 1 {
        return Err("Expected exactly one mirror line after 'about'".into());
    }
    let mirror_line = resolve_line(&ctx.sketch, after_about[0])?;

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
            return Err("No lines, arcs, or points in selection".into());
        }
    } else {
        for &name in source_tokens {
            if name.starts_with('L') {
                let r = resolve_line(&ctx.sketch, name)?;
                if r == mirror_line { return Err(format!("Cannot mirror {} about itself", name).into()); }
                sources.push(MirrorSource::Line(r));
            } else if is_arc_name(name) {
                let r = resolve_arc(&ctx.sketch, name)?;
                sources.push(MirrorSource::Arc(r));
            } else if name.starts_with('P') {
                let r = resolve_point(&ctx.sketch, name)?;
                sources.push(MirrorSource::Point(r));
            } else {
                return Err(format!("Unknown entity: {}", name).into());
            }
        }
    }

    let ml = &ctx.sketch.lines[mirror_line];
    let mlp1 = ml.p1.value;
    let mlp2 = ml.p2.value;

    // The whole mirror runs as two batches (entities, then
    // constraints): a per-item exec pays a full validation and DOF
    // rank analysis per action.
    let old_cost = ctx.sketch.current_cost();
    ctx.begin_group();
    let mut warnings: Vec<String> = Vec::new();
    let mut applied: Vec<String> = Vec::new();
    let mut msgs = Vec::new();

    // Creation actions, one per source, computed from source values.
    let mut creates: Vec<Action> = Vec::new();
    for source in &sources {
        match source {
            MirrorSource::Line(src_ref) => {
                let l = &ctx.sketch.lines[*src_ref];
                creates.push(Action::AddLine {
                    p1: mirror_point_across(l.p1.value, mlp1, mlp2),
                    p2: mirror_point_across(l.p2.value, mlp1, mlp2),
                });
            }
            MirrorSource::Point(src_ref) => {
                let pt = &ctx.sketch.points[*src_ref];
                creates.push(Action::AddPoint { pos: mirror_point_across(pt.pos.value, mlp1, mlp2) });
            }
            MirrorSource::Arc(src_ref) => {
                let a = &ctx.sketch.arcs[*src_ref];
                let mc = mirror_point_across(a.center.value, mlp1, mlp2);
                let r = a.radius.value;
                if a.closed {
                    let edge = vect2d::new(mc.x + r, mc.y);
                    creates.push(Action::AddCircle { center: mc, edge });
                } else {
                    let ms = mirror_point_across(arc_start_pos(a), mlp1, mlp2);
                    let me = mirror_point_across(arc_end_pos(a), mlp1, mlp2);
                    let mid_angle = (a.start_angle.value + a.end_angle.value) / 2.0;
                    let mid_pt = vect2d::new(
                        a.center.value.x + r * mid_angle.cos(),
                        a.center.value.y + r * mid_angle.sin(),
                    );
                    creates.push(Action::AddArc { start: ms, end: me, mid: mirror_point_across(mid_pt, mlp1, mlp2) });
                }
            }
        }
    }
    let created = ctx.exec(Action::Batch { label: "Mirror".into(), actions: creates });
    if let Some(e) = ctx.status_error.take() {
        return Err(e);
    }
    let Created::Many(created) = created else {
        return Err("Internal: creation batch added nothing".into());
    };

    // Maps from source to mirrored refs
    let mut line_map: Vec<(Ref<Line>, Ref<Line>)> = Vec::new();
    let mut point_map: Vec<(Ref<Point>, Ref<Point>)> = Vec::new();
    let mut arc_map: Vec<(Ref<Arc>, Ref<Arc>)> = Vec::new();
    let mut mirrored_names: Vec<String> = Vec::new();
    for (source, c) in sources.iter().zip(created) {
        let (src_name, new_name) = match (source, c) {
            (MirrorSource::Line(src), Created::Line(new_ref)) => {
                line_map.push((*src, new_ref));
                (ctx.sketch.lines[*src].name.clone(), ctx.sketch.lines[new_ref].name.clone())
            }
            (MirrorSource::Point(src), Created::Point(new_ref)) => {
                point_map.push((*src, new_ref));
                (ctx.sketch.points[*src].name.clone(), ctx.sketch.points[new_ref].name.clone())
            }
            (MirrorSource::Arc(src), Created::Arc(new_ref)) => {
                arc_map.push((*src, new_ref));
                (ctx.sketch.arcs[*src].name.clone(), ctx.sketch.arcs[new_ref].name.clone())
            }
            (MirrorSource::Arc(_), _) => {
                return Err("Cannot mirror arc: degenerate mirrored geometry".into());
            }
            _ => return Err("Internal: creation action added no entity".into()),
        };
        msgs.push(format!("Mirrored {} -> {}", src_name, new_name));
        mirrored_names.push(new_name);
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
        // Coincidents apply before the symmetry constraints: they
        // merge the free copies first, so the deduped one-symmetry-
        // per-position set below removes exactly the remaining DOF.
        let mut constraint_actions = coinc_actions;

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
            constraint_actions.push((Action::ApplySymmetryPP { a, line: mirror_line, c }, desc));
        }

        // One batch for every coincident and symmetry constraint: the
        // mirrored geometry satisfies them exactly, so they skip the
        // per-constraint gate, like a pattern's image constraints.
        let watermark = ctx.sketch.next_constraint_id;
        if !constraint_actions.is_empty() {
            let (acts, descs): (Vec<Action>, Vec<String>) = constraint_actions.into_iter().unzip();
            ctx.exec(Action::Batch { label: "Mirror constraints".into(), actions: acts });
            if let Some(e) = ctx.status_error.take() {
                return Err(e);
            }
            applied.extend(descs);
            let names = constraint_names_since(ctx, watermark);
            if !names.is_empty() {
                applied.push(format!("constraints: {}", names.join(" ")));
            }
        }
    }

    // The per-constraint cost gate is gone with the batch; one
    // whole-operation check replaces it. A mirror is exact, so a cost
    // jump means a constraint could not be satisfied.
    let quick = ctx.sketch.current_cost();
    let new_cost = if quick <= old_cost + 1e-6 { quick } else { ctx.sketch.solve().end_cost };
    if new_cost > old_cost + 1e-3 {
        let m = format!("mirror could not satisfy all constraints (cost {:.3e} -> {:.3e})", old_cost, new_cost);
        if strict {
            use crate::corner_ops::ActionRunner;
            ctx.rollback_group();
            return Err(m);
        }
        warnings.push(m);
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
    Ok(ok(msg))
}

/// Which arc endpoint a helper point bridges to.
pub(crate) enum ArcEp { Center, Start, End }

/// Check if an arc endpoint already has a point_on_line constraint via a helper point.
pub(crate) fn has_arc_endpoint_on_line(s: &Sketch, arc: Ref<Arc>, ep: ArcEp, line: Ref<Line>) -> Option<u32> {
    // Find helper points bridged to this arc endpoint
    let bridged_points: Vec<Ref<Point>> = match ep {
        ArcEp::Center => s.coincident_arc_center.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::Start => s.coincident_arc_start.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::End => s.coincident_arc_end.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
    };
    // The point_on_line constraint holding one of those points, if any
    s.point_on_line.iter()
        .find(|c| c.line == line && bridged_points.contains(&c.point))
        .map(|c| c.nid)
}

/// The point_on_arc constraint an arc endpoint already carries via a
/// helper point, if any.
pub(crate) fn has_arc_endpoint_on_arc(s: &Sketch, src: Ref<Arc>, ep: ArcEp, target: Ref<Arc>) -> Option<u32> {
    let bridged_points: Vec<Ref<Point>> = match ep {
        ArcEp::Center => s.coincident_arc_center.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::Start => s.coincident_arc_start.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::End => s.coincident_arc_end.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
    };
    s.point_on_arc.iter()
        .find(|c| c.arc == target && bridged_points.contains(&c.point))
        .map(|c| c.nid)
}

pub(crate) fn cmd_point_on(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: point_on P0 L0  or  point_on L0.p1 A0".into()); }
    let ep = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
    let target = tokens[1];
    if target.starts_with('L') {
        let line = resolve_line(&ctx.sketch, target)?;
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.point_on_line.iter().find(|c| c.point == p && c.line == line).map(|c| c.nid),
            EndpointRef::LineP1(l) => s.line_p1_on_line.iter().find(|c| c.a == l && c.b == line).map(|c| c.nid),
            EndpointRef::LineP2(l) => s.line_p2_on_line.iter().find(|c| c.a == l && c.b == line).map(|c| c.nid),
            EndpointRef::ArcCenter(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::Center, line),
            EndpointRef::ArcStart(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::Start, line),
            EndpointRef::ArcEnd(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::End, line),
        };
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyPointOnLine { point: p, line },
            EndpointRef::LineP1(l) => Action::ApplyLineP1OnLine { a: l, b: line },
            EndpointRef::LineP2(l) => Action::ApplyLineP2OnLine { a: l, b: line },
            EndpointRef::ArcCenter(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcCenter(a), line },
            EndpointRef::ArcStart(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcStart(a), line },
            EndpointRef::ArcEnd(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcEnd(a), line },
        };
        apply_constraint(ctx, exists, "Point-on-line", action, last_minted_nid, "point-on-line")
    } else if is_arc_name(target) {
        let arc = resolve_arc(&ctx.sketch, target)?;
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.point_on_arc.iter().find(|c| c.point == p && c.arc == arc).map(|c| c.nid),
            EndpointRef::LineP1(l) => s.line_p1_on_arc.iter().find(|c| c.line == l && c.arc == arc).map(|c| c.nid),
            EndpointRef::LineP2(l) => s.line_p2_on_arc.iter().find(|c| c.line == l && c.arc == arc).map(|c| c.nid),
            EndpointRef::ArcCenter(src) => has_arc_endpoint_on_arc(s, src, ArcEp::Center, arc),
            EndpointRef::ArcStart(src) => has_arc_endpoint_on_arc(s, src, ArcEp::Start, arc),
            EndpointRef::ArcEnd(src) => has_arc_endpoint_on_arc(s, src, ArcEp::End, arc),
        };
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyPointOnArc { point: p, arc },
            EndpointRef::LineP1(l) => Action::ApplyLineP1OnArc { line: l, arc },
            EndpointRef::LineP2(l) => Action::ApplyLineP2OnArc { line: l, arc },
            EndpointRef::ArcCenter(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcCenter(a), arc },
            EndpointRef::ArcStart(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcStart(a), arc },
            EndpointRef::ArcEnd(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcEnd(a), arc },
        };
        apply_constraint(ctx, exists, "Point-on-arc", action, last_minted_nid, "point-on-arc")
    } else {
        Err("Second arg must be a line (L0) or arc (A0)".into())
    }
}

// ---------------------------------------------------------------------------
// Additional dimensions
// ---------------------------------------------------------------------------

