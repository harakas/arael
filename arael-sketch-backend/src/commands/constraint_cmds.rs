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
    let mut sources: Vec<MetaEntity> = Vec::new();
    if source_tokens.len() == 1 && source_tokens[0] == "selection" {
        for sel in &ctx.selection {
            let e = match sel {
                // The axis itself mirrors onto its own position; a
                // selected axis (select all) is excluded, not an error.
                Selection::Line(r) if *r == mirror_line => continue,
                Selection::Line(r) => MetaEntity::Line(*r),
                Selection::Arc(r) => MetaEntity::Arc(*r),
                Selection::Point(r) => MetaEntity::Point(*r),
                _ => continue, // skip endpoint selections, constraints, dimensions
            };
            if !sources.contains(&e) {
                sources.push(e);
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
                sources.push(MetaEntity::Line(r));
            } else if is_arc_name(name) {
                sources.push(MetaEntity::Arc(resolve_arc(&ctx.sketch, name)?));
            } else if name.starts_with('P') {
                sources.push(MetaEntity::Point(resolve_point(&ctx.sketch, name)?));
            } else {
                return Err(format!("Unknown entity: {}", name).into());
            }
        }
    }

    let params = crate::mirror::MirrorParams { axis: mirror_line, noconstraint, strict };
    let plan = crate::mirror::plan(&ctx.sketch, &sources, &params)?;
    let out = crate::mirror::apply(ctx, &plan)?;

    // Set session names
    if let Some((_, first)) = out.mirrored.first() {
        ctx.session_names.insert("_".into(), first.clone());
    }
    for (i, (_, name)) in out.mirrored.iter().enumerate() {
        ctx.session_names.insert(format!("_{}", i), name.clone());
    }

    let mut msg = out.mirrored.iter()
        .map(|(src, dst)| format!("Mirrored {} -> {}", src, dst))
        .collect::<Vec<_>>()
        .join("\n");
    for a in &out.applied {
        msg += &format!("\n  {}", a);
    }
    for w in &out.warnings {
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

