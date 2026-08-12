use super::*;

pub(crate) fn cmd_add_arc(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nocursor, noconnect, quiet, constr, notangent, driven] = peel_keywords(&mut tokens,
        ["nocursor", "noconnect", "quiet", "constr", "notangent", "driven"]);
    if tokens.len() != 3 { return Err("Usage: add_arc x1,y1 x2,y2 xm,ym [noconnect] [notangent] [nocursor] [driven]".into()); }
    let p1 = parse_coord(ctx, tokens[0], ctx.cursor)?;
    let p2 = parse_coord(ctx, tokens[1], Some(p1))?;
    let pm = parse_coord(ctx, tokens[2], None)?;
    ctx.begin_group();
    let Some(arc_ref) = ctx.exec(Action::AddArc { start: p1, end: p2, mid: pm }).arc() else {
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
    Ok(ok(msg))
}

/// Scan the four line-line coincident collections for a constraint
/// involving `(line, is_p1)`. Returns the partner line, which of its
/// endpoints is the shared corner, and the ConstraintId so the caller
/// can delete the coincidence during a fillet or similar topology
/// edit. Returns None if the endpoint isn't coincident with any other
/// line's endpoint directly -- shared-via-helper-point corners are
/// rejected on purpose, keeping the fillet scope tractable.
pub(crate) fn find_ll_coincident_partner(
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
pub(crate) struct CornerRefs {
    line_a: Ref<Line>,
    is_p1_a: bool,
    line_b: Ref<Line>,
    is_p1_b: bool,
    coincident_id: crate::ids::ConstraintId,
}

/// Resolve a single corner spec (list of 1 or 2 tokens) against the
/// sketch. Used by fillet and chamfer to accept any mix of
/// `L1 L2` and `L1.pN` args on one command line.
pub(crate) fn resolve_corner_tokens(sketch: &Sketch, tokens: &[String]) -> Result<CornerRefs, String> {
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
pub(crate) fn parse_corner_list(tokens: &[&str]) -> Result<Vec<Vec<String>>, String> {
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
pub(crate) fn parse_radius_token(sketch: &Sketch, tok: &str) -> Result<(f64, Option<String>), String> {
    match parse_dim_value(sketch, tok)? {
        (v, None) => Ok((v, None)),
        (_, Some(expr)) => {
            let v = eval_expr(sketch, &expr)
                .map_err(|e| format!("Cannot evaluate '{}': {}", expr, e))?;
            Ok((v, Some(expr)))
        }
    }
}

pub(crate) fn cmd_fillet(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [notangent, noradius] = peel_keywords(&mut tokens, ["notangent", "noradius"]);
    if tokens.len() < 2 {
        return Err("Usage: fillet <corner>... r [notangent] [noradius]  where each corner is Lx.pN or Lx Ly".into());
    }
    let radius_tok = tokens.pop().unwrap();
    let (radius, radius_expr) = parse_radius_token(&ctx.sketch, radius_tok)?;
    if radius <= 1e-9 { return Err("fillet radius must be positive".into()); }

    let corner_specs = parse_corner_list(&tokens).map_err(|e| format!("fillet: {}", e))?;
    if corner_specs.is_empty() {
        return Err("fillet: need at least one corner".into());
    }

    // Validate once up front so clearly-bad commands fail before any
    // mutation. The result is discarded; the actual fillet loop
    // re-resolves each spec after the previous fillet because
    // coincident-collection indices shift when an LL coincident is
    // removed.
    for spec in &corner_specs {
        if let Err(e) = resolve_corner_tokens(&ctx.sketch, spec) {
            return Err(format!("fillet {}: {}", spec.join(" "), e).into());
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
    if succeeded == 0 { Err(msg.into()) } else { Ok(ok(msg)) }
}

/// Apply one fillet at the resolved corner. Returns the ids of every
/// new constraint and the deleted coincident so the caller can
/// surface them in the command output. Returns Err with a
/// human-readable string for geometry errors (too short, zero-angle,
/// etc.) without mutating the sketch.
pub(crate) fn apply_one_fillet(
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
pub(crate) struct FilletOut {
    spec: Vec<String>,
    arc_name: String,
    removed: Option<String>,
    dim_name: Option<String>,
    added: Vec<String>,
}

pub(crate) fn cmd_chamfer(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err("Usage: chamfer <corner>... d  where each corner is Lx.pN or Lx Ly".into());
    }
    let dist_tok = tokens.pop().unwrap();
    let (distance, dist_expr) = parse_radius_token(&ctx.sketch, dist_tok)?;
    if distance <= 1e-9 { return Err("chamfer distance must be positive".into()); }

    let corner_specs = parse_corner_list(&tokens).map_err(|e| format!("chamfer: {}", e))?;
    if corner_specs.is_empty() { return Err("chamfer: need at least one corner".into()); }

    // Validate once up front; re-resolve inside the loop because
    // coincident-collection indices shift when an LL coincident is
    // removed by a prior chamfer.
    for spec in &corner_specs {
        if let Err(e) = resolve_corner_tokens(&ctx.sketch, spec) {
            return Err(format!("chamfer {}: {}", spec.join(" "), e).into());
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
    if succeeded == 0 { Err(msg.into()) } else { Ok(ok(msg)) }
}

pub(crate) struct ChamferOut {
    spec: Vec<String>,
    new_line_name: String,
    point_name: String,
    removed: Option<String>,
    primary_dim: Option<String>,
    secondary_dim: Option<String>,
    added: Vec<String>,
}

pub(crate) fn apply_one_chamfer(
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

pub(crate) fn cmd_offset_line(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return Err("Usage: offset_line L0 distance".into()); }
    let line = resolve_line(&ctx.sketch, tokens[0])?;
    let d = eval_expr(&ctx.sketch, tokens[1])?;
    let l = &ctx.sketch.lines[line];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 { return Err("Zero-length line".into()); }
    let nx = -dy / len * d;
    let ny = dx / len * d;
    let p1 = vect2d::new(l.p1.value.x + nx, l.p1.value.y + ny);
    let p2 = vect2d::new(l.p2.value.x + nx, l.p2.value.y + ny);
    ctx.begin_group();
    ctx.exec(Action::AddLine { p1, p2 });
    let name = ctx.sketch.lines.refs().last().map(|r| ctx.sketch.lines[r].name.clone()).unwrap_or_default();
    ctx.cursor = Some(p2);
    ctx.session_names.insert("_".into(), name.clone());
    Ok(ok(format!("Added {} (offset of {} by {})", name, tokens[0], d)))
}

// ---------------------------------------------------------------------------
// Additional constraints
// ---------------------------------------------------------------------------

