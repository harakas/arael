use super::*;
use crate::corner_ops::{apply_corner_ops, resolve_corner_spec, CornerKind, CornerOpConfig, CornerSpec};

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

/// Parse a single corner spec (list of 1 or 2 tokens) into a typed
/// [`CornerSpec`]. Used by fillet and chamfer to accept any mix of
/// `L1 L2` and `L1.pN` args on one command line; the shared-corner
/// lookup happens in `resolve_corner_spec` at apply time.
pub(crate) fn resolve_corner_tokens(sketch: &Sketch, tokens: &[String]) -> Result<CornerSpec, String> {
    match tokens.len() {
        1 => {
            let ep = resolve_endpoint_ref(sketch, &tokens[0])?;
            match ep {
                EndpointRef::LineP1(l) => Ok(CornerSpec::Endpoint { line: l, is_p1: true }),
                EndpointRef::LineP2(l) => Ok(CornerSpec::Endpoint { line: l, is_p1: false }),
                _ => Err(format!("endpoint must be a line end: {}", tokens[0])),
            }
        }
        2 => {
            let la = resolve_line(sketch, &tokens[0])?;
            let lb = resolve_line(sketch, &tokens[1])?;
            Ok(CornerSpec::Lines(la, lb))
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
pub fn parse_radius_token(sketch: &Sketch, tok: &str) -> Result<(f64, Option<String>), String> {
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

    let corner_tokens = parse_corner_list(&tokens).map_err(|e| format!("fillet: {}", e))?;
    if corner_tokens.is_empty() {
        return Err("fillet: need at least one corner".into());
    }
    run_corner_command(ctx, CornerKind::Fillet, &corner_tokens, radius, radius_expr,
        notangent, noradius)
}

/// Shared command tail for fillet and chamfer: parse each corner's
/// tokens, validate every corner up front (clearly-bad commands fail
/// before any mutation), run [`apply_corner_ops`], and format the
/// per-corner id report.
fn run_corner_command(
    ctx: &mut CommandContext,
    kind: CornerKind,
    corner_tokens: &[Vec<String>],
    radius: f64,
    radius_expr: Option<String>,
    notangent: bool,
    noradius: bool,
) -> CmdResult {
    let verb = match kind { CornerKind::Fillet => "fillet", CornerKind::Chamfer => "chamfer" };
    let mut corners: Vec<CornerSpec> = Vec::with_capacity(corner_tokens.len());
    for toks in corner_tokens {
        match resolve_corner_tokens(&ctx.sketch, toks) {
            Ok(spec) => {
                // Corner-connectivity validation up front; the apply
                // loop re-resolves because each corner op deletes one
                // LL coincident.
                if let Err(e) = resolve_corner_spec(&ctx.sketch, &spec) {
                    return Err(format!("{} {}: {}", verb, toks.join(" "), e));
                }
                corners.push(spec);
            }
            Err(e) => return Err(format!("{} {}: {}", verb, toks.join(" "), e)),
        }
    }

    let cfg = CornerOpConfig { kind, radius, radius_expr, notangent, noradius };
    let result = apply_corner_ops(ctx, &cfg, &corners);

    // Format one line per corner: the spec, the new entity, the dims,
    // the deleted coincident and the set of added constraint ids.
    let mut lines: Vec<String> = Vec::with_capacity(result.outcomes.len());
    for (toks, out) in corner_tokens.iter().zip(&result.outcomes) {
        let spec_str = toks.join(" ");
        if let Some(e) = &out.error {
            lines.push(format!("  {}: FAILED: {}", spec_str, e));
            continue;
        }
        let mut parts = Vec::new();
        parts.push(out.entity_name.clone());
        if let Some(p) = &out.point_name { parts.push(p.clone()); }
        if let Some(d) = &out.primary_dim { parts.push(d.clone()); }
        if let Some(d) = &out.secondary_dim { parts.push(d.clone()); }
        let mut tail = Vec::new();
        if let Some(r) = &out.removed { tail.push(format!("removed {}", r)); }
        if !out.added.is_empty() { tail.push(format!("added {}", out.added.join(" "))); }
        let tail_str = if tail.is_empty() { String::new() } else { format!(" [{}]", tail.join(", ")) };
        lines.push(format!("  {} -> {}{}", spec_str, parts.join(" "), tail_str));
    }
    let succeeded = result.outcomes.iter().filter(|o| o.error.is_none()).count();
    let (verb_done, unit) = match kind {
        CornerKind::Fillet => ("Filleted", "r"),
        CornerKind::Chamfer => ("Chamfered", "d"),
    };
    let header = if corner_tokens.len() == 1 {
        format!("{} ({}={:.4}):", verb_done, unit, radius)
    } else {
        format!("{} {} of {} corners ({}={:.4}):",
            verb_done, succeeded, corner_tokens.len(), unit, radius)
    };
    if let Some(last) = result.outcomes.iter().rev().find(|o| o.error.is_none()) {
        ctx.session_names.insert("_".into(), last.entity_name.clone());
    }
    let msg = format!("{}\n{}", header, lines.join("\n"));
    if succeeded == 0 { Err(msg) } else { Ok(ok(msg)) }
}
pub(crate) fn cmd_chamfer(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() < 2 {
        return Err("Usage: chamfer <corner>... d  where each corner is Lx.pN or Lx Ly".into());
    }
    let dist_tok = tokens.pop().unwrap();
    let (distance, dist_expr) = parse_radius_token(&ctx.sketch, dist_tok)?;
    if distance <= 1e-9 { return Err("chamfer distance must be positive".into()); }

    let corner_tokens = parse_corner_list(&tokens).map_err(|e| format!("chamfer: {}", e))?;
    if corner_tokens.is_empty() { return Err("chamfer: need at least one corner".into()); }
    run_corner_command(ctx, CornerKind::Chamfer, &corner_tokens, distance, dist_expr,
        false, false)
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

