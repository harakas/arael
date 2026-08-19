use super::*;

/// Parse a dimension value string. Returns (numeric_value, expression_string).
/// - `=expr` or `{expr}` → live expression: (0.0, Some("expr"))
/// - `"expr"` (quoted) → live expression (backwards compat): (0.0, Some("expr"))
/// - numeric literal → (value, None)
/// - anything else → evaluate as expression to number: (value, None)
pub(crate) fn parse_dim_value(sketch: &Sketch, val_str: &str) -> Result<(f64, Option<String>), String> {
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
pub(crate) fn find_existing_dimension(sketch: &Sketch, kind: &DimensionKind) -> Option<usize> {
    sketch.dimensions.iter().position(|d| match (&d.kind, kind) {
        (DimensionKind::Angle(da, db, _), DimensionKind::Angle(a, b, _)) =>
            (*da == *a && *db == *b) || (*da == *b && *db == *a),
        (DimensionKind::PointPointDistance(a1, b1), DimensionKind::PointPointDistance(a2, b2)) =>
            (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2),
        (a, b) => a == b,
    })
}

/// Sentence form of a range bound: "in LO to HI" for a Between, else
/// the comparison from format_range_bound.
fn range_phrase(rb: &RangeBound) -> String {
    match rb {
        RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
        _ => format_range_bound(rb),
    }
}

/// One dimension command's variabilia for `dim_command_tail`: the
/// operands are already resolved and measured at the call site.
struct DimTail<'a> {
    kind: DimensionKind,
    measured: f64,
    noun: &'a str,
    subject: String,
    usage: &'a str,
}

/// Shared skeleton of the dimension commands. Three forms, each
/// updating an existing dimension of the same kind in place:
/// - bare derived/driven (no value tokens): dimension at the current
///   measurement;
/// - range (`>=V | <=V | LO to HI`);
/// - plain value / expression.
fn dim_command_tail(ctx: &mut CommandContext, t: DimTail, val_tokens: &[&str],
                    is_derived: bool, is_driven: bool) -> CmdResult {
    if val_tokens.is_empty() && (is_derived || is_driven) {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &t.kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: t.measured, expr: None, range: None });
            let label = if is_derived { "derived" } else { "driven" };
            return Ok(ok_or_status(ctx, format!("Updated {} {} {} = ({:.4})", label, name, t.noun, t.measured)));
        }
        ctx.exec(Action::AddDimension { kind: t.kind, value: t.measured, expr: None, derived: is_derived, range: None });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return Ok(ok_or_status(ctx, format!("{} {} {} {} = ({:.4})", label, dim_name, t.noun, t.subject, t.measured)));
    }
    let range_opt = if !val_tokens.is_empty() && !is_derived && !is_driven {
        parse_range_tokens(&ctx.sketch, val_tokens)?
    } else { None };
    if let Some(rb) = range_opt {
        let bound_desc = range_phrase(&rb);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &t.kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: t.measured, expr: None, range: Some(rb) });
            return Ok(ok_or_status(ctx, format!("Updated {} {} {} (current {:.4})", name, t.noun, bound_desc, t.measured)));
        }
        ctx.exec(Action::AddDimension { kind: t.kind, value: t.measured, expr: None, derived: false, range: Some(rb) });
        let dim_name = last_dim_name(ctx);
        return Ok(ok_or_status(ctx, format!("Set {} {} {} {} (current {:.4})", dim_name, t.noun, t.subject, bound_desc, t.measured)));
    }
    if val_tokens.len() != 1 { return Err(t.usage.into()); }
    let (value, expr) = parse_dim_value(&ctx.sketch, val_tokens[0])?;
    let display = if expr.is_some() { val_tokens[0].to_string() } else { format!("{}", value) };
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &t.kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value, expr, range: None });
        return Ok(ok_or_status(ctx, format!("Updated {} {} = {}", name, t.noun, display)));
    }
    ctx.exec(Action::AddDimension { kind: t.kind, value, expr, derived: is_derived, range: None });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    Ok(ok_or_status(ctx, format!("{} {} {} {} = {}", prefix, dim_name, t.noun, t.subject, display)))
}

pub(crate) fn cmd_length(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let usage = "Usage: length L0 5.0 [derived|driven]  or  length L0 >=V | <=V | LO to HI";
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [is_derived, is_driven] = peel_keywords(&mut tokens, ["derived", "driven"]);
    if tokens.is_empty() { return Err(usage.into()); }
    let line = resolve_line(&ctx.sketch, tokens[0])?;
    let l = &ctx.sketch.lines[line];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let tail = DimTail {
        kind: DimensionKind::LineLength(line),
        measured: (dx * dx + dy * dy).sqrt(),
        noun: "length",
        subject: tokens[0].to_string(),
        usage,
    };
    dim_command_tail(ctx, tail, &tokens[1..], is_derived, is_driven)
}

pub(crate) fn cmd_radius(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let usage = "Usage: radius A0 1.5 [derived|driven]  or  radius A0 >=V | <=V | LO to HI";
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [is_derived, is_driven] = peel_keywords(&mut tokens, ["derived", "driven"]);
    if tokens.is_empty() { return Err(usage.into()); }
    let arc = resolve_arc(&ctx.sketch, tokens[0])?;
    let tail = DimTail {
        kind: DimensionKind::ArcRadius(arc),
        measured: ctx.sketch.arcs[arc].radius.value,
        noun: "radius",
        subject: tokens[0].to_string(),
        usage,
    };
    dim_command_tail(ctx, tail, &tokens[1..], is_derived, is_driven)
}

pub(crate) fn cmd_radius_b(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let usage = "Usage: radius_b A0 1.5 [derived|driven]  or  radius_b A0 >=V | <=V | LO to HI";
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [is_derived, is_driven] = peel_keywords(&mut tokens, ["derived", "driven"]);
    if tokens.is_empty() { return Err(usage.into()); }
    let arc = resolve_arc(&ctx.sketch, tokens[0])?;
    if !ctx.sketch.arcs[arc].is_ellipse {
        return Err("radius_b only applies to ellipses (use add_ellipse to create one)".into());
    }
    let tail = DimTail {
        kind: DimensionKind::ArcRadiusB(arc),
        measured: ctx.sketch.arcs[arc].radius_b.value,
        noun: "radius_b",
        subject: tokens[0].to_string(),
        usage,
    };
    dim_command_tail(ctx, tail, &tokens[1..], is_derived, is_driven)
}

pub(crate) fn cmd_sweep(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let usage = "Usage: sweep A0 180 [derived|driven]  or  sweep A0 >=V | <=V | LO to HI";
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [is_derived, is_driven] = peel_keywords(&mut tokens, ["derived", "driven"]);
    if tokens.is_empty() { return Err(usage.into()); }
    let arc = resolve_arc(&ctx.sketch, tokens[0])?;
    if ctx.sketch.arcs[arc].closed {
        return Err("Cannot set sweep on a full circle (angles are fixed)".into());
    }
    let a = &ctx.sketch.arcs[arc];
    let tail = DimTail {
        kind: DimensionKind::ArcSweep(arc),
        measured: arael::utils::rad2deg((a.end_angle.value - a.start_angle.value).abs()),
        noun: "sweep",
        subject: tokens[0].to_string(),
        usage,
    };
    dim_command_tail(ctx, tail, &tokens[1..], is_derived, is_driven)
}

// ---------------------------------------------------------------------------
// Lock/unlock
// ---------------------------------------------------------------------------


pub(crate) fn cmd_angle(ctx: &mut CommandContext, args: &str) -> CmdResult {
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
        let a = resolve_line(&ctx.sketch, tokens[0])?;
        let b = resolve_line(&ctx.sketch, tokens[1])?;
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
            return Ok(ok_or_status(ctx, format!("Updated {} {} angle = ({:.4})", label, name, val)));
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let label = if is_derived { "Derived" } else { "Driven" };
        return Ok(ok_or_status(ctx, format!("{} {} angle {} {} = ({:.4})", label, dim_name, tokens[0], tokens[1], val)));
    }

    // Range form: `angle L0 L1 >= V | <= V | LO to HI [supplement]`.
    // `closest` / `acute` / `obtuse` are value-selection heuristics
    // and are rejected with a range (no single target value).
    let range_opt = if tokens.len() >= 3 && !is_derived && !is_driven {
        parse_range_tokens(&ctx.sketch, &tokens[2..])?
    } else { None };
    if let Some(rb) = range_opt {
        if matches!(sector_mode, SectorMode::Closest | SectorMode::Acute | SectorMode::Obtuse) {
            return Err("closest / acute / obtuse require a specific target angle; use a bare value or `supplement`".into());
        }
        let a = resolve_line(&ctx.sketch, tokens[0])?;
        let b = resolve_line(&ctx.sketch, tokens[1])?;
        let (current_deg, supplement_deg) = compute_angle(ctx, a, b);
        let supplement = matches!(sector_mode, SectorMode::Supplement);
        let measured = if supplement { supplement_deg } else { current_deg };
        let kind = DimensionKind::Angle(a, b, supplement);
        let bound_desc = range_phrase(&rb);
        let sector = if supplement { " (supplement)" } else { "" };
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: measured, expr: None, range: Some(rb) });
            return Ok(ok_or_status(ctx, format!("Updated {} angle {}{} (current {:.4})", name, bound_desc, sector, measured)));
        }
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return Ok(ok_or_status(ctx, format!("Set {} angle {} {} {}{} (current {:.4})",
            dim_name, tokens[0], tokens[1], bound_desc, sector, measured)));
    }

    if tokens.len() != 3 { return Err("Usage: angle L0 L1 45 [supplement|closest|acute|obtuse] [derived|driven]  or  angle L0 L1 >=V | <=V | LO to HI [supplement]".into()); }
    let a = resolve_line(&ctx.sketch, tokens[0])?;
    let b = resolve_line(&ctx.sketch, tokens[1])?;
    let (value, expr) = parse_dim_value(&ctx.sketch, tokens[2])?;
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
        return Ok(ok_or_status(ctx, format!("Updated {} angle = {} {}", name, display, sector).trim_end().to_string()));
    }
    ctx.exec(Action::AddDimension { kind, value, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let sector = if supplement { " (supplement)" } else { "" };
    let prefix = if is_derived { "Derived" } else { "Set" };
    Ok(ok_or_status(ctx, format!("{} {} angle {} {} = {}{}", prefix, dim_name, tokens[0], tokens[1], display, sector)))
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

pub(crate) fn cmd_distance(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    let is_driven = !is_derived && tokens.last() == Some(&"driven");
    if is_derived || is_driven { tokens.pop(); }


    // "distance L0.p1 L1.p2 derived/driven" or "distance P0 L0 derived/driven" — measure-only, no value
    if tokens.len() == 2 && (is_derived || is_driven) {
        let label = if is_derived { "Derived" } else { "Driven" };
        // Try point-line distance first
        if (tokens[0].starts_with('P') || tokens[0].contains('.')) && tokens[1].starts_with('L') && !tokens[1].contains('.') {
            let ep = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
            let line = resolve_line(&ctx.sketch, tokens[1])?;
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
            return Ok(ok_or_status(ctx, format!("{} {} distance {} {} = ({:.4})", label, dim_name, tokens[0], tokens[1], dist)));
        }
        // Point-point distance
        let ep_a = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
        let ep_b = resolve_endpoint_ref(&ctx.sketch, tokens[1])?;
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
            return Ok(ok_or_status(ctx, format!("Updated {} {} distance = ({:.4})", name, ulabel, dist)));
        }
        ctx.exec(Action::AddDimension { kind, value: dist, expr: None, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        return Ok(ok_or_status(ctx, format!("{} {} distance {} {} = ({:.4})", label, dim_name, tokens[0], tokens[1], dist)));
    }

    // Range-dimension form: two entities + one of `>=V`, `<=V`,
    // `>= V`, `<= V`, or `LO to HI`. Each value is a numeric
    // literal, evaluate-once expression, or live expression
    // (`=expr` / `{expr}`). Incompatible with derived/driven (a
    // range isn't a measure-only or value-capture dimension).
    let range_opt = if tokens.len() >= 3 && !is_derived && !is_driven {
        parse_range_tokens(&ctx.sketch, &tokens[2..])?
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
                let a = resolve_line(&ctx.sketch, tokens[0])?;
                let b = resolve_line(&ctx.sketch, tokens[1])?;
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
                let ep = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
                let line = resolve_line(&ctx.sketch, tokens[1])?;
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
                && is_arc_name(tokens[0]) && is_arc_name(tokens[1])
                && let Ok(arc_a) = resolve_arc(&ctx.sketch, tokens[0])
                && let Ok(arc_b) = resolve_arc(&ctx.sketch, tokens[1])
                && arc_a != arc_b
                && ctx.sketch.arcs[arc_a].is_ellipse == ctx.sketch.arcs[arc_b].is_ellipse
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
                let ep_a = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
                let ep_b = resolve_endpoint_ref(&ctx.sketch, tokens[1])?;
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
        let bound_desc = range_phrase(&rb);
        ctx.exec(Action::AddDimension {
            kind, value: measured, expr: None, derived: false, range: Some(rb),
        });
        let dim_name = last_dim_name(ctx);
        return Ok(ok_or_status(ctx, format!(
            "Set {} distance {} {} {} (current {:.4})",
            dim_name, tokens[0], tokens[1], bound_desc, measured)));
    }

    if tokens.len() != 3 { return Err("Usage: distance L0.p1 L1.p2 5.0 [derived|driven]  or  distance P0 L0 3.0 [derived|driven]  or  distance A0 A1 5.0 (concentric circles)  or  distance L0 L1 5.0 (two lines; also applies a Parallel constraint)  or  distance <entities> >=V | <=V | LO to HI (range bound)".into()); }
    let (val, expr) = parse_dim_value(&ctx.sketch, tokens[2])?;

    // Two-line perpendicular distance: `distance L0 L1 v` for two bare
    // line names. Also applies a Parallel constraint between them -- the
    // gap between non-parallel lines is ill-defined, so the CLI makes
    // the pairing explicit. Idempotent when Parallel is already there.
    if !tokens[0].contains('.') && !tokens[1].contains('.')
        && tokens[0].starts_with('L') && tokens[1].starts_with('L')
    {
        let line_a = resolve_line(&ctx.sketch, tokens[0])?;
        let line_b = resolve_line(&ctx.sketch, tokens[1])?;
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
            return Ok(ok_or_status(ctx, format!("Updated {} line-line distance = {}", name, tokens[2])));
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
        return Ok(ok_or_status(ctx, format!("{} {} line-line distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2])));
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
        && is_arc_name(tokens[0]) && is_arc_name(tokens[1])
        && let Ok(arc_a) = resolve_arc(&ctx.sketch, tokens[0])
        && let Ok(arc_b) = resolve_arc(&ctx.sketch, tokens[1])
        && arc_a != arc_b
        && ctx.sketch.arcs[arc_a].is_ellipse == ctx.sketch.arcs[arc_b].is_ellipse
        && arcs_are_concentric(&ctx.sketch, arc_a, arc_b)
    {
        let kind = DimensionKind::ConcentricDistance(arc_a, arc_b);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
            return Ok(ok_or_status(ctx, format!("Updated {} concentric distance = {}", name, tokens[2])));
        }
        let already_concentric = ctx.sketch.concentric.iter().any(|c|
            (c.a == arc_a && c.b == arc_b) || (c.a == arc_b && c.b == arc_a));
        if !already_concentric {
            ctx.exec(Action::ApplyConcentric { a: arc_a, b: arc_b });
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let prefix = if is_derived { "Derived" } else { "Set" };
        return Ok(ok_or_status(ctx, format!("{} {} concentric distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2])));
    }

    // Try point-line distance
    if (tokens[0].starts_with('P') || tokens[0].contains('.')) && tokens[1].starts_with('L') && !tokens[1].contains('.') {
        let ep = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
        let line = resolve_line(&ctx.sketch, tokens[1])?;
        let kind = DimensionKind::PointLineDistance(to_dim_endpoint(ep), line);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
            return Ok(ok_or_status(ctx, format!("Updated {} distance = {}", name, tokens[2])));
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
        let dim_name = last_dim_name(ctx);
        let prefix = if is_derived { "Derived" } else { "Set" };
        return Ok(ok_or_status(ctx, format!("{} {} distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2])));
    }

    // Point-point distance
    let ep_a = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
    let ep_b = resolve_endpoint_ref(&ctx.sketch, tokens[1])?;
    let kind = DimensionKind::PointPointDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b));
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { did: ctx.sketch.dimensions[idx].did, value: val, expr, range: None,  });
        return Ok(ok_or_status(ctx, format!("Updated {} distance = {}", name, tokens[2])));
    }
    ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived, range: None,  });
    let dim_name = last_dim_name(ctx);
    let prefix = if is_derived { "Derived" } else { "Set" };
    Ok(ok_or_status(ctx, format!("{} {} distance {} {} = {}", prefix, dim_name, tokens[0], tokens[1], tokens[2])))
}

pub(crate) fn cmd_hdistance(ctx: &mut CommandContext, args: &str) -> CmdResult {
    cmd_axis_distance(ctx, args, true)
}

pub(crate) fn cmd_vdistance(ctx: &mut CommandContext, args: &str) -> CmdResult {
    cmd_axis_distance(ctx, args, false)
}

pub(crate) fn cmd_axis_distance(ctx: &mut CommandContext, args: &str, horizontal: bool) -> CmdResult {
    let label = if horizontal { "hdistance" } else { "vdistance" };
    let usage = format!("Usage: {} L0.p1 L1.p2 5 [derived|driven]  or  {} <a> <b> >=V | <=V | LO to HI", label, label);
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [is_derived, is_driven] = peel_keywords(&mut tokens, ["derived", "driven"]);
    if tokens.len() < 2 { return Err(usage); }
    let ep_a = resolve_endpoint_ref(&ctx.sketch, tokens[0])?;
    let ep_b = resolve_endpoint_ref(&ctx.sketch, tokens[1])?;
    let pa = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
    let pb = resolve_endpoint_pos(&ctx.sketch, tokens[1]).unwrap();
    let tail = DimTail {
        kind: if horizontal { DimensionKind::HDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) }
              else { DimensionKind::VDistance(to_dim_endpoint(ep_a), to_dim_endpoint(ep_b)) },
        measured: if horizontal { (pa.x - pb.x).abs() } else { (pa.y - pb.y).abs() },
        noun: label,
        subject: format!("{} {}", tokens[0], tokens[1]),
        usage: &usage,
    };
    dim_command_tail(ctx, tail, &tokens[2..], is_derived, is_driven)
}

/// Resolve the entity argument of `xangle` to either a line (angle
/// of the line from the x-axis) or an ellipse (rotation of the
/// major axis). Rejects circular arcs, whose rotation is a fixed
/// Param::fixed(0.0).
pub(crate) fn resolve_xangle_target(
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

pub(crate) fn cmd_xangle(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let usage = "Usage: xangle L0 45 [derived|driven]  or  xangle A0 45 [derived|driven]  or  xangle L0 >=V | <=V | LO to HI";
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [is_derived, is_driven] = peel_keywords(&mut tokens, ["derived", "driven"]);
    if tokens.is_empty() { return Err(usage.into()); }
    let (kind, measured) = resolve_xangle_target(&ctx.sketch, tokens[0])?;
    let tail = DimTail {
        kind,
        measured,
        noun: "xangle",
        subject: tokens[0].to_string(),
        usage,
    };
    dim_command_tail(ctx, tail, &tokens[1..], is_derived, is_driven)
}

