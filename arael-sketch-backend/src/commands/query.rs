use super::*;

/// What a dimension measures, in command vocabulary: `radius EA0`,
/// `distance L0.p1 P0`, `angle L0 L1 supplement`.
pub(crate) fn dim_meaning(sketch: &Sketch, dim: &Dimension) -> String {
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
        K::LineLength(r) => format!("length {}", sketch.lines[r].name),
        K::LineAngle(r) => format!("xangle {}", sketch.lines[r].name),
        K::ArcRadius(r) => format!("radius {}", sketch.arcs[r].name),
        K::ArcRadiusB(r) => format!("radius_b {}", sketch.arcs[r].name),
        K::ArcSweep(r) => format!("sweep {}", sketch.arcs[r].name),
        K::ArcRotation(r) => format!("xangle {}", sketch.arcs[r].name),
        K::PointPointDistance(a, b) => format!("distance {} {}", ep_name(&a), ep_name(&b)),
        K::PointLineDistance(ep, l) => format!("distance {} {}", ep_name(&ep), sketch.lines[l].name),
        K::Angle(a, b, sup) => format!("angle {} {}{}", sketch.lines[a].name, sketch.lines[b].name, if sup { " supplement" } else { "" }),
        K::HDistance(a, b) => format!("hdistance {} {}", ep_name(&a), ep_name(&b)),
        K::VDistance(a, b) => format!("vdistance {} {}", ep_name(&a), ep_name(&b)),
        K::ConcentricDistance(a, b) => format!("distance {} {}", sketch.arcs[a].name, sketch.arcs[b].name),
        K::LineLineDistance(a, b) => format!("distance {} {}", sketch.lines[a].name, sketch.lines[b].name),
    }
}

/// One `list dims` line: name, meaning, and value -- `d0: radius EA0
/// = 3.0000`; an expression dim shows the expression with the value
/// in parentheses (`= w/2 (1.5000)`, the `list params` shape); a
/// range dim shows its bound the way it was typed (`>= 2 (5.0000)`);
/// `derived` / `broken` tags follow.
pub(crate) fn dim_line(sketch: &Sketch, dim: &Dimension) -> String {
    let meaning = dim_meaning(sketch, dim);
    let mut s = if let Some(rb) = &dim.range {
        format!("{}: {} {} ({:.4})", dim.name, meaning, format_range_bound(rb), dim.value)
    } else if let Some(expr) = &dim.expr_str {
        format!("{}: {} = {} ({:.4})", dim.name, meaning, expr, dim.value)
    } else {
        format!("{}: {} = {:.4}", dim.name, meaning, dim.value)
    };
    if dim.derived { s.push_str(" derived"); }
    if dim.broken { s.push_str(" broken"); }
    s
}

// ---------------------------------------------------------------------------
// Constraint commands
// ---------------------------------------------------------------------------


pub(crate) fn cmd_print(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let args = args.trim();
    // Try geometric coordinate function first (prints x,y)
    if let Some(result) = eval_geo_coord(&ctx.sketch, args) {
        return match result {
            Ok(v) => Ok(ok(format!("({:.6}, {:.6})", v.x, v.y))),
            Err(e) => Err(e.into()),
        };
    }
    // Try geometric scalar function
    if let Some(result) = eval_geo_scalar(&ctx.sketch, args) {
        return match result {
            Ok(v) => Ok(ok(format!("{:.6}", v))),
            Err(e) => Err(e.into()),
        };
    }
    // Try session variable
    if let Some(v) = ctx.session_vars.get(args) {
        return Ok(ok(format!("{:.6}", v)));
    }
    if let Some(v) = ctx.session_vecs.get(args) {
        return Ok(ok(format!("({:.6}, {:.6})", v.x, v.y)));
    }
    // Eval as expression
    match eval_expr_with(&ctx.sketch, args, &ctx.session_vars) {
        Ok(v) => Ok(ok(format!("{:.6}", v))),
        Err(e) => Err(format!("Eval error: {}", e).into()),
    }
}

/// Get all constraints that mention a given entity name.
pub(crate) fn constraints_for(sketch: &Sketch, name: &str) -> Vec<String> {
    sketch.list_constraints().into_iter()
        .filter(|c| mentions(c, name))
        .collect()
}

/// Get all dimensions that mention a given entity name, as `list dims`
/// lines.
pub(crate) fn dims_for(sketch: &Sketch, name: &str) -> Vec<String> {
    sketch.dimensions.iter()
        .map(|d| dim_line(sketch, d))
        .filter(|c| mentions(c, name))
        .collect()
}

/// Whether a listing line names the entity or one of its parts
/// (`L0`, `L0.p1`), as a whole word.
fn mentions(line: &str, name: &str) -> bool {
    line.split_whitespace().any(|w| w == name || w.starts_with(&format!("{}.", name)))
}

/// `info <entity>` tail: its constraints and dimensions, each on its
/// own line when present.
fn entity_refs_info(sketch: &Sketch, name: &str) -> String {
    let mut s = String::new();
    let cstrs = constraints_for(sketch, name);
    if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
    let dims = dims_for(sketch, name);
    if !dims.is_empty() { s += &format!("\n  dims: {}", dims.join(", ")); }
    s
}

/// Format a `RangeBound` for script / info output using the same
/// surface syntax that the user typed: `>= v`, `<= v`, `lo to hi`.
pub(crate) fn format_range_bound(rb: &RangeBound) -> String {
    match rb {
        RangeBound::Min(v) => format!(">= {}", v),
        RangeBound::Max(v) => format!("<= {}", v),
        RangeBound::Between(lo, hi) => format!("{} to {}", lo, hi),
    }
}

pub(crate) fn cmd_info(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let name = args.trim();
    // Constraint name lookup: C<nid> (every numeric constraint name,
    // including dimension-managed distance constraints) and
    // synthetic flag names like CL0H / CL2V.
    if name.starts_with('C') && !name.contains('.')
        && let Some(desc) = ctx.sketch.find_constraint_description(name) {
        return Ok(ok(format!("{}: {}", name, desc)));
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
            s += &entity_refs_info(&ctx.sketch, name);
            return Ok(ok(s));
        }
    if name.starts_with('L') && !name.contains('.') {
        let r = resolve_line(&ctx.sketch, name)?;
        let l = &ctx.sketch.lines[r];
        let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
        let mut s = format!("{}: ({:.4},{:.4})-({:.4},{:.4}) len={:.4} style={}",
            l.name, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y, len, l.style.name());
        if l.construction { s += " [constr]"; }
        if l.quiet { s += " [quiet]"; }
        if !l.p1.optimize { s += " [p1 locked]"; }
        if !l.p2.optimize { s += " [p2 locked]"; }
        s += &entity_refs_info(&ctx.sketch, name);
        Ok(ok(s))
    } else if name.starts_with('P') && !name.contains('.') {
        let r = resolve_point(&ctx.sketch, name)?;
        let p = &ctx.sketch.points[r];
        let locked = p.constraints.has_fix_x || p.constraints.has_fix_y;
        let mut s = format!("{}: ({:.4},{:.4}){}{}", p.name, p.pos.value.x, p.pos.value.y,
            if locked { " [locked]" } else { "" },
            if p.quiet { " [quiet]" } else { "" });
        s += &entity_refs_info(&ctx.sketch, name);
        Ok(ok(s))
    } else if is_arc_name(name) && !name.contains('.') {
        let r = resolve_arc(&ctx.sketch, name)?;
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
        s += &entity_refs_info(&ctx.sketch, name);
        Ok(ok(s))
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
        Ok(ok(format!("{}: {} value={:.4} {} offset={:.2} along={:.2}{}",
            d.name, dim_meaning(&ctx.sketch, d), d.value, source, d.offset.y, d.text_along, flags)))
    } else if let Some(p) = ctx.sketch.user_params.iter().find(|p| p.name == name) {
        // Checked after dimensions: a user param may start with 'd'.
        Ok(ok(format!("{}: value={:.4} expr={}{}", p.name, p.value, p.expr_str,
            if p.broken { " broken" } else { "" })))
    } else if name.starts_with('d') && name[1..].bytes().all(|b| b.is_ascii_digit()) {
        Err(format!("Unknown dimension: {}", name).into())
    } else {
        Err(format!("Unknown entity: {}", name).into())
    }
}

pub(crate) fn cmd_measure(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return Err("Usage: measure L0 | measure L0 L1 | measure P0 P1".into()); }

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
        let e = resolve(tokens[0])?;
        match e {
            Entity::Line(r) => {
                let l = &ctx.sketch.lines[r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx).to_degrees();
                Ok(ok(format!("{}: length={:.4}, angle={:.4} deg\n  p1=({:.4},{:.4}) p2=({:.4},{:.4})",
                    l.name, len, angle, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y)))
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
                Ok(ok(s))
            }
            Entity::Point(pos, name) => {
                Ok(ok(format!("{}: ({:.4},{:.4})", name, pos.x, pos.y)))
            }
        }
    } else if tokens.len() == 2 {
        let e1 = resolve(tokens[0])?;
        let e2 = resolve(tokens[1])?;
        match (e1, e2) {
            (Entity::Point(a, _), Entity::Point(b, _)) => {
                let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
                Ok(ok(format!("distance: {:.4}", d)))
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
                Ok(ok(lines.join("\n")))
            }
            (Entity::Point(p, _), Entity::Line(r)) | (Entity::Line(r), Entity::Point(p, _)) => {
                let l = &ctx.sketch.lines[r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let perp_dist = if len > 1e-12 {
                    ((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx).abs() / len
                } else { 0.0 };
                Ok(ok(format!("perpendicular distance: {:.4}", perp_dist)))
            }
            (Entity::Point(p, _), Entity::Arc(r)) | (Entity::Arc(r), Entity::Point(p, _)) => {
                let a = &ctx.sketch.arcs[r];
                let dc = ((p.x - a.center.value.x).powi(2) + (p.y - a.center.value.y).powi(2)).sqrt();
                let dist_to_arc = (dc - a.radius.value).abs();
                Ok(ok(format!("distance to center: {:.4}, distance to arc: {:.4}", dc, dist_to_arc)))
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
                Ok(ok(lines.join("\n")))
            }
            (Entity::Arc(a), Entity::Arc(b)) => {
                let aa = &ctx.sketch.arcs[a];
                let ab = &ctx.sketch.arcs[b];
                let dc = ((aa.center.value.x - ab.center.value.x).powi(2) +
                          (aa.center.value.y - ab.center.value.y).powi(2)).sqrt();
                Ok(ok(format!("center-to-center: {:.4}, radii: {:.4} + {:.4} = {:.4}, gap: {:.4}",
                    dc, aa.radius.value, ab.radius.value,
                    aa.radius.value + ab.radius.value,
                    dc - aa.radius.value - ab.radius.value)))
            }
        }
    } else {
        Err("Usage: measure L0 | measure L0 L1 | measure P0 P1".into())
    }
}

/// Resolve endpoint ref to position.
pub(crate) fn resolve_endpoint_pos_from_ref(sketch: &Sketch, ep: &EndpointRef) -> vect2d {
    match ep {
        EndpointRef::Point(r) => sketch.points[*r].pos.value,
        EndpointRef::LineP1(r) => sketch.lines[*r].p1.value,
        EndpointRef::LineP2(r) => sketch.lines[*r].p2.value,
        EndpointRef::ArcCenter(r) => sketch.arcs[*r].center.value,
        EndpointRef::ArcStart(r) => crate::geometry::arc_start_pos(&sketch.arcs[*r]),
        EndpointRef::ArcEnd(r) => crate::geometry::arc_end_pos(&sketch.arcs[*r]),
    }
}

pub(crate) fn cmd_list(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let filter = args.trim();
    if filter == "selection" {
        if ctx.selection.is_empty() { return Ok(ok("(no selection)")); }
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
        return Ok(ok(names.join(", ")));
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
        return if filtered.is_empty() { Ok(ok("(empty)")) } else { Ok(ok(filtered.join("\n"))) };
    }
    if DIMENSION_FILTERS.contains(&filter) {
        // Dimension kinds live in the dims listing (`d<n>: radius A0
        // = 2`); an orphan value flag with no dimension still shows
        // in list_constraints, so search both.
        let mut all: Vec<String> = ctx.sketch.dimensions.iter().map(|d| dim_line(&ctx.sketch, d)).collect();
        all.extend(ctx.sketch.list_constraints());
        let filtered: Vec<String> = all.into_iter().filter(|s| body_matches(s, filter)).collect();
        return if filtered.is_empty() { Ok(ok("(empty)")) } else { Ok(ok(filtered.join("\n"))) };
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
        return if items.is_empty() { Ok(ok("(no construction entities)")) } else { Ok(ok(items.join("\n"))) };
    }
    if !filter.is_empty() && !matches!(filter, "all" | "lines" | "points" | "arcs" | "dims" | "params" | "constraints") {
        return Err(format!("Unknown filter: {}. Use: all, lines, points, arcs, dims, params, constraints, constr, selection, or a constraint type (horizontal, parallel, ...)", filter).into());
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
            lines.push(dim_line(&ctx.sketch, d));
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
        Ok(ok("(empty)"))
    } else {
        Ok(ok(lines.join("\n")))
    }
}

// ---------------------------------------------------------------------------
// Undo/Redo/History
// ---------------------------------------------------------------------------

