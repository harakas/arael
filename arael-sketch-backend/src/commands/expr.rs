use super::*;

pub(crate) fn eval_context(sketch: &Sketch) -> HashMap<String, f64> {
    let mut vars = HashMap::new();
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        vars.insert(format!("{}.x", p.name), p.pos.value.x);
        vars.insert(format!("{}.y", p.name), p.pos.value.y);
        vars.insert(format!("{}.pos.x", p.name), p.pos.value.x);
        vars.insert(format!("{}.pos.y", p.name), p.pos.value.y);
    }
    for r in sketch.lines.refs() {
        let l = &sketch.lines[r];
        vars.insert(format!("{}.p1.x", l.name), l.p1.value.x);
        vars.insert(format!("{}.p1.y", l.name), l.p1.value.y);
        vars.insert(format!("{}.p2.x", l.name), l.p2.value.x);
        vars.insert(format!("{}.p2.y", l.name), l.p2.value.y);
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        vars.insert(format!("{}.length", l.name), (dx*dx + dy*dy).sqrt());
        vars.insert(format!("{}.angle", l.name), dy.atan2(dx).to_degrees());
    }
    for r in sketch.arcs.refs() {
        let a = &sketch.arcs[r];
        vars.insert(format!("{}.center.x", a.name), a.center.value.x);
        vars.insert(format!("{}.center.y", a.name), a.center.value.y);
        vars.insert(format!("{}.radius", a.name), a.radius.value);
        vars.insert(format!("{}.radius_b", a.name), a.radius_b.value);
        vars.insert(format!("{}.rotation", a.name), a.rotation.value);
        vars.insert(format!("{}.diameter", a.name), a.radius.value * 2.0);
        vars.insert(format!("{}.start_angle", a.name), a.start_angle.value);
        vars.insert(format!("{}.end_angle", a.name), a.end_angle.value);
        vars.insert(format!("{}.sweep", a.name), (a.end_angle.value - a.start_angle.value).abs().to_degrees());
        let sp = crate::geometry::arc_start_pos(a);
        let ep = crate::geometry::arc_end_pos(a);
        vars.insert(format!("{}.start.x", a.name), sp.x);
        vars.insert(format!("{}.start.y", a.name), sp.y);
        vars.insert(format!("{}.end.x", a.name), ep.x);
        vars.insert(format!("{}.end.y", a.name), ep.y);
    }
    for d in &sketch.dimensions {
        vars.insert(d.name.clone(), d.value);
    }
    for p in &sketch.user_params {
        vars.insert(p.name.clone(), p.value);
    }
    vars
}

/// Pre-substitute geometric function calls (angle, dist) in an expression string
/// with their numeric values, so the symbolic parser can handle them.
pub(crate) fn presubst_geo_functions(sketch: &Sketch, expr: &str) -> String {
    let mut result = expr.to_string();
    for fname in &["angle", "dist"] {
        loop {
            let Some(start) = result.find(&format!("{}(", fname)) else { break };
            let after_paren = start + fname.len() + 1;
            // Find matching closing paren
            let mut depth = 1;
            let mut end = after_paren;
            for (i, ch) in result[after_paren..].char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => { depth -= 1; if depth == 0 { end = after_paren + i; break; } }
                    _ => {}
                }
            }
            if depth != 0 { break; }
            let call = &result[start..=end];
            if let Some(Ok(val)) = eval_geo_scalar(sketch, call) {
                result = format!("{}{}{}", &result[..start], val, &result[end + 1..]);
            } else {
                break; // can't evaluate, leave as-is for the parser to report the error
            }
        }
    }
    result
}

pub(crate) fn eval_expr_with(sketch: &Sketch, expr_str: &str, extra: &HashMap<String, f64>) -> Result<f64, String> {
    let expr_str = presubst_geo_functions(sketch, expr_str);
    let parsed = arael_sym::parse(&expr_str).map_err(|e| e.msg)?;
    let mut ctx = eval_context(sketch);
    for (k, v) in extra { ctx.insert(k.clone(), *v); }
    let vars: HashMap<&str, f64> = ctx.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    parsed.eval(&vars)
}

pub fn eval_expr(sketch: &Sketch, expr_str: &str) -> Result<f64, String> {
    eval_expr_with(sketch, expr_str, &HashMap::new())
}

// ---------------------------------------------------------------------------
// Geometric functions (return coordinates or scalars)
// ---------------------------------------------------------------------------

pub(crate) fn eval_geo_coord(sketch: &Sketch, call: &str) -> Option<Result<vect2d, String>> {
    let call = call.trim();
    if !call.contains('(') { return None; }
    let (fname, args_str) = call.split_once('(')?;
    let args_str = args_str.strip_suffix(')')?;
    let fname = fname.trim();
    let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

    Some(match fname {
        "intersect" => {
            if args.len() != 2 { return Some(Err("intersect(L0, L1)".into())); }
            let la = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let lb = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let a = &sketch.lines[la]; let b = &sketch.lines[lb];
            let d1 = vect2d::new(a.p2.value.x - a.p1.value.x, a.p2.value.y - a.p1.value.y);
            let d2 = vect2d::new(b.p2.value.x - b.p1.value.x, b.p2.value.y - b.p1.value.y);
            let cross = d1.x * d2.y - d1.y * d2.x;
            if cross.abs() < 1e-12 { return Some(Err("Lines are parallel".into())); }
            let dx = b.p1.value.x - a.p1.value.x;
            let dy = b.p1.value.y - a.p1.value.y;
            let t = (dx * d2.y - dy * d2.x) / cross;
            Ok(vect2d::new(a.p1.value.x + t * d1.x, a.p1.value.y + t * d1.y))
        }
        "midpoint" => {
            if args.len() != 1 { return Some(Err("midpoint(L0)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            Ok(vect2d::new((line.p1.value.x + line.p2.value.x) / 2.0,
                           (line.p1.value.y + line.p2.value.y) / 2.0))
        }
        "project" => {
            if args.len() != 2 { return Some(Err("project(P0, L0)".into())); }
            let pt = match resolve_endpoint_pos(sketch, args[0]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let l = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len2 = dx * dx + dy * dy;
            if len2 < 1e-24 { return Some(Ok(line.p1.value)); }
            let t = ((pt.x - line.p1.value.x) * dx + (pt.y - line.p1.value.y) * dy) / len2;
            Ok(vect2d::new(line.p1.value.x + t * dx, line.p1.value.y + t * dy))
        }
        "along" => {
            if args.len() != 2 { return Some(Err("along(L0, 0.5)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let t = match eval_expr(sketch, args[1]) { Ok(v) => v, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            Ok(vect2d::new(line.p1.value.x + t * (line.p2.value.x - line.p1.value.x),
                           line.p1.value.y + t * (line.p2.value.y - line.p1.value.y)))
        }
        "arc_point" => {
            if args.len() != 2 { return Some(Err("arc_point(A0, 45)".into())); }
            let a = match resolve_arc(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let angle_deg = match eval_expr(sketch, args[1]) { Ok(v) => v, Err(e) => return Some(Err(e)) };
            let arc = &sketch.arcs[a];
            let angle = angle_deg.to_radians();
            Ok(vect2d::new(arc.center.value.x + arc.radius.value * angle.cos(),
                           arc.center.value.y + arc.radius.value * angle.sin()))
        }
        "rotate" => {
            // rotate(point, center, angle_deg)
            if args.len() != 3 { return Some(Err("rotate(P0, center, angle_deg)".into())); }
            let pt = match resolve_endpoint_pos(sketch, args[0]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let center = match resolve_endpoint_pos(sketch, args[1]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let angle = match eval_expr(sketch, args[2]) { Ok(v) => v.to_radians(), Err(e) => return Some(Err(e)) };
            let dx = pt.x - center.x;
            let dy = pt.y - center.y;
            let c = angle.cos(); let s = angle.sin();
            Ok(vect2d::new(center.x + dx * c - dy * s, center.y + dx * s + dy * c))
        }
        "mirror" => {
            if args.len() != 2 { return Some(Err("mirror(P0, L0)".into())); }
            let pt = match resolve_endpoint_pos(sketch, args[0]) { Ok(p) => p, Err(e) => return Some(Err(e)) };
            let l = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len2 = dx * dx + dy * dy;
            if len2 < 1e-24 { return Some(Ok(pt)); }
            let t = ((pt.x - line.p1.value.x) * dx + (pt.y - line.p1.value.y) * dy) / len2;
            let proj = vect2d::new(line.p1.value.x + t * dx, line.p1.value.y + t * dy);
            Ok(vect2d::new(2.0 * proj.x - pt.x, 2.0 * proj.y - pt.y))
        }
        "tangent" => {
            if args.len() != 1 { return Some(Err("tangent(L0)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { return Some(Err("Zero-length line".into())); }
            Ok(vect2d::new(dx / len, dy / len))
        }
        "normal" => {
            if args.len() != 1 { return Some(Err("normal(L0)".into())); }
            let l = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let line = &sketch.lines[l];
            let dx = line.p2.value.x - line.p1.value.x;
            let dy = line.p2.value.y - line.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { return Some(Err("Zero-length line".into())); }
            Ok(vect2d::new(-dy / len, dx / len))
        }
        _ => return None,
    })
}

pub(crate) fn eval_geo_scalar(sketch: &Sketch, call: &str) -> Option<Result<f64, String>> {
    let call = call.trim();
    if !call.contains('(') { return None; }
    let (fname, args_str) = call.split_once('(')?;
    let args_str = args_str.strip_suffix(')')?;
    let fname = fname.trim();
    let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

    Some(match fname {
        "dist" => {
            if args.len() == 2 {
                // Try point-point
                if let (Ok(a), Ok(b)) = (resolve_endpoint_pos(sketch, args[0]), resolve_endpoint_pos(sketch, args[1])) {
                    let dx = a.x - b.x; let dy = a.y - b.y;
                    return Some(Ok((dx * dx + dy * dy).sqrt()));
                }
                // Try point-to-line
                if let (Ok(pt), Ok(lr)) = (resolve_endpoint_pos(sketch, args[0]), resolve_line(sketch, args[1])) {
                    let l = &sketch.lines[lr];
                    let dx = l.p2.value.x - l.p1.value.x;
                    let dy = l.p2.value.y - l.p1.value.y;
                    let len = (dx * dx + dy * dy).sqrt();
                    if len < 1e-12 { return Some(Ok(((l.p1.value.x - pt.x).powi(2) + (l.p1.value.y - pt.y).powi(2)).sqrt())); }
                    let cross = ((pt.x - l.p1.value.x) * dy - (pt.y - l.p1.value.y) * dx).abs();
                    return Some(Ok(cross / len));
                }
                Err(format!("Cannot resolve dist({}, {})", args[0], args[1]))
            } else {
                Err("dist(P0, P1) or dist(P0, L0)".into())
            }
        }
        "angle" => {
            if args.len() != 2 { return Some(Err("angle(L0, L1)".into())); }
            let la = match resolve_line(sketch, args[0]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let lb = match resolve_line(sketch, args[1]) { Ok(r) => r, Err(e) => return Some(Err(e)) };
            let a = &sketch.lines[la]; let b = &sketch.lines[lb];
            let d1x = a.p2.value.x - a.p1.value.x; let d1y = a.p2.value.y - a.p1.value.y;
            let d2x = b.p2.value.x - b.p1.value.x; let d2y = b.p2.value.y - b.p1.value.y;
            let cross = d1x * d2y - d1y * d2x;
            let dot = d1x * d2x + d1y * d2y;
            Ok(cross.atan2(dot).to_degrees())
        }
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Coordinate parsing
// ---------------------------------------------------------------------------

pub(crate) fn parse_coord(ctx: &CommandContext, arg: &str, cursor: Option<vect2d>) -> Result<vect2d, String> {
    let sketch = &ctx.sketch;
    let arg = arg.trim();
    // Special @-keywords
    if arg == "@tangent" {
        return ctx.cursor_tangent.ok_or("No tangent direction available (set by add_line/add_arc/add_earc)".into());
    }
    if arg == "@cursor" || arg == "cursor" {
        return cursor.ok_or("No cursor position available".into());
    }
    // Relative coordinate: @dx,dy
    if let Some(rest) = arg.strip_prefix('@') {
        let prev = cursor.ok_or("No previous point for relative coordinate")?;
        if let Some((x_str, y_str)) = rest.split_once(',') {
            let dx = eval_expr_with(sketch, x_str.trim(), &ctx.session_vars)?;
            let dy = eval_expr_with(sketch, y_str.trim(), &ctx.session_vars)?;
            return Ok(vect2d::new(prev.x + dx, prev.y + dy));
        }
        return Err(format!("Relative coordinate needs @dx,dy format: {}", arg));
    }
    // Cursor keyword (kept for backward compatibility)
    if arg == "cursor" {
        return ctx.cursor.ok_or("Cursor not set".into());
    }
    // Geometric function: intersect(L0, L1), midpoint(L0), tangent(L0), etc.
    if let Some(result) = eval_geo_coord(sketch, arg) {
        return result;
    }
    // Session vector variable
    if let Some(v) = ctx.session_vecs.get(arg) {
        return Ok(*v);
    }
    // Endpoint reference: L0.p2, P0, A0.center
    if let Ok(pos) = resolve_endpoint_pos(sketch, arg) {
        return Ok(pos);
    }
    // x,y with possible expressions
    if let Some((x_str, y_str)) = arg.split_once(',') {
        let x = eval_expr_with(sketch, x_str.trim(), &ctx.session_vars)?;
        let y = eval_expr_with(sketch, y_str.trim(), &ctx.session_vars)?;
        return Ok(vect2d::new(x, y));
    }
    // Vector expression: L0.p2 + normal(L0) * 3, etc.
    // Pre-expand vector sub-expressions to temp vars, then eval as two scalar expressions.
    if let Ok(v) = parse_vec_expr(ctx, arg) {
        return Ok(v);
    }
    Err(format!("Cannot parse coordinate: {}", arg))
}

/// Parse a vector expression like `L0.p2 + normal(L0) * 3`.
/// Scans for vector-valued sub-expressions, replaces them with temp vars,
/// then evaluates x and y components as separate scalar expressions.
pub(crate) fn parse_vec_expr(ctx: &CommandContext, expr: &str) -> Result<vect2d, String> {
    let sketch = &ctx.sketch;
    let mut tmp_vars: HashMap<String, f64> = ctx.session_vars.clone();
    let mut work = expr.to_string();
    let mut counter = 0u32;

    // Repeatedly scan for vector-valued tokens and replace them
    for _ in 0..32 {
        let mut found = false;
        // Find function calls: name(args)
        if let Some(start) = find_func_call(&work) {
            let call = &work[start.0..start.1];
            if let Some(Ok(v)) = eval_geo_coord(sketch, call) {
                let name = format!("__v{}", counter);
                counter += 1;
                tmp_vars.insert(format!("{}.x", name), v.x);
                tmp_vars.insert(format!("{}.y", name), v.y);
                work = format!("{}{}{}", &work[..start.0], name, &work[start.1..]);
                found = true;
            }
        }
        if found { continue; }
        // Find endpoint refs and session vecs: scan for identifiers that resolve to vec2
        let tokens = extract_identifiers(&work);
        let mut replaced = false;
        for (tstart, tend, token) in tokens.iter().rev() {
            // Skip if already a temp var
            if token.starts_with("__v") { continue; }
            // Try session vec
            if let Some(v) = ctx.session_vecs.get(*token) {
                let name = format!("__v{}", counter);
                counter += 1;
                tmp_vars.insert(format!("{}.x", name), v.x);
                tmp_vars.insert(format!("{}.y", name), v.y);
                work = format!("{}{}{}", &work[..*tstart], name, &work[*tend..]);
                replaced = true;
                break;
            }
            // Try endpoint ref (L0.p2, P0, A0.center)
            if let Ok(v) = resolve_endpoint_pos(sketch, token) {
                let name = format!("__v{}", counter);
                counter += 1;
                tmp_vars.insert(format!("{}.x", name), v.x);
                tmp_vars.insert(format!("{}.y", name), v.y);
                work = format!("{}{}{}", &work[..*tstart], name, &work[*tend..]);
                replaced = true;
                break;
            }
        }
        if replaced { continue; }
        break;
    }

    // Now build x and y expressions by appending .x/.y to __vN tokens
    let x_expr = work.replace("__v", "__v").split("__v").enumerate().map(|(i, part)| {
        if i == 0 { part.to_string() }
        else {
            // part starts with the number and rest
            let num_end = part.find(|c: char| !c.is_ascii_digit()).unwrap_or(part.len());
            format!("__v{}.x{}", &part[..num_end], &part[num_end..])
        }
    }).collect::<String>();
    let y_expr = work.split("__v").enumerate().map(|(i, part)| {
        if i == 0 { part.to_string() }
        else {
            let num_end = part.find(|c: char| !c.is_ascii_digit()).unwrap_or(part.len());
            format!("__v{}.y{}", &part[..num_end], &part[num_end..])
        }
    }).collect::<String>();

    let x = eval_expr_with(sketch, &x_expr, &tmp_vars)?;
    let y = eval_expr_with(sketch, &y_expr, &tmp_vars)?;
    Ok(vect2d::new(x, y))
}

/// Find the first function call `name(...)` in the string, return (start, end) byte offsets.
pub(crate) fn find_func_call(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            // Walk back to find function name start
            let paren_pos = i;
            if paren_pos == 0 || !bytes[paren_pos - 1].is_ascii_alphanumeric() {
                i += 1; continue;
            }
            let mut name_start = paren_pos;
            while name_start > 0 && (bytes[name_start - 1].is_ascii_alphanumeric() || bytes[name_start - 1] == b'_') {
                name_start -= 1;
            }
            // Find matching closing paren
            let mut depth = 1;
            let mut j = paren_pos + 1;
            while j < bytes.len() && depth > 0 {
                if bytes[j] == b'(' { depth += 1; }
                if bytes[j] == b')' { depth -= 1; }
                j += 1;
            }
            if depth == 0 {
                let fname = &s[name_start..paren_pos];
                // Only match known geo functions
                if matches!(fname, "intersect" | "midpoint" | "project" | "along" | "arc_point" |
                    "rotate" | "mirror" | "tangent" | "normal") {
                    return Some((name_start, j));
                }
            }
        }
        i += 1;
    }
    None
}

/// Extract identifiers from an expression string as (start, end, &str) triples.
pub(crate) fn extract_identifiers(s: &str) -> Vec<(usize, usize, &str)> {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.') {
                i += 1;
            }
            result.push((start, i, &s[start..i]));
        } else {
            i += 1;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Endpoint resolution for coincident/constraints
// ---------------------------------------------------------------------------

