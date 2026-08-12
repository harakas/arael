use super::*;

pub(crate) fn resolve_line(sketch: &Sketch, name: &str) -> Result<Ref<Line>, String> {
    for r in sketch.lines.refs() {
        if sketch.lines[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown line: {}", name))
}

pub(crate) fn resolve_point(sketch: &Sketch, name: &str) -> Result<Ref<Point>, String> {
    for r in sketch.points.refs() {
        if sketch.points[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown point: {}", name))
}

/// Check if a name looks like an arc/ellipse reference.
pub(crate) fn is_arc_name(name: &str) -> bool {
    name.starts_with('A') || name.starts_with("EA")
}

pub(crate) fn resolve_arc(sketch: &Sketch, name: &str) -> Result<Ref<Arc>, String> {
    for r in sketch.arcs.refs() {
        if sketch.arcs[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown arc: {}", name))
}

/// True when two arcs' current centers coincide within a small
/// geometric tolerance. Used to gate the concentric-distance
/// dimension: the dim installs its own center-coincidence residual,
/// so callers only need to check that the pair is already
/// (approximately) concentric rather than that a paired `Concentric`
/// constraint exists.
pub(crate) fn arcs_are_concentric(sketch: &Sketch, a: Ref<Arc>, b: Ref<Arc>) -> bool {
    let ca = sketch.arcs[a].center.value;
    let cb = sketch.arcs[b].center.value;
    let dx = ca.x - cb.x;
    let dy = ca.y - cb.y;
    (dx * dx + dy * dy).sqrt() < 1e-3
}

/// Resolve an endpoint reference like "L0.p1", "P0", "A0.center"
pub(crate) fn resolve_endpoint_pos(sketch: &Sketch, name: &str) -> Result<vect2d, String> {
    if let Some((entity, field)) = name.split_once('.') {
        if entity.starts_with('L') {
            let r = resolve_line(sketch, entity)?;
            let l = &sketch.lines[r];
            match field {
                "p1" => return Ok(l.p1.value),
                "p2" => return Ok(l.p2.value),
                "p1.x" | "p1.y" | "p2.x" | "p2.y" => {
                    return Err(format!("Use {} as scalar, not coordinate", name));
                }
                _ => return Err(format!("Unknown field: {}.{}", entity, field)),
            }
        } else if is_arc_name(entity) {
            let r = resolve_arc(sketch, entity)?;
            let a = &sketch.arcs[r];
            match field {
                "center" => return Ok(a.center.value),
                "start" => return Ok(vect2d::new(
                    a.center.value.x + a.radius.value * a.start_angle.value.cos(),
                    a.center.value.y + a.radius.value * a.start_angle.value.sin(),
                )),
                "end" => return Ok(vect2d::new(
                    a.center.value.x + a.radius.value * a.end_angle.value.cos(),
                    a.center.value.y + a.radius.value * a.end_angle.value.sin(),
                )),
                _ => return Err(format!("Unknown field: {}.{}", entity, field)),
            }
        }
    }
    // Try as point name
    if name.starts_with('P') {
        let r = resolve_point(sketch, name)?;
        return Ok(sketch.points[r].pos.value);
    }
    Err(format!("Cannot resolve '{}' as coordinate", name))
}

// ---------------------------------------------------------------------------
// Expression evaluation context
// ---------------------------------------------------------------------------


#[derive(Clone, Copy, PartialEq)]
pub(crate) enum EndpointRef {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
}

pub(crate) fn resolve_endpoint_ref(sketch: &Sketch, name: &str) -> Result<EndpointRef, String> {
    if let Some((entity, field)) = name.split_once('.') {
        if entity.starts_with('L') {
            let r = resolve_line(sketch, entity)?;
            match field {
                "p1" => return Ok(EndpointRef::LineP1(r)),
                "p2" => return Ok(EndpointRef::LineP2(r)),
                _ => return Err(format!("Line endpoint must be p1 or p2: {}", name)),
            }
        } else if is_arc_name(entity) {
            let r = resolve_arc(sketch, entity)?;
            match field {
                "center" => return Ok(EndpointRef::ArcCenter(r)),
                "start" => return Ok(EndpointRef::ArcStart(r)),
                "end" => return Ok(EndpointRef::ArcEnd(r)),
                _ => return Err(format!("Arc endpoint must be center/start/end: {}", name)),
            }
        }
    }
    if name.starts_with('P') {
        let r = resolve_point(sketch, name)?;
        return Ok(EndpointRef::Point(r));
    }
    Err(format!("Cannot parse endpoint: {}", name))
}

// ---------------------------------------------------------------------------
// Command execution
// ---------------------------------------------------------------------------

/// Replace session name aliases in a string (word-boundary aware).
/// Replace session-name aliases with their entity names. One pass,
/// left to right, longest alias first at each position. Replaced text
/// is never rescanned, so an alias whose entity name collides with
/// another alias's name cannot chain, and the result does not depend
/// on map iteration order.
pub(crate) fn substitute_aliases(ctx: &CommandContext, input: &str) -> String {
    if ctx.session_names.is_empty() { return input.to_string(); }
    let mut aliases: Vec<(&str, &str)> = ctx.session_names.iter()
        .map(|(a, r)| (a.as_str(), r.as_str())).collect();
    aliases.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(b.0)));
    let bytes = input.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let boundary_before = i == 0 || !is_word(bytes[i - 1]);
        if boundary_before {
            if let Some((alias, real)) = aliases.iter().find(|(a, _)| {
                input[i..].starts_with(a)
                    && (i + a.len() >= input.len() || !is_word(bytes[i + a.len()]))
            }) {
                out.push_str(real);
                i += alias.len();
                continue;
            }
        }
        let ch_len = input[i..].chars().next().map_or(1, |c| c.len_utf8());
        out.push_str(&input[i..i + ch_len]);
        i += ch_len;
    }
    out
}

