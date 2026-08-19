//! `offset`: offset a sequence of lines and arcs (crate::offset), and edit
//! an existing offset; `select <entity> sequence`.

use super::*;
use crate::chain;
use crate::offset::{self, OffsetParams};

const KEYWORDS: &[&str] = &[
    "left", "right", "flip", "symmetric", "two", "one", "inward", "outward", "nopin", "pin", "round", "sharp", "caps",
];

/// Resolve a line / arc name into a sequence entity.
fn resolve_entity(sketch: &Sketch, name: &str) -> Result<OffsetEntity, String> {
    if name.starts_with('L') {
        Ok(OffsetEntity::Line(resolve_line(sketch, name)?))
    } else if is_arc_name(name) {
        Ok(OffsetEntity::Arc(resolve_arc(sketch, name)?))
    } else {
        Err(format!("offset takes lines and arcs: {}", name))
    }
}

/// The sequence's turning sense: positive for a counter-clockwise loop,
/// from the signed area of its joint points and arc midpoints.
fn loop_orientation(sketch: &Sketch, seq: &chain::Sequence) -> f64 {
    let mut pts: Vec<vect2d> = Vec::new();
    for s in &seq.segs {
        match s.entity {
            OffsetEntity::Line(l) => {
                let l = &sketch.lines[l];
                pts.push(if s.reversed { l.p2.value } else { l.p1.value });
            }
            OffsetEntity::Arc(a) => {
                let a = &sketch.arcs[a];
                // A full circle / ellipse runs a whole turn from its start.
                let (sa, ea) = (a.start_angle.value, if a.closed { a.start_angle.value + std::f64::consts::TAU } else { a.end_angle.value });
                let (t0, t1) = if s.reversed { (ea, sa) } else { (sa, ea) };
                for k in 0..4 {
                    pts.push(a.point_at(t0 + (t1 - t0) * k as f64 / 4.0));
                }
            }
        }
    }
    let mut area = 0.0;
    for i in 0..pts.len() {
        let (p, q) = (pts[i], pts[(i + 1) % pts.len()]);
        area += p.x * q.y - q.x * p.y;
    }
    area
}

/// Parse the trailing arguments onto `params`; returns the number of
/// distance values seen.
fn parse_args(
    ctx: &CommandContext,
    args: &[&str],
    params: &mut OffsetParams,
    seq: Option<&chain::Sequence>,
) -> Result<usize, String> {
    let mut values = 0usize;
    let mut kind_set = false;
    let mut caps_set = false;
    let mut iter = args.iter();
    while let Some(&tok) = iter.next() {
        match tok {
            "caps" => {
                params.caps = match iter.next().copied() {
                    Some("line") => CapKind::Line,
                    Some("round") => CapKind::Round,
                    Some("none") => CapKind::None,
                    other => return Err(format!("caps takes line, round or none, got {}", other.unwrap_or("nothing"))),
                };
                caps_set = true;
            }
            "left" => params.side = 1.0,
            "right" => params.side = -1.0,
            "flip" => params.side = -params.side,
            "symmetric" => { params.kind = OffsetKind::Symmetric; kind_set = true; }
            "two" => { params.kind = OffsetKind::TwoSides; kind_set = true; }
            "one" => { params.kind = OffsetKind::OneSide; kind_set = true; }
            "inward" | "outward" => {
                let Some(seq) = seq else {
                    return Err(format!("'{}' needs the sequence", tok));
                };
                if !seq.closed {
                    return Err(format!("'{}' applies to a closed sequence only", tok));
                }
                let ccw = loop_orientation(&ctx.sketch, seq) > 0.0;
                // Counter-clockwise: left is inward.
                let inward = if ccw { 1.0 } else { -1.0 };
                params.side = if tok == "inward" { inward } else { -inward };
            }
            "nopin" => params.pinned = false,
            "pin" => params.pinned = true,
            "round" => params.round = true,
            "sharp" => params.round = false,
            _ => {
                let v = offset::parse_value(&ctx.sketch, tok)?;
                match values {
                    0 => params.distance = v,
                    1 => { params.distance2 = Some(v); if !kind_set { params.kind = OffsetKind::TwoSides; } }
                    _ => return Err("offset takes at most two distances".into()),
                }
                values += 1;
            }
        }
    }
    if params.kind == OffsetKind::TwoSides && params.distance2.is_none() {
        params.distance2 = Some(params.distance.clone());
    }
    if params.kind != OffsetKind::TwoSides {
        params.distance2 = None;
    }
    // Going to one side drops the caps unless they were asked for (then
    // the plan refuses).
    if params.kind == OffsetKind::OneSide && !caps_set {
        params.caps = CapKind::None;
    }
    Ok(values)
}

fn outcome_text(ctx: &CommandContext, out: &offset::OffsetOutcome) -> String {
    let i = ctx.sketch.meta_index(out.mid);
    let mut s = match i {
        Some(i) => crate::meta::describe(&ctx.sketch, &ctx.sketch.metas[i]),
        None => out.name.clone(),
    };
    if !out.constraints.is_empty() {
        s += &format!("\n  constraints: {}", out.constraints.join(" "));
    }
    if !out.dims.is_empty() {
        s += &format!("\n  dims: {}", out.dims.join(" "));
    }
    if !out.dropped.is_empty() {
        s += &format!("\n  vanished: {} (offset inward past the radius; the neighbours meet directly)", out.dropped.join(" "));
    }
    if out.approximate {
        s += "\n  approximate: an ellipse's offset is not an ellipse; the result is the concentric ellipse with both semi-axes moved by the distance";
    }
    s
}

/// `offset <entities|sequence E|selection> d [d2] [left|right|flip|symmetric|two|inward|outward|round|caps line|round|nopin]`
/// and `offset M<n> [d [d2]] [flip|left|right|symmetric|two|one|round|sharp|caps line|round|none|nopin|pin]`.
pub(crate) fn cmd_offset(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() {
        return Err("Usage: offset L0 L1 A0 2 [left|right|symmetric|2 3|inward|outward|round|caps line|round|nopin] | offset sequence L0 2 | offset selection 2 | offset M0 3 [flip|symmetric|two 2 3|one|round|sharp|caps none]".into());
    }

    // Edit an existing offset: named, or the selected meta-constraint.
    let selected_meta = || match ctx.selection.as_slice() {
        [Selection::Meta(mid)] => ctx.sketch.meta_index(*mid),
        _ => None,
    };
    let edit = if tokens[0].starts_with('M') {
        ctx.sketch.find_meta(tokens[0]).map(|i| (i, 1))
    } else if tokens[0] == "selection" {
        selected_meta().map(|i| (i, 1))
    } else {
        None
    };
    if let Some((i, skip)) = edit {
        let m = &ctx.sketch.metas[i];
        let mid = m.mid;
        let Some(o) = m.as_offset() else {
            return Err(format!("{} is not an offset", m.name));
        };
        let mut params = offset::params_of(o);
        let seq = offset::sequence_of(o);
        parse_args(ctx, &tokens[skip..], &mut params, Some(&seq))?;
        let out = offset::update(ctx, mid, &params)?;
        let text = outcome_text(ctx, &out);
        return Ok(ok_or_status(ctx, text));
    }

    // The sequence: named entities, a walk from one, or the selection.
    let (seq, rest): (chain::Sequence, &[&str]) = if tokens[0] == "sequence" {
        let Some(seed) = tokens.get(1) else { return Err("offset sequence: missing entity".into()); };
        let seed = resolve_entity(&ctx.sketch, seed)?;
        (chain::walk(&ctx.sketch, seed), &tokens[2..])
    } else if tokens[0] == "selection" {
        let mut set: Vec<OffsetEntity> = Vec::new();
        for s in &ctx.selection {
            let e = match s {
                Selection::Line(r) => OffsetEntity::Line(*r),
                Selection::Arc(r) => OffsetEntity::Arc(*r),
                _ => continue,
            };
            if !set.contains(&e) { set.push(e); }
        }
        if set.is_empty() { return Err("offset selection: no lines or arcs selected".into()); }
        (chain::order(&ctx.sketch, &set)?, &tokens[1..])
    } else {
        let mut set: Vec<OffsetEntity> = Vec::new();
        let mut k = 0;
        while k < tokens.len() && (tokens[k].starts_with('L') || is_arc_name(tokens[k])) && !KEYWORDS.contains(&tokens[k]) {
            let e = resolve_entity(&ctx.sketch, tokens[k])?;
            if !set.contains(&e) { set.push(e); }
            k += 1;
        }
        if set.is_empty() { return Err("offset: name at least one line or arc".into()); }
        (chain::order(&ctx.sketch, &set)?, &tokens[k..])
    };

    let mut params = OffsetParams {
        kind: OffsetKind::OneSide,
        distance: OffsetValue { value: 0.0, expr: None },
        distance2: None,
        side: 1.0,
        pinned: true,
        round: false,
        caps: CapKind::None,
    };
    let values = parse_args(ctx, rest, &mut params, Some(&seq))?;
    if values == 0 {
        return Err("offset: missing the distance".into());
    }
    let plan = offset::plan(&ctx.sketch, &seq, &params)?;
    let out = offset::apply(ctx, &plan)?;
    // Name capture: `_` is the meta, `_0.._N` the result entities in order.
    ctx.session_names.insert("_".into(), out.name.clone());
    let mut idx = 0;
    for side in &out.entities {
        for n in side {
            ctx.session_names.insert(format!("_{}", idx), n.clone());
            idx += 1;
        }
    }
    let text = outcome_text(ctx, &out);
    Ok(ok_or_status(ctx, text))
}

/// `select <entity> sequence`: the walk the offset tool's double-click
/// makes, as a selection.
pub(crate) fn cmd_select_sequence(ctx: &mut CommandContext, seed: &str) -> CmdResult {
    let seed = resolve_entity(&ctx.sketch, seed)?;
    let seq = chain::walk(&ctx.sketch, seed);
    ctx.selection.clear();
    let mut names = Vec::new();
    for s in &seq.segs {
        match s.entity {
            OffsetEntity::Line(r) => ctx.selection.push(Selection::Line(r)),
            OffsetEntity::Arc(r) => ctx.selection.push(Selection::Arc(r)),
        }
        names.push(chain::entity_name(&ctx.sketch, s.entity));
    }
    Ok(ok(format!("Sequence{}: {}", if seq.closed { " (closed)" } else { "" }, names.join(" "))))
}
