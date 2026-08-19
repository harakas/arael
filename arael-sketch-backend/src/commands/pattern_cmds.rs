//! `pattern`: circular / rectangular patterns of entities (crate::pattern),
//! and editing an existing pattern.

use super::*;
use crate::pattern::{self, PatternParams, PatternSpec};

const KEYWORDS: &[&str] = &[
    "about", "full", "partial", "symmetric", "one", "by", "along", "noalong", "extent", "spacing",
];

/// Resolve an entity name into a pattern source.
fn resolve_entity(sketch: &Sketch, name: &str) -> Result<MetaEntity, String> {
    if name.starts_with('L') {
        Ok(MetaEntity::Line(resolve_line(sketch, name)?))
    } else if is_arc_name(name) {
        Ok(MetaEntity::Arc(resolve_arc(sketch, name)?))
    } else if name.starts_with('P') {
        Ok(MetaEntity::Point(resolve_point(sketch, name)?))
    } else {
        Err(format!("pattern takes lines, arcs and points: {}", name))
    }
}

/// The center of a circular pattern as named: a point or an endpoint.
fn resolve_center(sketch: &Sketch, name: &str) -> Result<CenterRef, String> {
    if name.contains('.') {
        return Ok(CenterRef::Endpoint(to_dim_endpoint(resolve_endpoint_ref(sketch, name)?)));
    }
    if name.starts_with('P') {
        return Ok(CenterRef::Point(resolve_point(sketch, name)?));
    }
    Err(format!("the center must be a point or an endpoint (P0, L0.p1, A0.center), got {}", name))
}

fn parse_quantity(tok: &str) -> Result<u32, String> {
    tok.parse::<u32>().map_err(|_| format!("quantity must be a whole number, got {}", tok))
}

/// Parse the trailing arguments onto `params`. For a circular pattern:
/// `[about C] [N] [full | partial A | symmetric A]`; for a rectangular one:
/// `[N D] [symmetric | one] [by N D [symmetric | one]] [along L | noalong]
/// [extent | spacing]`.
fn parse_args(ctx: &CommandContext, args: &[&str], params: &mut PatternParams) -> Result<(), String> {
    let mut iter = args.iter().peekable();
    // Rectangular: which axis the numbers and symmetric / one apply to.
    let mut axis2 = false;
    let mut numbers_seen = 0usize;
    while let Some(&tok) = iter.next() {
        match (&mut params.kind, tok) {
            (PatternSpec::Circular { center, .. }, "about") => {
                let Some(name) = iter.next() else { return Err("about: missing the center".into()) };
                *center = resolve_center(&ctx.sketch, name)?;
            }
            (PatternSpec::Circular { distribution, .. }, "full") => *distribution = Distribution::Full,
            (PatternSpec::Circular { distribution, angle, .. }, "partial" | "symmetric") => {
                *distribution = if tok == "partial" { Distribution::Partial } else { Distribution::Symmetric };
                if let Some(next) = iter.peek()
                    && !KEYWORDS.contains(next)
                    && let Ok(v) = pattern::parse_value(&ctx.sketch, next)
                {
                    *angle = v;
                    iter.next();
                }
            }
            (PatternSpec::Circular { quantity, .. }, tok) => {
                *quantity = parse_quantity(tok)?;
            }
            (PatternSpec::Rectangular { .. }, "by") => {
                axis2 = true;
                numbers_seen = 0;
            }
            (PatternSpec::Rectangular { frame, .. }, "along") => {
                let Some(name) = iter.next() else { return Err("along: missing the line".into()) };
                *frame = Some(resolve_line(&ctx.sketch, name)?);
            }
            (PatternSpec::Rectangular { frame, .. }, "noalong") => *frame = None,
            (PatternSpec::Rectangular { extent, .. }, "extent") => *extent = true,
            (PatternSpec::Rectangular { extent, .. }, "spacing") => *extent = false,
            (PatternSpec::Rectangular { axis1, axis2: ax2, .. }, "symmetric" | "one") => {
                let ax = if axis2 { ax2 } else { axis1 };
                ax.symmetric = tok == "symmetric";
            }
            (PatternSpec::Rectangular { axis1, axis2: ax2, .. }, tok) => {
                let ax = if axis2 { ax2 } else { axis1 };
                match numbers_seen {
                    0 => ax.quantity = parse_quantity(tok)?,
                    1 => ax.distance = pattern::parse_value(&ctx.sketch, tok)?,
                    _ => return Err(format!("unexpected {}: an axis takes a quantity and a distance", tok)),
                }
                numbers_seen += 1;
            }
        }
    }
    Ok(())
}

fn outcome_text(ctx: &CommandContext, out: &pattern::PatternOutcome) -> String {
    let i = ctx.sketch.meta_index(out.mid);
    let mut s = match i {
        Some(i) => crate::meta::describe(&ctx.sketch, &ctx.sketch.metas[i]),
        None => out.name.clone(),
    };
    if !out.constraints.is_empty() {
        s += &format!("\n  constraints: {}", out.constraints.join(" "));
    }
    s
}

/// `pattern circular <entities|selection> about <P|L.p1|A.center> N [full|partial A|symmetric A]`,
/// `pattern rect <entities|selection> N D [symmetric] [by N D [symmetric]] [along L] [extent]`,
/// and `pattern M<n> ...` with the same keywords.
pub(crate) fn cmd_pattern(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() {
        return Err("Usage: pattern circular L0 A0 about P0 6 [partial 90|symmetric 120] | pattern rect L0 L1 3 10 [by 2 5] [along L5] [extent] [symmetric] | pattern M0 ...".into());
    }

    // Edit an existing pattern: named, or the selected meta-constraint.
    let selected_meta = || match ctx.selection.as_slice() {
        [Selection::Meta(mid)] => ctx.sketch.meta_index(*mid),
        _ => None,
    };
    let edit = if tokens[0].starts_with('M') {
        ctx.sketch.find_meta(tokens[0]).map(|i| (i, 1))
    } else if tokens[0] == "selection" && selected_meta().is_some() {
        selected_meta().map(|i| (i, 1))
    } else {
        None
    };
    if let Some((i, skip)) = edit {
        let m = &ctx.sketch.metas[i];
        let mid = m.mid;
        let Some(p) = m.as_pattern() else {
            return Err(format!("{} is not a pattern", m.name));
        };
        let mut params = pattern::params_of(p);
        parse_args(ctx, &tokens[skip..], &mut params)?;
        let out = pattern::update(ctx, mid, &params)?;
        let text = outcome_text(ctx, &out);
        return Ok(ok_or_status(ctx, text));
    }

    let kind = tokens[0];
    if kind.starts_with('M') && kind[1..].chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("Unknown meta-constraint: {}", kind));
    }
    if kind != "circular" && kind != "rect" && kind != "rectangular" {
        return Err(format!("pattern: expected circular or rect, got {}", kind));
    }
    // The set: named entities or the selection, up to the first keyword /
    // number.
    let mut sources: Vec<MetaEntity> = Vec::new();
    let mut k = 1;
    if tokens.get(1) == Some(&"selection") {
        for s in &ctx.selection {
            let e = match s {
                Selection::Line(r) | Selection::LineP1(r) | Selection::LineP2(r) => MetaEntity::Line(*r),
                Selection::Arc(r) | Selection::ArcCenter(r) | Selection::ArcStart(r) | Selection::ArcEnd(r) => MetaEntity::Arc(*r),
                Selection::Point(r) => MetaEntity::Point(*r),
                _ => continue,
            };
            if !sources.contains(&e) {
                sources.push(e);
            }
        }
        if sources.is_empty() {
            return Err("pattern selection: no lines, arcs or points selected".into());
        }
        k = 2;
    } else {
        while k < tokens.len()
            && !KEYWORDS.contains(&tokens[k])
            && tokens[k].parse::<f64>().is_err()
            && (tokens[k].starts_with('L') || tokens[k].starts_with('P') || is_arc_name(tokens[k]))
        {
            let e = resolve_entity(&ctx.sketch, tokens[k])?;
            if !sources.contains(&e) {
                sources.push(e);
            }
            k += 1;
        }
        if sources.is_empty() {
            return Err("pattern: name at least one line, arc or point".into());
        }
    }
    let rest = &tokens[k..];
    let zero = MetaValue { value: 0.0, expr: None };
    let mut params = if kind == "circular" {
        // The center is required: the first `about`.
        let Some(pos) = rest.iter().position(|t| *t == "about") else {
            return Err("pattern circular: missing 'about <center>'".into());
        };
        let Some(name) = rest.get(pos + 1) else { return Err("about: missing the center".into()) };
        let center = resolve_center(&ctx.sketch, name)?;
        PatternParams { kind: PatternSpec::Circular { center, distribution: Distribution::Full, angle: zero.clone(), quantity: 0 } }
    } else {
        PatternParams {
            kind: PatternSpec::Rectangular {
                frame: None,
                extent: false,
                axis1: PatternAxis { quantity: 1, distance: zero.clone(), symmetric: false },
                axis2: PatternAxis { quantity: 1, distance: zero, symmetric: false },
            },
        }
    };
    parse_args(ctx, rest, &mut params)?;
    let plan = pattern::plan(&ctx.sketch, &sources, &params)?;
    let out = pattern::apply(ctx, &plan)?;
    // Name capture: `_` is the meta, `_0.._N` the copy entities in order.
    ctx.session_names.insert("_".into(), out.name.clone());
    let mut idx = 0;
    for copy in &out.entities {
        for n in copy {
            ctx.session_names.insert(format!("_{}", idx), n.clone());
            idx += 1;
        }
    }
    let text = outcome_text(ctx, &out);
    Ok(ok_or_status(ctx, text))
}
