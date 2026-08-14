//! `split` and `trim`: cut a line or arc at its intersections and
//! keep or delete the pieces. Both feed one engine (crate::split); a
//! trim is a split whose keep mask drops the clicked span.

use super::*;
use crate::split::{
    apply_split_target_names, bracket_cuts, find_cuts, piece_count, post_split_actions,
    target_param_near, Cutter, SplitPlan, SplitTarget,
};

pub(crate) fn resolve_split_target(sketch: &Sketch, name: &str) -> Result<SplitTarget, String> {
    if is_arc_name(name) {
        resolve_arc(sketch, name).map(SplitTarget::Arc)
    } else {
        resolve_line(sketch, name).map(SplitTarget::Line)
    }
}

fn resolve_cutter(sketch: &Sketch, name: &str) -> Result<Cutter, String> {
    if is_arc_name(name) {
        resolve_arc(sketch, name).map(Cutter::Arc)
    } else {
        resolve_line(sketch, name).map(Cutter::Line)
    }
}

fn target_is_closed(sketch: &Sketch, target: SplitTarget) -> bool {
    match target {
        SplitTarget::Line(_) => false,
        SplitTarget::Arc(r) => sketch.arcs[r].closed,
    }
}

fn target_name(sketch: &Sketch, target: SplitTarget) -> String {
    match target {
        SplitTarget::Line(r) => sketch.lines[r].name.clone(),
        SplitTarget::Arc(r) => sketch.arcs[r].name.clone(),
    }
}

/// Run the split plan plus its gated follow-up constraints inside the
/// current group, composing the id-rich report.
fn run_split(
    ctx: &mut CommandContext,
    plan: SplitPlan,
    pin: bool,
    header: String,
    mut extra_notes: Vec<String>,
) -> CmdResult {
    let outcome = ctx.exec_split(plan.clone())?;
    // Gated follow-ups; a rejection here means "already implied".
    let mark = ctx.sketch.next_constraint_id;
    let mut implied: Vec<String> = Vec::new();
    for action in post_split_actions(&plan, &outcome.pieces, pin) {
        let saved_skip = ctx.skip_dof_check;
        ctx.exec(action);
        ctx.skip_dof_check = saved_skip;
        if let Some(e) = ctx.status_error.take() {
            implied.push(e);
        }
    }
    // Everything minted after `mark` is a follow-up constraint.
    let mut added: Vec<String> = Vec::new();
    ctx.sketch.for_each_constraint_collection_ref(|_, meta, coll| {
        if meta.dimension_backed {
            return;
        }
        for i in 0..coll.len() {
            let c = coll.item(i);
            if c.nid() >= mark {
                added.push(format!("C{} {}", c.nid(), c.describe(&ctx.sketch)));
            }
        }
    });

    // Capture piece names for `a, b = split ...` and `_`.
    let kept: Vec<String> = outcome.piece_names.iter().flatten().cloned().collect();
    for (i, name) in kept.iter().enumerate() {
        ctx.session_names.insert(format!("_{}", i), name.clone());
    }
    if let Some(first) = kept.first() {
        ctx.session_names.insert("_".into(), first.clone());
    }

    let mut lines = vec![header];
    let section = |label: &str, items: &[String]| -> Option<String> {
        if items.is_empty() {
            None
        } else {
            Some(format!("  {}: {}", label, items.join("; ")))
        }
    };
    if let Some(s) = section("added", &added) { lines.push(s); }
    if let Some(s) = section("moved", &outcome.moved) { lines.push(s); }
    if let Some(s) = section("copied", &outcome.copied) { lines.push(s); }
    if let Some(s) = section("dropped", &outcome.dropped) { lines.push(s); }
    if let Some(s) = section("expressions", &outcome.expr_report) { lines.push(s); }
    if let Some(s) = section("implied", &implied) { lines.push(s); }
    lines.append(&mut extra_notes);
    Ok(ok(lines.join("\n")))
}

/// Delete the whole entity (trim with nothing to cut at). Reports the
/// cascade like `delete` does.
fn trim_delete_all(ctx: &mut CommandContext, target: SplitTarget) -> CmdResult {
    let name = target_name(&ctx.sketch, target);
    // Constraints and dimensions that will cascade.
    let mut cascaded: Vec<String> = Vec::new();
    ctx.sketch.for_each_constraint_collection_ref(|_, meta, coll| {
        for i in 0..coll.len() {
            let c = coll.item(i);
            let hits = match target {
                SplitTarget::Line(t) => c.references_line(t),
                SplitTarget::Arc(t) => c.references_arc(t),
            };
            if hits && !meta.dimension_backed {
                cascaded.push(format!("C{}", c.nid()));
            }
        }
    });
    for d in ctx.sketch.dimensions.iter() {
        let hits = match target {
            SplitTarget::Line(t) => d.kind.references_line(t),
            SplitTarget::Arc(t) => d.kind.references_arc(t),
        };
        if hits {
            cascaded.push(d.name.clone());
        }
    }
    ctx.begin_group();
    match target {
        SplitTarget::Line(r) => { ctx.exec(Action::DeleteLine { line: r }); }
        SplitTarget::Arc(r) => { ctx.exec(Action::DeleteArc { arc: r }); }
    }
    let tail = if cascaded.is_empty() {
        String::new()
    } else {
        format!(" [removed {}]", cascaded.join(" "))
    };
    Ok(ok(format!("Trimmed (no intersections): deleted {}{}", name, tail)))
}

fn cut_note(plan: &SplitPlan, sketch: &Sketch) -> Vec<String> {
    plan.cuts
        .iter()
        .map(|c| {
            let on = match c.cutter {
                Some(Cutter::Line(r)) => format!(" on {}", sketch.lines[r].name),
                Some(Cutter::Arc(r)) => format!(" on {}", sketch.arcs[r].name),
                None => String::new(),
            };
            format!("cut ({:.3},{:.3}){}", c.pos.x, c.pos.y, on)
        })
        .collect()
}

pub(crate) fn cmd_split(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nopin] = peel_keywords(&mut tokens, ["nopin"]);
    if tokens.is_empty() {
        return Err("Usage: split <entity> <coord> [radius] | split <entity> by <cutter>... [nopin]".into());
    }
    let target = resolve_split_target(&ctx.sketch, tokens[0])?;
    let closed = target_is_closed(&ctx.sketch, target);
    let tname = target_name(&ctx.sketch, target);

    let (cuts, refused) = if tokens.len() >= 3 && tokens[1] == "by" {
        let cutters: Vec<Cutter> = tokens[2..]
            .iter()
            .map(|t| resolve_cutter(&ctx.sketch, t))
            .collect::<Result<_, _>>()?;
        find_cuts(&ctx.sketch, target, Some(&cutters))
    } else {
        if tokens.len() < 2 || tokens.len() > 3 {
            return Err("Usage: split <entity> <coord> [radius] | split <entity> by <cutter>... [nopin]".into());
        }
        let coord = parse_coord(ctx, tokens[1], ctx.cursor)?;
        let (t, dist) = target_param_near(&ctx.sketch, target, coord);
        if tokens.len() == 3 {
            let radius = eval_expr(&ctx.sketch, tokens[2])?;
            if dist > radius {
                return Err(format!(
                    "nearest point on {} is {:.4} away (search radius {:.4})",
                    tname, dist, radius
                ));
            }
        }
        let (all_cuts, _) = find_cuts(&ctx.sketch, target, None);
        if all_cuts.is_empty() {
            return Err(format!("no intersections on {} to cut at", tname));
        }
        if closed && all_cuts.len() < 2 {
            return Err(format!(
                "{} is closed and has only one crossing; splitting needs two",
                tname
            ));
        }
        let (cuts, _) = bracket_cuts(&ctx.sketch, target, &all_cuts, t, closed);
        (cuts, Vec::new())
    };
    if cuts.is_empty() {
        let mut msg = format!("no usable intersections on {}", tname);
        if !refused.is_empty() {
            msg += &format!(" ({})", refused.join("; "));
        }
        return Err(msg);
    }
    if closed && cuts.len() < 2 {
        return Err(format!(
            "{} is closed; the named cutters cross it only once",
            tname
        ));
    }
    let n = piece_count(closed, cuts.len());
    let plan = SplitPlan { target, cuts, keep: vec![true; n] };
    let notes = cut_note(&plan, &ctx.sketch);
    ctx.begin_group();
    let names_preview = apply_split_target_names(&ctx.sketch, &plan);
    let header = format!(
        "Split {} -> {} [{}]",
        tname,
        names_preview.join(" "),
        notes.join(", ")
    );
    let mut extra = Vec::new();
    if !refused.is_empty() {
        extra.push(format!("  refused: {}", refused.join("; ")));
    }
    run_split(ctx, plan, !nopin, header, extra)
}

pub(crate) fn cmd_trim(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let [nopin] = peel_keywords(&mut tokens, ["nopin"]);
    if tokens.is_empty() {
        return Err("Usage: trim <entity> <coord> [radius] | trim <entity> by <c1> <c2> | trim <entity> by <c> forward|backward [nopin]".into());
    }
    let target = resolve_split_target(&ctx.sketch, tokens[0])?;
    let closed = target_is_closed(&ctx.sketch, target);
    let tname = target_name(&ctx.sketch, target);

    let (cuts, drop_idx) = if tokens.len() >= 3 && tokens[1] == "by" {
        if closed {
            return Err(format!(
                "{} is closed: 'by' names two complementary spans; use the coordinate form",
                tname
            ));
        }
        let rest = &tokens[2..];
        match rest {
            [cutter, dir @ ("forward" | "backward")] => {
                let c = resolve_cutter(&ctx.sketch, cutter)?;
                let (cuts, refused) = find_cuts(&ctx.sketch, target, Some(&[c]));
                if cuts.is_empty() {
                    let mut msg = format!("{} does not cross {}", cutter, tname);
                    if !refused.is_empty() {
                        msg += &format!(" ({})", refused.join("; "));
                    }
                    return Err(msg);
                }
                // Cuts are sorted along the target's direction:
                // forward trims past the crossing nearest the end.
                if *dir == "forward" {
                    (vec![cuts[cuts.len() - 1].clone()], 1)
                } else {
                    (vec![cuts[0].clone()], 0)
                }
            }
            [c1, c2] => {
                let r1 = resolve_cutter(&ctx.sketch, c1)?;
                let r2 = resolve_cutter(&ctx.sketch, c2)?;
                let (cuts1, _) = find_cuts(&ctx.sketch, target, Some(&[r1]));
                let (cuts2, _) = find_cuts(&ctx.sketch, target, Some(&[r2]));
                if cuts1.is_empty() {
                    return Err(format!("{} does not cross {}", c1, tname));
                }
                if cuts2.is_empty() {
                    return Err(format!("{} does not cross {}", c2, tname));
                }
                // Closest pair, one crossing per cutter.
                let mut best: Option<(f64, crate::split::SplitCut, crate::split::SplitCut)> = None;
                let mut tie = false;
                for a in &cuts1 {
                    for b in &cuts2 {
                        let d = (a.param - b.param).abs();
                        match &best {
                            Some((bd, ..)) if (d - bd).abs() < 1e-12 => tie = true,
                            Some((bd, ..)) if d < *bd => {
                                best = Some((d, a.clone(), b.clone()));
                                tie = false;
                            }
                            None => best = Some((d, a.clone(), b.clone())),
                            _ => {}
                        }
                    }
                }
                let (_, a, b) = best.unwrap();
                if tie {
                    return Err(format!(
                        "ambiguous: {} and {} cross {} at equally-spaced pairs; use the coordinate form",
                        c1, c2, tname
                    ));
                }
                let mut pair = vec![a, b];
                pair.sort_by(|x, y| x.param.partial_cmp(&y.param).unwrap());
                (pair, 1)
            }
            _ => {
                return Err("Usage: trim <entity> by <c1> <c2> | trim <entity> by <c> forward|backward".into());
            }
        }
    } else {
        if tokens.len() < 2 || tokens.len() > 3 {
            return Err("Usage: trim <entity> <coord> [radius] | trim <entity> by <c1> <c2> | trim <entity> by <c> forward|backward [nopin]".into());
        }
        let coord = parse_coord(ctx, tokens[1], ctx.cursor)?;
        let (t, dist) = target_param_near(&ctx.sketch, target, coord);
        if tokens.len() == 3 {
            let radius = eval_expr(&ctx.sketch, tokens[2])?;
            if dist > radius {
                return Err(format!(
                    "nearest point on {} is {:.4} away (search radius {:.4})",
                    tname, dist, radius
                ));
            }
        }
        let (all_cuts, _) = find_cuts(&ctx.sketch, target, None);
        if all_cuts.is_empty() || (closed && all_cuts.len() < 2) {
            // Nothing to cut at: trim deletes the whole entity.
            return trim_delete_all(ctx, target);
        }
        bracket_cuts(&ctx.sketch, target, &all_cuts, t, closed)
    };

    let n = piece_count(closed, cuts.len());
    let mut keep = vec![true; n];
    if drop_idx >= n {
        return Err("internal: trim span index out of range".into());
    }
    keep[drop_idx] = false;
    let plan = SplitPlan { target, cuts, keep };
    let notes = cut_note(&plan, &ctx.sketch);
    ctx.begin_group();
    let names_preview = apply_split_target_names(&ctx.sketch, &plan);
    let header = format!(
        "Trimmed {} -> {} [{}]",
        tname,
        names_preview.join(" "),
        notes.join(", ")
    );
    run_split(ctx, plan, !nopin, header, Vec::new())
}
