use super::*;

pub(crate) fn cmd_select(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();

    // select all
    if tokens.len() == 1 && tokens[0] == "all" {
        ctx.selection.clear();
        for r in ctx.sketch.points.refs() {
            if !ctx.sketch.points[r].helper { ctx.selection.push(Selection::Point(r)); }
        }
        for r in ctx.sketch.lines.refs() { ctx.selection.push(Selection::Line(r)); }
        for r in ctx.sketch.arcs.refs() { ctx.selection.push(Selection::Arc(r)); }
        return Ok(ok(format!("Selected {} entities", ctx.selection.len())));
    }

    // select <entity> chain — follow coincident endpoint connections
    if tokens.len() == 2 && tokens[1] == "chain" {
        let seed = tokens[0];
        return cmd_select_chain(ctx, seed);
    }

    // select <entity> sequence -- end to end until an end or a branch
    if tokens.len() == 2 && tokens[1] == "sequence" {
        let seed = tokens[0];
        return cmd_select_sequence(ctx, seed);
    }
    // select <entity> linked — follow all constraint relationships
    if tokens.len() == 2 && tokens[1] == "linked" {
        let seed = tokens[0];
        return cmd_select_linked(ctx, seed);
    }

    for name in args.split_whitespace() {
        if name.contains('.') {
            // Endpoint selection
            let sel = match resolve_endpoint_ref(&ctx.sketch, name) {
                Ok(EndpointRef::Point(r)) => Selection::Point(r),
                Ok(EndpointRef::LineP1(r)) => Selection::LineP1(r),
                Ok(EndpointRef::LineP2(r)) => Selection::LineP2(r),
                Ok(EndpointRef::ArcCenter(r)) => Selection::ArcCenter(r),
                Ok(EndpointRef::ArcStart(r)) => Selection::ArcStart(r),
                Ok(EndpointRef::ArcEnd(r)) => Selection::ArcEnd(r),
                Err(e) => return Err(e),
            };
            ctx.selection.push(sel);
        } else if name.starts_with('L') {
            let r = resolve_line(&ctx.sketch, name)?;
            ctx.selection.push(Selection::Line(r));
        } else if name.starts_with('P') {
            let r = resolve_point(&ctx.sketch, name)?;
            ctx.selection.push(Selection::Point(r));
        } else if is_arc_name(name) {
            let r = resolve_arc(&ctx.sketch, name)?;
            ctx.selection.push(Selection::Arc(r));
        } else if name.starts_with('M') {
            let i = crate::meta::resolve(&ctx.sketch, name)?;
            ctx.selection.push(Selection::Meta(ctx.sketch.metas[i].mid));
        } else {
            return Err(format!("Cannot select: {}", name).into());
        }
    }
    Ok(ok(format!("Selected {} entities", args.split_whitespace().count())))
}

/// Select all entities connected via coincident endpoint constraints, recursively.
pub(crate) fn cmd_select_chain(ctx: &mut CommandContext, seed: &str) -> CmdResult {
    // Resolve seed to a line or arc index
    let mut line_set: std::collections::HashSet<Ref<Line>> = std::collections::HashSet::new();
    let mut arc_set: std::collections::HashSet<Ref<Arc>> = std::collections::HashSet::new();

    if seed.starts_with('L') {
        let r = resolve_line(&ctx.sketch, seed)?;
        line_set.insert(r);
    } else if is_arc_name(seed) {
        let r = resolve_arc(&ctx.sketch, seed)?;
        arc_set.insert(r);
    } else {
        return Err("chain requires a line or arc".into());
    }

    // Flood fill via coincident constraints
    loop {
        let before = line_set.len() + arc_set.len();
        // LL coincident
        for c in &ctx.sketch.coincident_ll11 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_ll12 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_ll21 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_ll22 { if line_set.contains(&c.a) { line_set.insert(c.b); } if line_set.contains(&c.b) { line_set.insert(c.a); } }
        // Line-Arc coincident
        for c in &ctx.sketch.coincident_lp1_arc_start { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp2_arc_start { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp1_arc_end { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp2_arc_end { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp1_arc_center { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        for c in &ctx.sketch.coincident_lp2_arc_center { if line_set.contains(&c.line) { arc_set.insert(c.arc); } if arc_set.contains(&c.arc) { line_set.insert(c.line); } }
        // Arc-Arc coincident
        for c in &ctx.sketch.coincident_arc_center_start { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_center_end { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_start_center { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_end_center { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_start_start { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_start_end { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_end_start { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        for c in &ctx.sketch.coincident_arc_end_end { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        // Concentric
        for c in &ctx.sketch.concentric { if arc_set.contains(&c.a) { arc_set.insert(c.b); } if arc_set.contains(&c.b) { arc_set.insert(c.a); } }
        if line_set.len() + arc_set.len() == before { break; }
    }

    ctx.selection.clear();
    let mut names = Vec::new();
    for r in &line_set {
        let r = *r;
        ctx.selection.push(Selection::Line(r));
        names.push(ctx.sketch.lines[r].name.clone());
    }
    for r in &arc_set {
        let r = *r;
        ctx.selection.push(Selection::Arc(r));
        names.push(ctx.sketch.arcs[r].name.clone());
    }
    names.sort();
    Ok(ok(format!("Chain: {}", names.join(", "))))
}

/// Select all entities sharing any constraint relationship, recursively.
pub(crate) fn cmd_select_linked(ctx: &mut CommandContext, seed: &str) -> CmdResult {
    // Start with seed entity
    let mut line_set: std::collections::HashSet<Ref<Line>> = std::collections::HashSet::new();
    let mut arc_set: std::collections::HashSet<Ref<Arc>> = std::collections::HashSet::new();

    if seed.starts_with('L') {
        let r = resolve_line(&ctx.sketch, seed)?;
        line_set.insert(r);
    } else if is_arc_name(seed) {
        let r = resolve_arc(&ctx.sketch, seed)?;
        arc_set.insert(r);
    } else {
        return Err("linked requires a line or arc".into());
    }

    // Flood fill: use list_constraints to find all relationships
    // Simpler approach: iterate all constraint vectors and propagate
    loop {
        let before = line_set.len() + arc_set.len();

        // All line-line constraints
        macro_rules! link_ll {
            ($vec:expr) => {
                for c in &$vec {
                    if line_set.contains(&c.a) { line_set.insert(c.b); }
                    if line_set.contains(&c.b) { line_set.insert(c.a); }
                }
            };
        }
        link_ll!(ctx.sketch.parallel);
        link_ll!(ctx.sketch.perpendicular);
        link_ll!(ctx.sketch.equal_length);
        link_ll!(ctx.sketch.collinear);
        link_ll!(ctx.sketch.coincident_ll11);
        link_ll!(ctx.sketch.coincident_ll12);
        link_ll!(ctx.sketch.coincident_ll21);
        link_ll!(ctx.sketch.coincident_ll22);
        link_ll!(ctx.sketch.on_normal_ll);

        // All arc-arc constraints
        macro_rules! link_aa {
            ($vec:expr) => {
                for c in &$vec {
                    if arc_set.contains(&c.a) { arc_set.insert(c.b); }
                    if arc_set.contains(&c.b) { arc_set.insert(c.a); }
                }
            };
        }
        link_aa!(ctx.sketch.equal_radius);
        link_aa!(ctx.sketch.tangent_aa);
        link_aa!(ctx.sketch.concentric);
        link_aa!(ctx.sketch.on_normal_aa);
        link_aa!(ctx.sketch.coincident_arc_center_start);
        link_aa!(ctx.sketch.coincident_arc_center_end);
        link_aa!(ctx.sketch.coincident_arc_start_center);
        link_aa!(ctx.sketch.coincident_arc_end_center);
        link_aa!(ctx.sketch.coincident_arc_start_start);
        link_aa!(ctx.sketch.coincident_arc_start_end);
        link_aa!(ctx.sketch.coincident_arc_end_start);
        link_aa!(ctx.sketch.coincident_arc_end_end);

        // Line-Arc constraints
        for c in &ctx.sketch.tangent_la {
            if line_set.contains(&c.line) { arc_set.insert(c.arc); }
            if arc_set.contains(&c.arc) { line_set.insert(c.line); }
        }
        macro_rules! link_la {
            ($vec:expr, $l:ident, $a:ident) => {
                for c in &$vec {
                    if line_set.contains(&c.$l) { arc_set.insert(c.$a); }
                    if arc_set.contains(&c.$a) { line_set.insert(c.$l); }
                }
            };
        }
        link_la!(ctx.sketch.coincident_lp1_arc_center, line, arc);
        link_la!(ctx.sketch.coincident_lp2_arc_center, line, arc);
        link_la!(ctx.sketch.coincident_lp1_arc_start, line, arc);
        link_la!(ctx.sketch.coincident_lp2_arc_start, line, arc);
        link_la!(ctx.sketch.coincident_lp1_arc_end, line, arc);
        link_la!(ctx.sketch.coincident_lp2_arc_end, line, arc);

        // Symmetry
        for c in &ctx.sketch.symmetry_ll {
            let has = line_set.contains(&c.a) || line_set.contains(&c.b) || line_set.contains(&c.c);
            if has { line_set.insert(c.a); line_set.insert(c.b); line_set.insert(c.c); }
        }

        // Angle constraints
        for c in &ctx.sketch.angle {
            if line_set.contains(&c.a) { line_set.insert(c.b); }
            if line_set.contains(&c.b) { line_set.insert(c.a); }
        }

        if line_set.len() + arc_set.len() == before { break; }
    }

    ctx.selection.clear();
    let mut names = Vec::new();
    for r in &line_set {
        let r = *r;
        ctx.selection.push(Selection::Line(r));
        names.push(ctx.sketch.lines[r].name.clone());
    }
    for r in &arc_set {
        let r = *r;
        ctx.selection.push(Selection::Arc(r));
        names.push(ctx.sketch.arcs[r].name.clone());
    }
    names.sort();
    Ok(ok(format!("Linked: {}", names.join(", "))))
}

pub(crate) fn cmd_deselect(ctx: &mut CommandContext, args: &str) -> CmdResult {
    if args.trim().is_empty() {
        ctx.selection.clear();
        return Ok(ok("Selection cleared"));
    }
    // Deselect specific entities. An already-deselected entity is a
    // no-op; a name that resolves to nothing is an error.
    for name in args.split_whitespace() {
        if name.starts_with('L') {
            match resolve_line(&ctx.sketch, name) {
                Ok(r) => ctx.selection.retain(|s| !matches!(s, Selection::Line(l) if *l == r)),
                Err(e) => return Err(e),
            }
        } else if is_arc_name(name) {
            match resolve_arc(&ctx.sketch, name) {
                Ok(r) => ctx.selection.retain(|s| !matches!(s, Selection::Arc(a) if *a == r)),
                Err(e) => return Err(e),
            }
        } else if name.starts_with('P') {
            match resolve_point(&ctx.sketch, name) {
                Ok(r) => ctx.selection.retain(|s| !matches!(s, Selection::Point(p) if *p == r)),
                Err(e) => return Err(e),
            }
        } else {
            return Err(format!("Unknown entity: {}", name).into());
        }
    }
    Ok(ok(format!("Selection: {} entities", ctx.selection.len())))
}

// ---------------------------------------------------------------------------
// Print / Info / List
// ---------------------------------------------------------------------------

