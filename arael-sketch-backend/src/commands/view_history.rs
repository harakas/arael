use super::*;

pub(crate) fn cmd_undo(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let n: usize = args.trim().parse().unwrap_or(1);
    for _ in 0..n {
        if let Some((s, c)) = ctx.history.undo() {
            ctx.sketch = s.into();
            ctx.cursor = c.pos;
            ctx.cursor_tangent = c.tangent;
        } else {
            return Ok(ok("Nothing to undo"));
        }
    }
    // History::undo returns a solved sketch; no second solve.
    Ok(ok(format!("Undone {} step(s)", n)))
}

pub(crate) fn cmd_redo(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let n: usize = args.trim().parse().unwrap_or(1);
    for _ in 0..n {
        if let Some((s, c)) = ctx.history.redo() {
            ctx.sketch = s.into();
            ctx.cursor = c.pos;
            ctx.cursor_tangent = c.tangent;
        } else {
            return Ok(ok("Nothing to redo"));
        }
    }
    // History::redo returns a solved sketch; no second solve.
    Ok(ok(format!("Redone {} step(s)", n)))
}

pub(crate) fn cmd_history(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let n: usize = args.trim().parse().unwrap_or(usize::MAX);
    let groups = ctx.history.group_list();
    let total = groups.len();
    let start = total.saturating_sub(n);
    let cursor = ctx.history.cursor;
    let mut lines = Vec::new();
    for (i, (_, end_pos, desc)) in groups.iter().enumerate().skip(start) {
        let marker = if *end_pos == cursor { " <--" } else { "" };
        lines.push(format!("[{}] {}{}", i + 1, desc, marker));
    }
    if cursor == 0 {
        lines.insert(0, "[0] (initial state) <--".into());
    } else {
        lines.insert(0, "[0] (initial state)".into());
    }
    Ok(ok(lines.join("\n")))
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

pub(crate) fn cmd_center(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let args = args.trim();
    if args.is_empty() {
        // Fit all -- delegate to pending_fit
        ctx.pending_fit = true;
        return Ok(ok("Fitting all"));
    }
    let pos = match parse_coord(ctx, args, None) {
        Ok(p) => p,
        Err(_) => {
            // Try as entity name
            if let Ok(p) = resolve_endpoint_pos(&ctx.sketch, args) { p }
            else if args.starts_with('L') {
                if let Ok(r) = resolve_line(&ctx.sketch, args) {
                    let l = &ctx.sketch.lines[r];
                    vect2d::new((l.p1.value.x + l.p2.value.x) / 2.0, (l.p1.value.y + l.p2.value.y) / 2.0)
                } else { return Err(format!("Unknown: {}", args).into()); }
            } else { return Err(format!("Cannot resolve: {}", args).into()); }
        }
    };
    // Center the view on pos
    ctx.offset_x = 400.0 - pos.x as f32 * ctx.scale;
    ctx.offset_y = 300.0 + pos.y as f32 * ctx.scale;
    Ok(ok(format!("Centered on ({:.2},{:.2})", pos.x, pos.y)))
}

pub(crate) fn cmd_zoom(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let args = args.trim();
    match args {
        "+" => { ctx.scale *= 1.5; Ok(ok(format!("Zoom: {:.1}", ctx.scale))) }
        "-" => { ctx.scale /= 1.5; Ok(ok(format!("Zoom: {:.1}", ctx.scale))) }
        _ => {
            if let Ok(v) = args.parse::<f32>() {
                ctx.scale = v.clamp(1e-4, 1e7);
                Ok(ok(format!("Zoom: {:.1}", ctx.scale)))
            } else {
                Err("Usage: zoom +  or  zoom -  or  zoom 2.0".into())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Additional geometry
// ---------------------------------------------------------------------------

