// Command system: parse and execute text commands for the sketch.
// Decoupled from GUI -- operates on CommandContext which holds sketch state.

use arael::simple_lm::RootProblem;
use std::collections::HashMap;
use arael::model::JacobianModel;
use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::{Action, Created};
use crate::geometry::{arc_start_pos, arc_end_pos};
use crate::history::{History, CursorState};
use crate::ids::Selection;


mod context;
mod resolve;
mod expr;
mod entities;
mod arcs_fillet;
mod constraint_cmds;
mod dimension_cmds;
mod edit;
mod selection;
mod query;
mod view_history;
mod session;
mod dof;
mod help;
mod split_cmds;
mod offset_cmds;

pub use context::*;
pub(crate) use resolve::*;
pub use expr::*;
pub(crate) use entities::*;
pub(crate) use arcs_fillet::*;
// The GUI's corner tools parse radius tokens the same way the
// commands do.
pub use arcs_fillet::parse_radius_token;
pub(crate) use constraint_cmds::*;
pub use dimension_cmds::*;
pub(crate) use edit::*;
pub(crate) use selection::*;
pub(crate) use query::*;
pub(crate) use view_history::*;
pub(crate) use session::*;
pub(crate) use dof::*;
pub use help::*;
pub(crate) use split_cmds::*;
pub(crate) use offset_cmds::*;

#[cfg(test)]
mod tests;

pub fn execute(ctx: &mut CommandContext, input: &str) -> Vec<CommandResult> {
    use std::io::Write;
    let mut results = Vec::new();
    for cmd in input.split(';') {
        let cmd = cmd.trim();
        if cmd.is_empty() { continue; }
        if ctx.echo_stdout && !cmd.starts_with('#') {
            println!("> {}", cmd);
            let _ = std::io::stdout().flush();
        }
        let mut r = execute_one(ctx, cmd);
        if r.is_error && !r.output.starts_with('>') {
            r.output = format!("'{}': {}", cmd, r.output);
        }
        // Notices raised by the actions this command ran (a meta-
        // constraint dropped because its result was edited) ride along.
        for n in std::mem::take(&mut ctx.notices) {
            if !r.output.is_empty() {
                r.output.push('\n');
            }
            r.output.push_str("notice: ");
            r.output.push_str(&n);
            r.no_echo = false;
        }
        if ctx.echo_stdout && !r.output.is_empty() {
            println!("{}", r.output);
            let _ = std::io::stdout().flush();
        }
        let is_err = r.is_error;
        results.push(r);
        // Commands that delete entities or replace the sketch (delete,
        // clear, load, undo, redo, goto) leave the selection pointing
        // at freed arena slots or shifted indices; any later read
        // panics. Prune after every command.
        prune_selection(ctx);
        if is_err { break; }
    }
    if results.is_empty() {
        results.push(ok(""));
    }
    let dof_start = results.len();
    append_dof_tail(ctx, input, &mut results);
    if ctx.echo_stdout {
        for r in &results[dof_start..] {
            if !r.output.is_empty() {
                println!("{}", r.output);
                let _ = std::io::stdout().flush();
            }
        }
    }
    results
}

/// Append a `DOF: <n>` summary line at the end of a command
/// sequence so the user always knows the current degree-of-freedom
/// state after their edits -- especially important when the
/// sequence errored out partway through, since the user needs to
/// see how their edits left the sketch before diagnosing the
/// failure. Skipped only when:
///
/// - the sketch is empty (no entities -> no meaningful DOF);
/// - every command in the sequence is purely observational
///   (`dof*`, `list`, `info`, `find`, `help`, `print`, `cost`,
///   `history`, `measure`, `msg`) -- DOF hasn't changed, no need
///   to re-announce it.
fn append_dof_tail(ctx: &mut CommandContext, input: &str, results: &mut Vec<CommandResult>) {
    if ctx.sketch.lines.refs().next().is_none()
        && ctx.sketch.points.refs().next().is_none()
        && ctx.sketch.arcs.refs().next().is_none() {
        return;
    }
    const OBSERVATIONAL: &[&str] = &[
        "dof", "list", "info", "find", "help", "print", "cost",
        "history", "measure", "msg",
    ];
    let all_observational = input.split(';').map(|c| c.trim()).filter(|c| !c.is_empty())
        .all(|c| {
            let head = c.split_whitespace().next().unwrap_or("");
            OBSERVATIONAL.contains(&head)
        });
    if all_observational { return; }
    let Ok(dof) = ctx.sketch.dof() else { return };
    results.push(CommandResult {
        output: format!("DOF: {}", dof),
        is_error: false,
        no_echo: false,
        markdown: false,
    });
}

fn execute_one(ctx: &mut CommandContext, input: &str) -> CommandResult {
    let input = input.trim();

    // Strip inline comments (# not inside quotes) first, then the
    // trailing "force" keyword, so `horizontal L0 force # note`
    // parses. `msg` is exempt from both: its text is verbatim.
    let (input, force) = if input.starts_with("msg ") || input == "msg" {
        (input, false)
    } else {
        strip_force(strip_inline_comment(input))
    };
    ctx.skip_dof_check = force;
    let input = input.trim();

    // Comments (entire line)
    if input.is_empty() || input.starts_with('#') {
        return CommandResult { output: String::new(), is_error: false, no_echo: true, markdown: false };
    }

    // Assignment: "name = command args" or "let name = ..."
    let assign_input = input.strip_prefix("let ").map(|s| (true, s)).unwrap_or((false, input));
    if let Some((lhs, rhs)) = assign_input.1.split_once('=') {
        let var_name = lhs.trim();
        let rhs = rhs.trim();
        // Multi-assignment: "a, b, c = add_line ..."
        if var_name.contains(',') {
            let names: Vec<&str> = var_name.split(',').map(|s| s.trim()).collect();
            if names.iter().all(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')) {
                let result = execute_one(ctx, rhs);
                if !result.is_error {
                    for (i, name) in names.iter().enumerate() {
                        if let Some(entity) = ctx.session_names.get(&format!("_{}", i)).cloned() {
                            ctx.session_names.insert(name.to_string(), entity);
                        }
                    }
                }
                return result;
            }
        }
        // Check if lhs is a simple identifier
        if !var_name.is_empty()
            && !var_name.contains('.')
            && !var_name.contains(' ')
            && var_name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            && !var_name.bytes().next().unwrap_or(b'0').is_ascii_digit()
        {
            // Try as a command first (e.g. "base = add_line 0,0 5,0")
            let first_word = rhs.split_whitespace().next().unwrap_or("");
            let is_command = matches!(first_word,
                "add_line" | "add_rect" | "add_rect3" | "add_rectcenter" |
                "add_point" | "add_circle" | "add_circle2" | "add_circle3" |
                "add_circle2t" | "add_circle3t" | "add_ellipse" | "add_arc" |
                "add_earc" | "add_earc3" | "add_earc_center" | "add_earc_tangent" | "add_earc_rtangent" | "offset_line" | "offset" | "fillet" | "chamfer" | "mirror" |
                "split" | "trim" |
                "length" | "radius" | "radius_b" | "sweep" | "angle" | "distance");
            if is_command {
                let dim_count_before = ctx.sketch.dimensions.len();
                let prev_entity = ctx.session_names.get("_").cloned();
                let result = execute_one(ctx, rhs);
                if !result.is_error {
                    // Check for new entity name (only if "_" was updated by this command)
                    let cur_entity = ctx.session_names.get("_").cloned();
                    let entity_captured = if cur_entity != prev_entity {
                        if let Some(entity_name) = cur_entity {
                            ctx.session_names.insert(var_name.to_string(), entity_name);
                            true
                        } else { false }
                    } else {
                        false
                    };
                    // Check for new dimension — dimension commands (length, angle, etc.)
                    // don't set "_" like geometry commands do, so we detect new dimensions
                    // by comparing count before/after and capture the dimension name.
                    // Only when the command didn't already produce an entity (e.g. "driven"
                    // on geometry commands creates dimensions as a side effect).
                    if !entity_captured && ctx.sketch.dimensions.len() > dim_count_before
                        && let Some(dim) = ctx.sketch.dimensions.last() {
                            ctx.session_names.insert(var_name.to_string(), dim.name.clone());
                        }
                }
                return result;
            }
            // Otherwise treat as scalar/vector assignment (existing let behavior)
            return cmd_let(ctx, &format!("{} = {}", var_name, rhs)).unwrap_or_else(err);
        }
    }

    // Substitute session_names aliases in arguments
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0];

    // Check if command is blocked in this context
    if ctx.blocked_commands.contains(&cmd) {
        return err(format!("'{}' is not allowed in this context", cmd));
    }
    let raw_args = if parts.len() > 1 { parts[1].trim() } else { "" };

    // Replace known aliases in args (word-boundary aware)
    let args_str = substitute_aliases(ctx, raw_args);
    let args_str = args_str.as_str();

    let result: CmdResult = match cmd {
        "explain" => cmd_explain(ctx, args_str),
        "add_line" => cmd_add_line(ctx, args_str),
        "add_rect" => cmd_add_rect(ctx, args_str),
        "add_rect3" => cmd_add_rect3(ctx, args_str),
        "add_rectcenter" => cmd_add_rectcenter(ctx, args_str),
        "add_point" => cmd_add_point(ctx, args_str),
        "add_circle" => cmd_add_circle(ctx, args_str),
        "add_circle2" => cmd_add_circle2(ctx, args_str),
        "add_circle3" => cmd_add_circle3(ctx, args_str),
        "add_circle2t" => cmd_add_circle2t(ctx, args_str),
        "add_circle3t" => cmd_add_circle3t(ctx, args_str),
        "add_ellipse" => cmd_add_ellipse(ctx, args_str),
        "add_arc" => cmd_add_arc(ctx, args_str),
        "add_earc" => cmd_add_earc(ctx, args_str),
        "add_earc3" => cmd_add_earc3(ctx, args_str),
        "add_earc_center" => cmd_add_earc_center(ctx, args_str),
        "add_earc_tangent" => cmd_add_earc_tangent(ctx, args_str),
        "add_earc_rtangent" => cmd_add_earc_rtangent(ctx, args_str),
        "offset_line" => cmd_offset_line(ctx, args_str),
        "offset" => cmd_offset(ctx, args_str),
        "fillet" => cmd_fillet(ctx, args_str),
        "chamfer" => cmd_chamfer(ctx, args_str),
        "split" => cmd_split(ctx, args_str),
        "trim" => cmd_trim(ctx, args_str),
        "scale" => cmd_scale(ctx, args_str),
        "delete" => cmd_delete(ctx, args_str),
        "horizontal" => cmd_horizontal(ctx, args_str),
        "vertical" => cmd_vertical(ctx, args_str),
        "parallel" => cmd_parallel(ctx, args_str),
        "perpendicular" | "perp" => cmd_perpendicular(ctx, args_str),
        "equal" => cmd_equal(ctx, args_str),
        "collinear" => cmd_collinear(ctx, args_str),
        "tangent" => cmd_tangent(ctx, args_str),
        "coincident" => cmd_coincident(ctx, args_str),
        "concentric" => cmd_concentric(ctx, args_str),
        "on_normal" => cmd_on_normal(ctx, args_str),
        "midpoint" => cmd_midpoint(ctx, args_str),
        "symmetry" => cmd_symmetry(ctx, args_str),
        "mirror" => cmd_mirror(ctx, args_str),
        "point_on" => cmd_point_on(ctx, args_str),
        "length" => cmd_length(ctx, args_str),
        "radius" => cmd_radius(ctx, args_str),
        "radius_b" => cmd_radius_b(ctx, args_str),
        "sweep" => cmd_sweep(ctx, args_str),
        "angle" => cmd_angle(ctx, args_str),
        "distance" => cmd_distance(ctx, args_str),
        "hdistance" => cmd_hdistance(ctx, args_str),
        "vdistance" => cmd_vdistance(ctx, args_str),
        "xangle" => cmd_xangle(ctx, args_str),
        "lock" => cmd_lock(ctx, args_str),
        "unlock" => cmd_unlock(ctx, args_str),
        "param" => cmd_param(ctx, args_str),
        "del_param" => cmd_del_param(ctx, args_str),
        "rename_param" => cmd_rename_param(ctx, args_str),
        "style" => cmd_style(ctx, args_str),
        "quiet" => cmd_quiet(ctx, args_str),
        "constr" => cmd_constr(ctx, args_str),
        "drag" => cmd_drag(ctx, args_str),
        "select" => cmd_select(ctx, args_str),
        "deselect" => cmd_deselect(ctx, args_str),
        "print" => cmd_print(ctx, args_str),
        "info" => cmd_info(ctx, args_str),
        "measure" => cmd_measure(ctx, args_str),
        "list" => cmd_list(ctx, args_str),
        "find" => cmd_find(ctx, args_str),
        "dof" => cmd_dof(ctx, args_str),
        "cost" => {
            let cost = ctx.sketch.current_cost();
            Ok(ok(format!("Cost: {:.6}", cost)))
        }
        "undo" => cmd_undo(ctx, args_str),
        "redo" => cmd_redo(ctx, args_str),
        "history" => cmd_history(ctx, args_str),
        "goto" => cmd_goto(ctx, args_str),
        "center" => cmd_center(ctx, args_str),
        "zoom" => cmd_zoom(ctx, args_str),
        "msg" => Ok(CommandResult {
            output: args_str.replace("\\n", "\n"), is_error: false, no_echo: true, markdown: true,
        }),
        "cursor" => cmd_cursor(ctx, args_str),
        "dim_pos" => cmd_dim_pos(ctx, args_str),
        "set_derived" => cmd_set_derived(ctx, args_str),
        "set_driven" => cmd_set_driven(ctx, args_str),
        "freeze" => cmd_freeze(ctx, args_str),
        "clear" => { ctx.sketch = Sketch::new().into(); ctx.history = crate::history::History::new(&ctx.sketch); Ok(ok("Cleared")) },
        "let" => cmd_let(ctx, args_str),
        "save" => cmd_save(ctx, args_str),
        "load" => cmd_load(ctx, args_str),
        "help" => cmd_help(args_str),
        "exit" | "quit" => { ctx.exit_requested = true; Ok(ok("Exiting")) },
        "ai" => Ok(ok("AI assistant not yet configured. Use --mcp to enable MCP server for external AI agents.")),
        _ if cmd.starts_with('!') => Ok(ok("AI assistant not yet configured. Use --mcp to enable MCP server for external AI agents.")),
        _ => Err(format!("Unknown command: {}. Type 'help' for commands.", cmd)),
    };
    result.unwrap_or_else(err)
}

// ---------------------------------------------------------------------------
// Geometry commands
// ---------------------------------------------------------------------------


/// Strip trailing "force" keyword from args, return (cleaned_args, is_force).
/// Strip inline comment: everything after `#` that is not inside quotes.
fn strip_inline_comment(input: &str) -> &str {
    let mut in_quote = false;
    for (i, ch) in input.char_indices() {
        if ch == '"' { in_quote = !in_quote; }
        else if ch == '#' && !in_quote { return &input[..i]; }
    }
    input
}

fn strip_force(args: &str) -> (&str, bool) {
    if args.trim().ends_with(" force") || args.trim() == "force" {
        let trimmed = args.trim();
        if trimmed == "force" {
            ("", true)
        } else {
            (&trimmed[..trimmed.len() - 6], true)
        }
    } else {
        (args, false)
    }
}

/// Dry-run wrapper: snapshot the sketch, execute the inner command
/// (which may add a constraint or dimension), capture whether it was
/// accepted or rejected (and any blocker info), then restore the
/// sketch so no state is kept. Returns a human-readable explanation.
/// Intended for the `explain` command and for UI code that wants to
/// preview whether a proposed constraint would succeed.
pub fn dry_run(ctx: &mut CommandContext, input: &str) -> DryRunOutcome {
    let snapshot = bincode::serialize(&ctx.sketch).ok();
    let history_cursor_before = ctx.history.cursor;
    let status_err_before = ctx.status_error.take();
    let blockers_before = ctx.status_blocker_names.take();
    let last_cost_before = ctx.last_cost;
    let dof_before = ctx.sketch.cached_dof();

    let result = execute_one(ctx, input);
    let err = ctx.status_error.take().or_else(||
        if result.is_error { Some(result.output.clone()) } else { None });
    let blockers = ctx.status_blocker_names.take();

    if let Some(snap) = snapshot
        && let Ok(restored) = bincode::deserialize::<Sketch>(&snap) {
        ctx.sketch = restored.into();
    }
    // Roll history back: drop any entries the inner command pushed
    // past the original cursor and reset the cursor itself.
    ctx.history.actions.truncate(history_cursor_before);
    ctx.history.snapshots.truncate(history_cursor_before);
    ctx.history.cursors.truncate(history_cursor_before);
    ctx.history.groups.truncate(history_cursor_before);
    ctx.history.cursor = history_cursor_before;
    ctx.status_error = status_err_before;
    ctx.status_blocker_names = blockers_before;
    ctx.last_cost = last_cost_before;
    // Re-seed the cache at the restored sketch's generation; the
    // value is still true for the restored state.
    if let Some(d) = dof_before {
        ctx.sketch.mutate_values(|s| s.set_cached_dof(d));
    }

    DryRunOutcome {
        accepted: err.is_none(),
        message: err.unwrap_or(result.output),
        blocker_names: blockers.unwrap_or_default(),
    }
}

/// Result of a dry-run command evaluation.
#[allow(dead_code)]
pub struct DryRunOutcome {
    /// True if the inner command succeeded.
    pub accepted: bool,
    /// Success output or rejection message (includes blocker hint
    /// when the inner path produced one).
    pub message: String,
    /// User-facing names of conflicting constraints when the inner
    /// path was a DOF-rejection with blocker analysis. Empty
    /// otherwise. Intended for UI consumers (e.g. drag preview) that
    /// want the structured names rather than parsing the message.
    pub blocker_names: Vec<String>,
}

fn cmd_explain(ctx: &mut CommandContext, args: &str) -> CmdResult {
    let inner = args.trim();
    if inner.is_empty() {
        return Err("Usage: explain <constraint-command> [args]".into());
    }
    let outcome = dry_run(ctx, inner);
    let tag = if outcome.accepted { "accepts" } else { "rejects" };
    let msg = if outcome.message.is_empty() {
        format!("'{}': {} (no further detail)", inner, tag)
    } else {
        format!("'{}': {} -- {}", inner, tag, outcome.message)
    };
    Ok(ok(msg))
}

