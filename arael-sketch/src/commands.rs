// Command system: parse and execute text commands for the sketch.
// Decoupled from GUI -- operates on CommandContext which holds sketch state.

use std::collections::HashMap;
use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::Action;
use crate::geometry::{arc_start_pos, arc_end_pos};
use crate::history::History;
use crate::tools::Selection;

// ---------------------------------------------------------------------------
// CommandContext: GUI-free state for command execution
// ---------------------------------------------------------------------------

pub struct CommandContext {
    pub sketch: Sketch,
    pub history: History,
    pub selection: Vec<Selection>,
    pub session_vars: HashMap<String, f64>,
    pub session_vecs: HashMap<String, vect2d>,
    pub session_names: HashMap<String, String>, // variable -> entity name aliases
    pub cursor: Option<vect2d>,
    pub status_error: Option<String>,
    pub last_cost: f64,
    pub dof: Option<usize>,
    pub cached_dof: Option<usize>,
    pub skip_dof_check: bool,
    // View state (used by center/zoom commands; GUI overrides with real values)
    pub scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub pending_fit: bool,
    /// Commands blocked in this context (e.g. "save", "load" for MCP).
    pub blocked_commands: Vec<&'static str>,
}

impl CommandContext {
    pub fn new() -> Self {
        let sketch = Sketch::new();
        let history = History::new(&sketch);
        CommandContext {
            sketch, history,
            selection: Vec::new(),
            session_vars: HashMap::new(),
            session_vecs: HashMap::new(),
            session_names: HashMap::new(),
            cursor: None,
            status_error: None,
            last_cost: 0.0,
            dof: None,
            cached_dof: None,
            skip_dof_check: false,
            scale: 80.0,
            offset_x: 400.0,
            offset_y: 300.0,
            pending_fit: false,
            blocked_commands: Vec::new(),
        }
    }

    pub fn with_sketch(sketch: Sketch) -> Self {
        let history = History::new(&sketch);
        CommandContext {
            sketch, history,
            selection: Vec::new(),
            session_vars: HashMap::new(),
            session_vecs: HashMap::new(),
            session_names: HashMap::new(),
            cursor: None,
            status_error: None,
            last_cost: 0.0,
            dof: None,
            cached_dof: None,
            skip_dof_check: false,
            scale: 80.0,
            offset_x: 400.0,
            offset_y: 300.0,
            pending_fit: false,
            blocked_commands: Vec::new(),
        }
    }

    /// Set skip_dof_check for the duration of a constraint command.
    fn set_force(&mut self, force: bool) {
        self.skip_dof_check = force;
    }

    pub fn begin_group(&mut self) {
        self.history.begin_group();
    }
}

/// Validate and apply a constraint action on a sketch.
/// Returns Ok(new_cost) on success, Err(message) on rejection.
/// Handles snapshot/restore, cost checking, and DOF checking.
pub fn validate_and_apply_constraint(
    sketch: &mut Sketch,
    action: &Action,
    skip_dof_check: bool,
    cached_dof: &mut Option<usize>,
) -> Result<f64, String> {
    use arael::simple_lm::LmProblem;

    let snapshot = bincode::serialize(sketch).ok();
    let old_cost = {
        let mut params = Vec::new();
        sketch.serialize64(&mut params);
        sketch.calc_cost(&params)
    };

    // Skip DOF check for internal/non-constraining actions
    let should_check_dof = !skip_dof_check && match action {
        Action::UpdateDimension { .. } => false,
        Action::AddDimension { derived: true, .. } => false,
        Action::ApplyCoincidentPP { a, .. } => !sketch.points.get(*a).map_or(false, |p| p.helper),
        Action::ApplyCoincidentLP1 { point, .. } | Action::ApplyCoincidentLP2 { point, .. } =>
            !sketch.points.get(*point).map_or(false, |p| p.helper),
        Action::ApplyCoincidentArcCenter { point, .. } | Action::ApplyCoincidentArcStart { point, .. } |
        Action::ApplyCoincidentArcEnd { point, .. } =>
            !sketch.points.get(*point).map_or(false, |p| p.helper),
        _ => true,
    };

    let old_dof = if should_check_dof {
        Some(cached_dof.unwrap_or_else(|| compute_dof_count(sketch)))
    } else {
        None
    };

    action.apply(sketch);
    sketch.dedup_constraints();

    // Quick cost check
    let quick_cost = {
        let mut params = Vec::new();
        sketch.serialize64(&mut params);
        sketch.calc_cost(&params)
    };
    let new_cost = if quick_cost <= old_cost + 1e-6 {
        quick_cost
    } else {
        sketch.solve().end_cost
    };

    // Cost rejection
    if new_cost > old_cost + 1e-3 {
        if let Some(ref snap) = snapshot {
            if let Ok(restored) = bincode::deserialize(snap) {
                *sketch = restored;
                return Err(format!(
                    "Constraint rejected: could not satisfy all constraints (cost {:.4} -> {:.4})",
                    old_cost, new_cost));
            }
        }
    }

    // DOF rejection
    if let Some(old_dof) = old_dof {
        let new_dof = compute_dof_count(sketch);
        if new_dof >= old_dof {
            if let Some(ref snap) = snapshot {
                if let Ok(restored) = bincode::deserialize(snap) {
                    *sketch = restored;
                    *cached_dof = Some(old_dof);
                    return Err(format!(
                        "Constraint rejected: DOF unchanged at {}. Constraint is redundant or degenerate. Use 'force' to override.",
                        new_dof));
                }
            }
        }
        *cached_dof = Some(new_dof);
    } else {
        *cached_dof = None;
    }

    Ok(new_cost)
}

impl CommandContext {

    /// Execute an action: apply to sketch and record in history.
    /// For constraint actions: validates by solving, checking cost, and optionally checking DOF.
    pub fn exec(&mut self, action: Action) {
        self.status_error = None;

        if action.is_constraint_action() {
            match validate_and_apply_constraint(
                &mut self.sketch, &action, self.skip_dof_check, &mut self.cached_dof)
            {
                Ok(new_cost) => {
                    self.last_cost = new_cost;
                    self.history.push(action, &self.sketch);
                }
                Err(msg) => {
                    self.status_error = Some(msg);
                }
            }
        } else {
            action.apply(&mut self.sketch);
            self.sketch.dedup_constraints();
            self.cached_dof = None; // geometry changed, invalidate
            self.history.push(action, &self.sketch);
        }
    }
}

pub struct CommandResult {
    pub output: String,
    pub is_error: bool,
    pub no_echo: bool,
    pub markdown: bool,
}

fn ok(msg: impl Into<String>) -> CommandResult {
    CommandResult { output: msg.into(), is_error: false, no_echo: false, markdown: false }
}

/// Return ok or the status_error if the last exec was rejected.
/// Also resets skip_dof_check to false.
fn ok_or_status(ctx: &mut CommandContext, msg: impl Into<String>) -> CommandResult {
    ctx.skip_dof_check = false;
    if let Some(e) = ctx.status_error.take() {
        CommandResult { output: e, is_error: true, no_echo: false, markdown: false }
    } else {
        let m = msg.into();
        if m.is_empty() {
            CommandResult { output: m, is_error: false, no_echo: true, markdown: false }
        } else {
            CommandResult { output: m, is_error: false, no_echo: false, markdown: false }
        }
    }
}

fn err(msg: impl Into<String>) -> CommandResult {
    CommandResult { output: msg.into(), is_error: true, no_echo: false, markdown: false }
}

// ---------------------------------------------------------------------------
// Entity resolution
// ---------------------------------------------------------------------------

fn resolve_entity_name<'a>(ctx: &'a CommandContext, name: &'a str) -> &'a str {
    ctx.session_names.get(name).map(|s| s.as_str()).unwrap_or(name)
}

fn resolve_line(sketch: &Sketch, name: &str) -> Result<Ref<Line>, String> {
    for r in sketch.lines.refs() {
        if sketch.lines[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown line: {}", name))
}

fn resolve_point(sketch: &Sketch, name: &str) -> Result<Ref<Point>, String> {
    for r in sketch.points.refs() {
        if sketch.points[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown point: {}", name))
}

fn resolve_arc(sketch: &Sketch, name: &str) -> Result<Ref<Arc>, String> {
    for r in sketch.arcs.refs() {
        if sketch.arcs[r].name == name { return Ok(r); }
    }
    Err(format!("Unknown arc: {}", name))
}

/// Resolve an endpoint reference like "L0.p1", "P0", "A0.center"
fn resolve_endpoint_pos(sketch: &Sketch, name: &str) -> Result<vect2d, String> {
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
        } else if entity.starts_with('A') {
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

fn eval_context(sketch: &Sketch) -> HashMap<String, f64> {
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
        vars.insert(format!("{}.diameter", a.name), a.radius.value * 2.0);
        vars.insert(format!("{}.start_angle", a.name), a.start_angle.value);
        vars.insert(format!("{}.end_angle", a.name), a.end_angle.value);
        vars.insert(format!("{}.sweep", a.name), (a.end_angle.value - a.start_angle.value).to_degrees());
        vars.insert(format!("{}.start.x", a.name), a.center.value.x + a.radius.value * a.start_angle.value.cos());
        vars.insert(format!("{}.start.y", a.name), a.center.value.y + a.radius.value * a.start_angle.value.sin());
        vars.insert(format!("{}.end.x", a.name), a.center.value.x + a.radius.value * a.end_angle.value.cos());
        vars.insert(format!("{}.end.y", a.name), a.center.value.y + a.radius.value * a.end_angle.value.sin());
    }
    for d in &sketch.dimensions {
        vars.insert(d.name.clone(), d.value);
    }
    for p in &sketch.user_params {
        vars.insert(p.name.clone(), p.value);
    }
    vars
}

fn eval_expr_with(sketch: &Sketch, expr_str: &str, extra: &HashMap<String, f64>) -> Result<f64, String> {
    let parsed = arael_sym::parse(expr_str).map_err(|e| e.msg)?;
    let mut ctx = eval_context(sketch);
    for (k, v) in extra { ctx.insert(k.clone(), *v); }
    let vars: HashMap<&str, f64> = ctx.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    parsed.eval(&vars)
}

fn eval_expr(sketch: &Sketch, expr_str: &str) -> Result<f64, String> {
    eval_expr_with(sketch, expr_str, &HashMap::new())
}

// ---------------------------------------------------------------------------
// Geometric functions (return coordinates or scalars)
// ---------------------------------------------------------------------------

fn eval_geo_coord(sketch: &Sketch, call: &str) -> Option<Result<vect2d, String>> {
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

fn eval_geo_scalar(sketch: &Sketch, call: &str) -> Option<Result<f64, String>> {
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

fn parse_coord(ctx: &CommandContext, arg: &str, cursor: Option<vect2d>) -> Result<vect2d, String> {
    let sketch = &ctx.sketch;
    let arg = arg.trim();
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
    // Cursor keyword
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
fn parse_vec_expr(ctx: &CommandContext, expr: &str) -> Result<vect2d, String> {
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
fn find_func_call(s: &str) -> Option<(usize, usize)> {
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
fn extract_identifiers(s: &str) -> Vec<(usize, usize, &str)> {
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

#[derive(Clone, Copy, PartialEq)]
enum EndpointRef {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
}

fn resolve_endpoint_ref(sketch: &Sketch, name: &str) -> Result<EndpointRef, String> {
    if let Some((entity, field)) = name.split_once('.') {
        if entity.starts_with('L') {
            let r = resolve_line(sketch, entity)?;
            match field {
                "p1" => return Ok(EndpointRef::LineP1(r)),
                "p2" => return Ok(EndpointRef::LineP2(r)),
                _ => return Err(format!("Line endpoint must be p1 or p2: {}", name)),
            }
        } else if entity.starts_with('A') {
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
fn substitute_aliases(ctx: &CommandContext, input: &str) -> String {
    if ctx.session_names.is_empty() { return input.to_string(); }
    let mut result = input.to_string();
    for (alias, real_name) in &ctx.session_names {
        if alias == "_" && !input.contains('_') { continue; }
        // Word-boundary replacement: alias must not be part of a larger identifier
        let mut new = String::new();
        let mut rest = result.as_str();
        while let Some(pos) = rest.find(alias.as_str()) {
            let before = pos > 0 && rest.as_bytes()[pos - 1].is_ascii_alphanumeric();
            let after_pos = pos + alias.len();
            let after = after_pos < rest.len()
                && (rest.as_bytes()[after_pos].is_ascii_alphanumeric() || rest.as_bytes()[after_pos] == b'_');
            new.push_str(&rest[..pos]);
            if before || after {
                new.push_str(alias);
            } else {
                new.push_str(real_name);
            }
            rest = &rest[after_pos..];
        }
        new.push_str(rest);
        result = new;
    }
    result
}

pub fn execute(ctx: &mut CommandContext, input: &str) -> Vec<CommandResult> {
    let mut results = Vec::new();
    for cmd in input.split(';') {
        let cmd = cmd.trim();
        if cmd.is_empty() { continue; }
        let mut r = execute_one(ctx, cmd);
        if r.is_error && !r.output.starts_with('>') {
            r.output = format!("'{}': {}", cmd, r.output);
        }
        results.push(r);
    }
    if results.is_empty() {
        results.push(ok(""));
    }
    results
}

fn execute_one(ctx: &mut CommandContext, input: &str) -> CommandResult {
    let input = input.trim();

    // Strip trailing "force" keyword — skips DOF check on constraint commands
    let (input, force) = strip_force(input);
    ctx.skip_dof_check = force;

    // Comments
    if input.starts_with('#') {
        return CommandResult { output: String::new(), is_error: false, no_echo: true, markdown: false };
    }

    // Assignment: "name = command args" or "let name = ..."
    let assign_input = input.strip_prefix("let ").map(|s| (true, s)).unwrap_or((false, input));
    if let Some((lhs, rhs)) = assign_input.1.split_once('=') {
        let var_name = lhs.trim();
        let rhs = rhs.trim();
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
                "add_line" | "add_point" | "add_circle" | "add_arc" | "offset_line" | "offset");
            if is_command {
                let result = execute_one(ctx, rhs);
                if !result.is_error {
                    if let Some(entity_name) = ctx.session_names.get("_").cloned() {
                        ctx.session_names.insert(var_name.to_string(), entity_name);
                    }
                }
                return result;
            }
            // Otherwise treat as scalar/vector assignment (existing let behavior)
            return cmd_let(ctx, &format!("{} = {}", var_name, rhs));
        }
    }

    // Substitute session_names aliases in arguments
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0];

    // Check if command is blocked in this context
    if ctx.blocked_commands.iter().any(|&b| b == cmd) {
        return err(format!("'{}' is not allowed in this context", cmd));
    }
    let raw_args = if parts.len() > 1 { parts[1].trim() } else { "" };

    // Replace known aliases in args (word-boundary aware)
    let args_str = substitute_aliases(ctx, raw_args);
    let args_str = args_str.as_str();

    match cmd {
        "add_line" => cmd_add_line(ctx, args_str),
        "add_point" => cmd_add_point(ctx, args_str),
        "add_circle" => cmd_add_circle(ctx, args_str),
        "add_arc" => cmd_add_arc(ctx, args_str),
        "offset_line" | "offset" => cmd_offset_line(ctx, args_str),
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
        "midpoint" => cmd_midpoint(ctx, args_str),
        "symmetry" => cmd_symmetry(ctx, args_str),
        "point_on" => cmd_point_on(ctx, args_str),
        "length" => cmd_length(ctx, args_str),
        "radius" => cmd_radius(ctx, args_str),
        "sweep" => cmd_sweep(ctx, args_str),
        "angle" => cmd_angle(ctx, args_str),
        "distance" => cmd_distance(ctx, args_str),
        "remove_dim" => cmd_remove_dim(ctx, args_str),
        "remove_constraint" | "rc" => cmd_remove_constraint(ctx, args_str),
        "lock" => cmd_lock(ctx, args_str),
        "unlock" => cmd_unlock(ctx, args_str),
        "param" => cmd_param(ctx, args_str),
        "del_param" => cmd_del_param(ctx, args_str),
        "rename_param" => cmd_rename_param(ctx, args_str),
        "style" => cmd_style(ctx, args_str),
        "select" => cmd_select(ctx, args_str),
        "deselect" => cmd_deselect(ctx, args_str),
        "print" => cmd_print(ctx, args_str),
        "info" => cmd_info(ctx, args_str),
        "list" => cmd_list(ctx, args_str),
        "find" => cmd_find(ctx, args_str),
        "dof" => cmd_dof(ctx, args_str),
        "cost" => ok(format!("Cost: {:.6}", ctx.last_cost)),
        "undo" => cmd_undo(ctx, args_str),
        "redo" => cmd_redo(ctx, args_str),
        "history" => cmd_history(ctx, args_str),
        "goto" => cmd_goto(ctx, args_str),
        "center" => cmd_center(ctx, args_str),
        "zoom" => cmd_zoom(ctx, args_str),
        "msg" => CommandResult {
            output: args_str.replace("\\n", "\n"), is_error: false, no_echo: true, markdown: true,
        },
        "cursor" => cmd_cursor(ctx, args_str),
        "dim_pos" => cmd_dim_pos(ctx, args_str),
        "set_derived" => cmd_set_derived(ctx, args_str),
        "set_driven" => cmd_set_driven(ctx, args_str),
        "clear" => { ctx.sketch = Sketch::new(); ctx.history = crate::history::History::new(&ctx.sketch); ok("Cleared") },
        "let" => cmd_let(ctx, args_str),
        "save" => cmd_save(ctx, args_str),
        "load" => cmd_load(ctx, args_str),
        "help" => cmd_help(args_str),
        "ai" => ok("AI assistant not yet configured. Use --mcp to enable MCP server for external AI agents."),
        _ if cmd.starts_with('!') => ok("AI assistant not yet configured. Use --mcp to enable MCP server for external AI agents."),
        _ => err(format!("Unknown command: {}. Type 'help' for commands.", cmd)),
    }
}

// ---------------------------------------------------------------------------
// Geometry commands
// ---------------------------------------------------------------------------

const SNAP_THRESHOLD: f64 = 1e-3;

fn snap_near(a: vect2d, b: vect2d) -> bool {
    (a.x - b.x).abs() < SNAP_THRESHOLD && (a.y - b.y).abs() < SNAP_THRESHOLD
}

/// Auto-connect endpoints of the last created line to nearby existing endpoints.
fn auto_coincident_line(ctx: &mut CommandContext, line_ref: Ref<Line>) -> Vec<String> {
    let mut actions: Vec<(Action, String)> = Vec::new();
    let l = &ctx.sketch.lines[line_ref];
    let p1 = l.p1.value;
    let p2 = l.p2.value;
    let this_name = l.name.clone();

    for r in ctx.sketch.lines.refs() {
        if r == line_ref { continue; }
        let other = &ctx.sketch.lines[r];
        if snap_near(p1, other.p1.value) {
            actions.push((Action::ApplyCoincidentLL11 { a: line_ref, b: r },
                format!("{}.p1={}.p1", this_name, other.name)));
        } else if snap_near(p1, other.p2.value) {
            actions.push((Action::ApplyCoincidentLL12 { a: line_ref, b: r },
                format!("{}.p1={}.p2", this_name, other.name)));
        }
        if snap_near(p2, other.p1.value) {
            actions.push((Action::ApplyCoincidentLL21 { a: line_ref, b: r },
                format!("{}.p2={}.p1", this_name, other.name)));
        } else if snap_near(p2, other.p2.value) {
            actions.push((Action::ApplyCoincidentLL22 { a: line_ref, b: r },
                format!("{}.p2={}.p2", this_name, other.name)));
        }
    }
    for r in ctx.sketch.points.refs() {
        let pt = &ctx.sketch.points[r];
        if pt.helper { continue; }
        if snap_near(p1, pt.pos.value) {
            actions.push((Action::ApplyCoincidentLP1 { line: line_ref, point: r },
                format!("{}.p1={}", this_name, pt.name)));
        }
        if snap_near(p2, pt.pos.value) {
            actions.push((Action::ApplyCoincidentLP2 { line: line_ref, point: r },
                format!("{}.p2={}", this_name, pt.name)));
        }
    }
    for r in ctx.sketch.arcs.refs() {
        let arc = &ctx.sketch.arcs[r];
        let ac = arc.center.value;
        let a_start = arc_start_pos(arc);
        let a_end = arc_end_pos(arc);
        if snap_near(p1, ac) {
            actions.push((Action::ApplyCoincidentLP1ArcCenter { line: line_ref, arc: r },
                format!("{}.p1={}.center", this_name, arc.name)));
        }
        if snap_near(p1, a_start) {
            actions.push((Action::ApplyCoincidentLP1ArcStart { line: line_ref, arc: r },
                format!("{}.p1={}.start", this_name, arc.name)));
        }
        if snap_near(p1, a_end) {
            actions.push((Action::ApplyCoincidentLP1ArcEnd { line: line_ref, arc: r },
                format!("{}.p1={}.end", this_name, arc.name)));
        }
        if snap_near(p2, ac) {
            actions.push((Action::ApplyCoincidentLP2ArcCenter { line: line_ref, arc: r },
                format!("{}.p2={}.center", this_name, arc.name)));
        }
        if snap_near(p2, a_start) {
            actions.push((Action::ApplyCoincidentLP2ArcStart { line: line_ref, arc: r },
                format!("{}.p2={}.start", this_name, arc.name)));
        }
        if snap_near(p2, a_end) {
            actions.push((Action::ApplyCoincidentLP2ArcEnd { line: line_ref, arc: r },
                format!("{}.p2={}.end", this_name, arc.name)));
        }
    }
    let mut connected = Vec::new();
    let saved = ctx.skip_dof_check;
    ctx.skip_dof_check = true; // auto-coincident is positional, don't DOF-check
    for (action, desc) in actions {
        ctx.exec(action);
        connected.push(desc);
    }
    ctx.skip_dof_check = saved;
    connected
}

/// Auto-connect arc endpoints to nearby existing geometry.
/// center_only=true for circles (start/end are edge points, not snap targets).
fn auto_coincident_arc(ctx: &mut CommandContext, arc_ref: Ref<Arc>, center_only: bool) -> Vec<String> {
    let mut actions: Vec<(Action, String)> = Vec::new();
    let arc = &ctx.sketch.arcs[arc_ref];
    let center = arc.center.value;
    let start = arc_start_pos(arc);
    let end = arc_end_pos(arc);
    let this_name = arc.name.clone();

    // Check against line endpoints
    for r in ctx.sketch.lines.refs() {
        let line = &ctx.sketch.lines[r];
        let lp1 = line.p1.value;
        let lp2 = line.p2.value;
        if snap_near(center, lp1) {
            actions.push((Action::ApplyCoincidentLP1ArcCenter { line: r, arc: arc_ref },
                format!("{}.center={}.p1", this_name, line.name)));
        }
        if snap_near(center, lp2) {
            actions.push((Action::ApplyCoincidentLP2ArcCenter { line: r, arc: arc_ref },
                format!("{}.center={}.p2", this_name, line.name)));
        }
        if !center_only {
            if snap_near(start, lp1) {
                actions.push((Action::ApplyCoincidentLP1ArcStart { line: r, arc: arc_ref },
                    format!("{}.start={}.p1", this_name, line.name)));
            }
            if snap_near(start, lp2) {
                actions.push((Action::ApplyCoincidentLP2ArcStart { line: r, arc: arc_ref },
                    format!("{}.start={}.p2", this_name, line.name)));
            }
            if snap_near(end, lp1) {
                actions.push((Action::ApplyCoincidentLP1ArcEnd { line: r, arc: arc_ref },
                    format!("{}.end={}.p1", this_name, line.name)));
            }
            if snap_near(end, lp2) {
                actions.push((Action::ApplyCoincidentLP2ArcEnd { line: r, arc: arc_ref },
                    format!("{}.end={}.p2", this_name, line.name)));
            }
        }
    }

    // Check against other arc endpoints
    for r in ctx.sketch.arcs.refs() {
        if r == arc_ref { continue; }
        let other = &ctx.sketch.arcs[r];
        let oc = other.center.value;
        let os = arc_start_pos(other);
        let oe = arc_end_pos(other);
        if snap_near(center, oc) {
            actions.push((Action::ApplyConcentric { a: arc_ref, b: r },
                format!("{}.center={}.center", this_name, other.name)));
        }
        if snap_near(center, os) {
            actions.push((Action::ApplyCoincidentArcCenterStart { a: arc_ref, b: r },
                format!("{}.center={}.start", this_name, other.name)));
        }
        if snap_near(center, oe) {
            actions.push((Action::ApplyCoincidentArcCenterEnd { a: arc_ref, b: r },
                format!("{}.center={}.end", this_name, other.name)));
        }
        if !center_only {
            if snap_near(start, oc) {
                actions.push((Action::ApplyCoincidentArcStartCenter { a: arc_ref, b: r },
                    format!("{}.start={}.center", this_name, other.name)));
            }
            if snap_near(start, os) {
                actions.push((Action::ApplyCoincidentArcStartStart { a: arc_ref, b: r },
                    format!("{}.start={}.start", this_name, other.name)));
            }
            if snap_near(start, oe) {
                actions.push((Action::ApplyCoincidentArcStartEnd { a: arc_ref, b: r },
                    format!("{}.start={}.end", this_name, other.name)));
            }
            if snap_near(end, oc) {
                actions.push((Action::ApplyCoincidentArcEndCenter { a: arc_ref, b: r },
                    format!("{}.end={}.center", this_name, other.name)));
            }
            if snap_near(end, os) {
                actions.push((Action::ApplyCoincidentArcEndStart { a: arc_ref, b: r },
                    format!("{}.end={}.start", this_name, other.name)));
            }
            if snap_near(end, oe) {
                actions.push((Action::ApplyCoincidentArcEndEnd { a: arc_ref, b: r },
                    format!("{}.end={}.end", this_name, other.name)));
            }
        }
    }

    // Check against free points (skip helpers)
    for r in ctx.sketch.points.refs() {
        let pt = &ctx.sketch.points[r];
        if pt.helper { continue; }
        if snap_near(center, pt.pos.value) {
            actions.push((Action::ApplyCoincidentArcCenter { point: r, arc: arc_ref },
                format!("{}.center={}", this_name, pt.name)));
        }
        if !center_only {
            if snap_near(start, pt.pos.value) {
                actions.push((Action::ApplyCoincidentArcStart { point: r, arc: arc_ref },
                    format!("{}.start={}", this_name, pt.name)));
            }
            if snap_near(end, pt.pos.value) {
                actions.push((Action::ApplyCoincidentArcEnd { point: r, arc: arc_ref },
                    format!("{}.end={}", this_name, pt.name)));
            }
        }
    }

    let mut connected = Vec::new();
    let saved = ctx.skip_dof_check;
    ctx.skip_dof_check = true; // auto-coincident is positional, don't DOF-check
    for (action, desc) in actions {
        ctx.exec(action);
        connected.push(desc);
    }
    ctx.skip_dof_check = saved;
    connected
}

fn cmd_add_line(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let nocursor = tokens.last() == Some(&"nocursor");
    if nocursor { tokens.pop(); }
    let noconnect = tokens.last() == Some(&"noconnect");
    if noconnect { tokens.pop(); }
    let (p1, p2) = match tokens.len() {
        2 => {
            let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) {
                Ok(p) => p, Err(e) => return err(e),
            };
            let p2 = match parse_coord(ctx, tokens[1], Some(p1)) {
                Ok(p) => p, Err(e) => return err(e),
            };
            (p1, p2)
        }
        1 => {
            let prev = match ctx.cursor {
                Some(p) => p,
                None => return err("No previous point. Use: add_line x1,y1 x2,y2"),
            };
            let p2 = match parse_coord(ctx, tokens[0], Some(prev)) {
                Ok(p) => p, Err(e) => return err(e),
            };
            (prev, p2)
        }
        _ => return err("Usage: add_line x1,y1 x2,y2  or  add_line x2,y2  or  add_line @dx,dy"),
    };
    ctx.begin_group();
    ctx.exec(Action::AddLine { p1, p2 });
    let line_ref = ctx.sketch.lines.refs().last().unwrap();
    let name = ctx.sketch.lines[line_ref].name.clone();
    if !nocursor { ctx.cursor = Some(p2); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: ({:.2},{:.2})-({:.2},{:.2})", name, p1.x, p1.y, p2.x, p2.y);
    if !noconnect {
        let connected = auto_coincident_line(ctx, line_ref);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    ok(msg)
}

fn cmd_add_point(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let nocursor = tokens.last() == Some(&"nocursor");
    if nocursor { tokens.pop(); }
    if tokens.len() != 1 { return err("Usage: add_point x,y [nocursor]"); }
    let pos = match parse_coord(ctx, tokens[0], ctx.cursor) {
        Ok(p) => p, Err(e) => return err(e),
    };
    ctx.begin_group();
    ctx.exec(Action::AddPoint { pos });
    let name = ctx.sketch.points.refs().last().map(|r| ctx.sketch.points[r].name.clone()).unwrap_or_default();
    if !nocursor { ctx.cursor = Some(pos); }
    ctx.session_names.insert("_".into(), name.clone());
    ok(format!("Added {}: ({:.2},{:.2})", name, pos.x, pos.y))
}

fn cmd_add_circle(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let nocursor = tokens.last() == Some(&"nocursor");
    if nocursor { tokens.pop(); }
    let noconnect = tokens.last() == Some(&"noconnect");
    if noconnect { tokens.pop(); }
    if tokens.len() != 2 {
        return err("Usage: add_circle cx,cy radius [noconnect] [nocursor]");
    }
    let center = match parse_coord(ctx, tokens[0], ctx.cursor) {
        Ok(p) => p, Err(e) => return err(e),
    };
    let r = match eval_expr(&ctx.sketch, tokens[1]) {
        Ok(v) => v, Err(e) => return err(e),
    };
    let edge = vect2d::new(center.x + r, center.y);
    ctx.begin_group();
    ctx.exec(Action::AddCircle { center, edge });
    let arc_ref = ctx.sketch.arcs.refs().last().unwrap();
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(center); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}: center=({:.2},{:.2}) r={:.2}", name, center.x, center.y, r);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, true);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    ok(msg)
}

fn cmd_delete(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    if name.starts_with('L') {
        let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::DeleteLine { line: r });
        ok(format!("Deleted {}", name))
    } else if name.starts_with('P') {
        let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::DeletePoint { point: r });
        ok(format!("Deleted {}", name))
    } else if name.starts_with('A') {
        let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::DeleteArc { arc: r });
        ok(format!("Deleted {}", name))
    } else {
        err(format!("Unknown entity: {}", name))
    }
}

// ---------------------------------------------------------------------------
// Constraint commands
// ---------------------------------------------------------------------------

/// Strip trailing "force" keyword from args, return (cleaned_args, is_force).
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

fn cmd_horizontal(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut lines = Vec::new();
    for name in args.split_whitespace() {
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => lines.push(r),
            Err(e) => return err(e),
        }
    }
    if lines.is_empty() { return err("Usage: horizontal L0 [L1 ...]"); }
    for &r in &lines {
        if ctx.sketch.lines[r].constraints.horizontal {
            return err(format!("{} is already horizontal", ctx.sketch.lines[r].name));
        }
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyHorizontal { lines });
    ok_or_status(ctx, "Applied horizontal")
}

fn cmd_vertical(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut lines = Vec::new();
    for name in args.split_whitespace() {
        match resolve_line(&ctx.sketch, name) {
            Ok(r) => lines.push(r),
            Err(e) => return err(e),
        }
    }
    if lines.is_empty() { return err("Usage: vertical L0 [L1 ...]"); }
    for &r in &lines {
        if ctx.sketch.lines[r].constraints.vertical {
            return err(format!("{} is already vertical", ctx.sketch.lines[r].name));
        }
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyVertical { lines });
    ok_or_status(ctx, "Applied vertical")
}

fn cmd_parallel(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: parallel L0 L1"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain a line parallel to itself"); }
    if ctx.sketch.parallel.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Parallel constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyParallel { a, b });
    ok_or_status(ctx, "Applied parallel")
}

fn cmd_perpendicular(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: perpendicular L0 L1"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain a line perpendicular to itself"); }
    if ctx.sketch.perpendicular.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Perpendicular constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyPerpendicular { a, b });
    ok_or_status(ctx, "Applied perpendicular")
}

fn cmd_equal(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: equal L0 L1  or  equal A0 A1"); }
    if tokens[0].starts_with('L') && tokens[1].starts_with('L') {
        let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if a == b { return err("Cannot constrain a line equal to itself"); }
        let exists = ctx.sketch.equal_length.iter().any(|c|
            (c.a == a && c.b == b) || (c.a == b && c.b == a));
        if exists { return err("Equal length constraint already exists"); }
        ctx.begin_group();
        ctx.exec(Action::ApplyEqualLength { a, b });
        ok_or_status(ctx, "Applied equal length")
    } else if tokens[0].starts_with('A') && tokens[1].starts_with('A') {
        let a = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if a == b { return err("Cannot constrain an arc equal to itself"); }
        let exists = ctx.sketch.equal_radius.iter().any(|c|
            (c.a == a && c.b == b) || (c.a == b && c.b == a));
        if exists { return err("Equal radius constraint already exists"); }
        ctx.begin_group();
        ctx.exec(Action::ApplyEqualRadius { a, b });
        ok_or_status(ctx, "Applied equal radius")
    } else {
        err("equal needs two lines or two arcs")
    }
}

fn cmd_collinear(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: collinear L0 L1"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain a line collinear with itself"); }
    if ctx.sketch.collinear.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Collinear constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyCollinear { a, b });
    ok_or_status(ctx, "Applied collinear")
}

fn cmd_tangent(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: tangent L0 A0  or  tangent A0 A1"); }
    if tokens[0].starts_with('L') && tokens[1].starts_with('A') {
        let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let arc = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if ctx.sketch.tangent_la.iter().any(|c| c.line == line && c.arc == arc) {
            return err("Tangent constraint already exists");
        }
        ctx.begin_group();
        ctx.exec(Action::ApplyTangentLA { line, arc });
        ok_or_status(ctx, "Applied tangent")
    } else if tokens[0].starts_with('A') && tokens[1].starts_with('A') {
        let a = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        if a == b { return err("Cannot constrain an arc tangent to itself"); }
        if ctx.sketch.tangent_aa.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
            return err("Tangent constraint already exists");
        }
        ctx.begin_group();
        ctx.exec(Action::ApplyTangentAA { a, b });
        ok_or_status(ctx, "Applied tangent")
    } else {
        err("tangent needs line+arc or arc+arc")
    }
}

fn cmd_coincident(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: coincident L0.p2 L1.p1"); }
    let a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain an endpoint coincident with itself"); }
    use EndpointRef::*;
    let s = &ctx.sketch;
    // Check for existing equivalent coincident constraint
    let exists = match (a, b) {
        (Point(a), Point(b)) => s.coincident_pp.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => s.coincident_lp1.iter().any(|c| c.line == l && c.point == p),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => s.coincident_lp2.iter().any(|c| c.line == l && c.point == p),
        (LineP1(a), LineP1(b)) => s.coincident_ll11.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (LineP1(a), LineP2(b)) => s.coincident_ll12.iter().any(|c| c.a == a && c.b == b)
            || s.coincident_ll21.iter().any(|c| c.a == b && c.b == a),
        (LineP2(a), LineP1(b)) => s.coincident_ll21.iter().any(|c| c.a == a && c.b == b)
            || s.coincident_ll12.iter().any(|c| c.a == b && c.b == a),
        (LineP2(a), LineP2(b)) => s.coincident_ll22.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)),
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => s.coincident_arc_center.iter().any(|c| c.point == p && c.arc == arc),
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => s.coincident_arc_start.iter().any(|c| c.point == p && c.arc == arc),
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => s.coincident_arc_end.iter().any(|c| c.point == p && c.arc == arc),
        (LineP1(line), ArcCenter(arc)) | (ArcCenter(arc), LineP1(line)) => s.coincident_lp1_arc_center.iter().any(|c| c.line == line && c.arc == arc),
        (LineP2(line), ArcCenter(arc)) | (ArcCenter(arc), LineP2(line)) => s.coincident_lp2_arc_center.iter().any(|c| c.line == line && c.arc == arc),
        (LineP1(line), ArcStart(arc)) | (ArcStart(arc), LineP1(line)) => s.coincident_lp1_arc_start.iter().any(|c| c.line == line && c.arc == arc),
        (LineP2(line), ArcStart(arc)) | (ArcStart(arc), LineP2(line)) => s.coincident_lp2_arc_start.iter().any(|c| c.line == line && c.arc == arc),
        (LineP1(line), ArcEnd(arc)) | (ArcEnd(arc), LineP1(line)) => s.coincident_lp1_arc_end.iter().any(|c| c.line == line && c.arc == arc),
        (LineP2(line), ArcEnd(arc)) | (ArcEnd(arc), LineP2(line)) => s.coincident_lp2_arc_end.iter().any(|c| c.line == line && c.arc == arc),
        _ => false,
    };
    if exists { return err("Coincident constraint already exists"); }
    let action = match (a, b) {
        (Point(a), Point(b)) => Action::ApplyCoincidentPP { a, b },
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => Action::ApplyCoincidentLP1 { line: l, point: p },
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => Action::ApplyCoincidentLP2 { line: l, point: p },
        (LineP1(a), LineP1(b)) => Action::ApplyCoincidentLL11 { a, b },
        (LineP1(a), LineP2(b)) => Action::ApplyCoincidentLL12 { a, b },
        (LineP2(a), LineP1(b)) => Action::ApplyCoincidentLL21 { a, b },
        (LineP2(a), LineP2(b)) => Action::ApplyCoincidentLL22 { a, b },
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => Action::ApplyCoincidentArcCenter { point: p, arc },
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => Action::ApplyCoincidentArcStart { point: p, arc },
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => Action::ApplyCoincidentArcEnd { point: p, arc },
        (LineP1(line), ArcCenter(arc)) | (ArcCenter(arc), LineP1(line)) => Action::ApplyCoincidentLP1ArcCenter { line, arc },
        (LineP2(line), ArcCenter(arc)) | (ArcCenter(arc), LineP2(line)) => Action::ApplyCoincidentLP2ArcCenter { line, arc },
        (LineP1(line), ArcStart(arc)) | (ArcStart(arc), LineP1(line)) => Action::ApplyCoincidentLP1ArcStart { line, arc },
        (LineP2(line), ArcStart(arc)) | (ArcStart(arc), LineP2(line)) => Action::ApplyCoincidentLP2ArcStart { line, arc },
        (LineP1(line), ArcEnd(arc)) | (ArcEnd(arc), LineP1(line)) => Action::ApplyCoincidentLP1ArcEnd { line, arc },
        (LineP2(line), ArcEnd(arc)) | (ArcEnd(arc), LineP2(line)) => Action::ApplyCoincidentLP2ArcEnd { line, arc },
        _ => return err("Unsupported coincident combination"),
    };
    ctx.begin_group();
    ctx.exec(action);
    ok_or_status(ctx, "Applied coincident")
}

fn cmd_concentric(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: concentric A0 A1"); }
    let a = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_arc(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    if a == b { return err("Cannot constrain an arc concentric with itself"); }
    if ctx.sketch.concentric.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a)) {
        return err("Concentric constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplyConcentric { a, b });
    ok_or_status(ctx, "Applied concentric")
}

// ---------------------------------------------------------------------------
// Dimension commands
// ---------------------------------------------------------------------------

/// Find existing dimension index matching the given kind.
/// For Angle, matches regardless of supplement flag. For PointPointDistance, matches either order.
fn find_existing_dimension(sketch: &Sketch, kind: &DimensionKind) -> Option<usize> {
    sketch.dimensions.iter().position(|d| match (&d.kind, kind) {
        (DimensionKind::Angle(da, db, _), DimensionKind::Angle(a, b, _)) =>
            (*da == *a && *db == *b) || (*da == *b && *db == *a),
        (DimensionKind::PointPointDistance(a1, b1), DimensionKind::PointPointDistance(a2, b2)) =>
            (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2),
        (a, b) => a == b,
    })
}

fn cmd_length(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    if is_derived { tokens.pop(); }
    if tokens.len() == 1 && is_derived {
        let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let l = &ctx.sketch.lines[line];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let len = (dx * dx + dy * dy).sqrt();
        let kind = DimensionKind::LineLength(line);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: len, expr: None });
            return ok_or_status(ctx, format!("Updated {} derived length = ({:.4})", name, len));
        }
        ctx.exec(Action::AddDimension { kind, value: len, expr: None, derived: true });
        return ok_or_status(ctx, format!("Derived {} length = ({:.4})", tokens[0], len));
    }
    if tokens.len() != 2 { return err("Usage: length L0 5.0 [derived]"); }
    let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = DimensionKind::LineLength(line);
    let val_str = tokens[1].trim().trim_matches('"');
    if let Ok(value) = val_str.parse::<f64>() {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value, expr: None });
            return ok_or_status(ctx, format!("Updated {} length = {}", name, value));
        }
        ctx.exec(Action::AddDimension { kind, value, expr: None, derived: is_derived });
        let prefix = if is_derived { "Derived" } else { "Set" };
        ok_or_status(ctx, format!("{} {} length = {}", prefix, tokens[0], value))
    } else {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: 0.0, expr: Some(val_str.to_string()) });
            return ok_or_status(ctx, format!("Updated {} length = {}", name, val_str));
        }
        ctx.exec(Action::AddDimension { kind, value: 0.0, expr: Some(val_str.to_string()), derived: is_derived });
        ok_or_status(ctx, format!("Set {} length = {}", tokens[0], val_str))
    }
}

fn cmd_radius(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    if is_derived { tokens.pop(); }
    if tokens.len() == 1 && is_derived {
        let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let r = ctx.sketch.arcs[arc].radius.value;
        let kind = DimensionKind::ArcRadius(arc);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: r, expr: None });
            return ok_or_status(ctx, format!("Updated {} derived radius = ({:.4})", name, r));
        }
        ctx.exec(Action::AddDimension { kind, value: r, expr: None, derived: true });
        return ok_or_status(ctx, format!("Derived {} radius = ({:.4})", tokens[0], r));
    }
    if tokens.len() != 2 { return err("Usage: radius A0 1.5 [derived]"); }
    let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = DimensionKind::ArcRadius(arc);
    let val_str = tokens[1].trim().trim_matches('"');
    if let Ok(value) = val_str.parse::<f64>() {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value, expr: None });
            return ok_or_status(ctx, format!("Updated {} radius = {}", name, value));
        }
        ctx.exec(Action::AddDimension { kind, value, expr: None, derived: is_derived });
        let prefix = if is_derived { "Derived" } else { "Set" };
        ok_or_status(ctx, format!("{} {} radius = {}", prefix, tokens[0], value))
    } else {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: 0.0, expr: Some(val_str.to_string()) });
            return ok_or_status(ctx, format!("Updated {} radius = {}", name, val_str));
        }
        ctx.exec(Action::AddDimension { kind, value: 0.0, expr: Some(val_str.to_string()), derived: is_derived });
        ok_or_status(ctx, format!("Set {} radius = {}", tokens[0], val_str))
    }
}

fn cmd_sweep(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    if is_derived { tokens.pop(); }
    if tokens.is_empty() { return err("Usage: sweep A0 180 [derived]"); }
    let arc = match resolve_arc(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    if ctx.sketch.arcs[arc].closed {
        return err("Cannot set sweep on a full circle (angles are fixed)");
    }
    let kind = DimensionKind::ArcSweep(arc);
    if tokens.len() == 1 && is_derived {
        let a = &ctx.sketch.arcs[arc];
        let sweep_deg = arael::utils::rad2deg(a.end_angle.value - a.start_angle.value);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: sweep_deg, expr: None });
            return ok_or_status(ctx, format!("Updated {} derived sweep = ({:.4})", name, sweep_deg));
        }
        ctx.exec(Action::AddDimension { kind, value: sweep_deg, expr: None, derived: true });
        return ok_or_status(ctx, format!("Derived {} sweep = ({:.4})", tokens[0], sweep_deg));
    }
    if tokens.len() != 2 { return err("Usage: sweep A0 180 [derived]"); }
    let val_str = tokens[1].trim().trim_matches('"');
    if let Ok(value) = val_str.parse::<f64>() {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value, expr: None });
            return ok_or_status(ctx, format!("Updated {} sweep = {}", name, value));
        }
        ctx.exec(Action::AddDimension { kind, value, expr: None, derived: is_derived });
        let prefix = if is_derived { "Derived" } else { "Set" };
        ok_or_status(ctx, format!("{} {} sweep = {}", prefix, tokens[0], value))
    } else {
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: 0.0, expr: Some(val_str.to_string()) });
            return ok_or_status(ctx, format!("Updated {} sweep = {}", name, val_str));
        }
        ctx.exec(Action::AddDimension { kind, value: 0.0, expr: Some(val_str.to_string()), derived: is_derived });
        ok_or_status(ctx, format!("Set {} sweep = {}", tokens[0], val_str))
    }
}

// ---------------------------------------------------------------------------
// Lock/unlock
// ---------------------------------------------------------------------------

fn cmd_lock(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: lock P0  or  lock L0.p1  or  lock L0.p1 x,y"); }
    let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let pos = if tokens.len() > 1 {
        match parse_coord(ctx, tokens[1], None) { Ok(p) => p, Err(e) => return err(e) }
    } else {
        match resolve_endpoint_pos(&ctx.sketch, tokens[0]) { Ok(p) => p, Err(e) => return err(e) }
    };
    let action = match ep {
        EndpointRef::Point(p) => Action::LockPoint { point: p, pos },
        EndpointRef::LineP1(l) => Action::LockLineP1 { line: l, pos },
        EndpointRef::LineP2(l) => Action::LockLineP2 { line: l, pos },
        EndpointRef::ArcCenter(a) => Action::LockArcCenter { arc: a, pos },
        _ => return err("Can only lock points, line endpoints, and arc centers"),
    };
    ctx.begin_group();
    ctx.exec(action);
    ok(format!("Locked {} at ({:.2},{:.2})", tokens[0], pos.x, pos.y))
}

fn cmd_unlock(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    let ep = match resolve_endpoint_ref(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
    let action = match ep {
        EndpointRef::Point(p) => Action::UnlockPoint { point: p },
        EndpointRef::LineP1(l) => Action::UnlockLineP1 { line: l },
        EndpointRef::LineP2(l) => Action::UnlockLineP2 { line: l },
        EndpointRef::ArcCenter(a) => Action::UnlockArcCenter { arc: a },
        _ => return err("Can only unlock points, line endpoints, and arc centers"),
    };
    ctx.begin_group();
    ctx.exec(action);
    ok(format!("Unlocked {}", name))
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

fn cmd_param(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
    if tokens.len() != 2 { return err("Usage: param name value"); }
    let name = tokens[0].trim();
    let expr = tokens[1].trim();
    // Check if param exists -> update
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == name) {
        ctx.begin_group();
        ctx.exec(Action::UpdateUserParam { index: idx, name: name.to_string(), expr_str: expr.to_string() });
        ctx.sketch.update_expr_dim_values();
        let val = ctx.sketch.user_params.iter().find(|p| p.name == name).map(|p| p.value).unwrap_or(0.0);
        ok(format!("Updated {} = {} ({:.4})", name, expr, val))
    } else {
        if let Err(e) = ctx.sketch.validate_param_name(name, None) {
            return err(e);
        }
        ctx.begin_group();
        ctx.exec(Action::AddUserParam { name: name.to_string(), expr_str: expr.to_string() });
        ctx.sketch.update_expr_dim_values();
        let val = ctx.sketch.user_params.iter().find(|p| p.name == name).map(|p| p.value).unwrap_or(0.0);
        ok(format!("Added {} = {} ({:.4})", name, expr, val))
    }
}

fn cmd_del_param(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == name) {
        ctx.begin_group();
        ctx.exec(Action::RemoveUserParam { index: idx });
        ok(format!("Deleted parameter {}", name))
    } else {
        err(format!("Unknown parameter: {}", name))
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

fn cmd_style(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: style L0 [solid|dashed|dashdot]"); }
    let name = tokens[0];
    if tokens.len() == 1 {
        // Query
        if name.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            return ok(format!("{}: {}", name, ctx.sketch.lines[r].style.name()));
        } else if name.starts_with('A') {
            let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            return ok(format!("{}: {}", name, ctx.sketch.arcs[r].style.name()));
        }
        return err("Style applies to lines and arcs");
    }
    let style = match LineStyle::from_name(tokens[1]) {
        Some(s) => s,
        None => return err(format!("Unknown style '{}'. Use: solid, dashed, dashdot", tokens[1])),
    };
    if name.starts_with('L') {
        let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::SetStyleLine { line: r, style });
        ok(format!("{}: {}", name, style.name()))
    } else if name.starts_with('A') {
        let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        ctx.begin_group();
        ctx.exec(Action::SetStyleArc { arc: r, style });
        ok(format!("{}: {}", name, style.name()))
    } else {
        err("Style applies to lines and arcs")
    }
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

fn cmd_select(ctx: &mut CommandContext, args: &str) -> CommandResult {
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
                Err(e) => return err(e),
            };
            ctx.selection.push(sel);
        } else if name.starts_with('L') {
            let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            ctx.selection.push(Selection::Line(r));
        } else if name.starts_with('P') {
            let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            ctx.selection.push(Selection::Point(r));
        } else if name.starts_with('A') {
            let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
            ctx.selection.push(Selection::Arc(r));
        } else {
            return err(format!("Cannot select: {}", name));
        }
    }
    ok(format!("Selected {} entities", args.split_whitespace().count()))
}

fn cmd_deselect(ctx: &mut CommandContext, _args: &str) -> CommandResult {
    ctx.selection.clear();
    ok("Selection cleared")
}

// ---------------------------------------------------------------------------
// Print / Info / List
// ---------------------------------------------------------------------------

fn cmd_print(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    // Try geometric coordinate function first (prints x,y)
    if let Some(result) = eval_geo_coord(&ctx.sketch, args) {
        return match result {
            Ok(v) => ok(format!("({:.6}, {:.6})", v.x, v.y)),
            Err(e) => err(e),
        };
    }
    // Try geometric scalar function
    if let Some(result) = eval_geo_scalar(&ctx.sketch, args) {
        return match result {
            Ok(v) => ok(format!("{:.6}", v)),
            Err(e) => err(e),
        };
    }
    // Try session variable
    if let Some(v) = ctx.session_vars.get(args) {
        return ok(format!("{:.6}", v));
    }
    if let Some(v) = ctx.session_vecs.get(args) {
        return ok(format!("({:.6}, {:.6})", v.x, v.y));
    }
    // Eval as expression
    match eval_expr_with(&ctx.sketch, args, &ctx.session_vars) {
        Ok(v) => ok(format!("{:.6}", v)),
        Err(e) => err(format!("Eval error: {}", e)),
    }
}

/// Get all constraints that mention a given entity name.
fn constraints_for(sketch: &Sketch, name: &str) -> Vec<String> {
    sketch.list_constraints().into_iter()
        .filter(|c| c.split_whitespace().any(|w| w == name || w.starts_with(&format!("{}.", name))))
        .collect()
}

fn cmd_info(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    // Endpoint info: L0.p1, L0.p2, A0.center, etc.
    if name.contains('.') {
        if let Ok(pos) = resolve_endpoint_pos(&ctx.sketch, name) {
            let mut s = format!("{}: ({:.4}, {:.4})", name, pos.x, pos.y);
            // Check lock status
            if let Some((entity, field)) = name.split_once('.') {
                if entity.starts_with('L') {
                    if let Ok(r) = resolve_line(&ctx.sketch, entity) {
                        let l = &ctx.sketch.lines[r];
                        if field == "p1" && !l.p1.optimize { s += " [locked]"; }
                        if field == "p2" && !l.p2.optimize { s += " [locked]"; }
                    }
                }
            }
            let cstrs = constraints_for(&ctx.sketch, name);
            if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
            return ok(s);
        }
    }
    if name.starts_with('L') && !name.contains('.') {
        let r = match resolve_line(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        let l = &ctx.sketch.lines[r];
        let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
        let mut s = format!("{}: ({:.4},{:.4})-({:.4},{:.4}) len={:.4} style={}",
            l.name, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y, len, l.style.name());
        if !l.p1.optimize { s += " [p1 locked]"; }
        if !l.p2.optimize { s += " [p2 locked]"; }
        let cstrs = constraints_for(&ctx.sketch, name);
        if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
        ok(s)
    } else if name.starts_with('P') && !name.contains('.') {
        let r = match resolve_point(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        let p = &ctx.sketch.points[r];
        let locked = p.constraints.has_fix_x || p.constraints.has_fix_y;
        let mut s = format!("{}: ({:.4},{:.4}){}", p.name, p.pos.value.x, p.pos.value.y,
            if locked { " [locked]" } else { "" });
        let cstrs = constraints_for(&ctx.sketch, name);
        if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
        ok(s)
    } else if name.starts_with('A') && !name.contains('.') {
        let r = match resolve_arc(&ctx.sketch, name) { Ok(r) => r, Err(e) => return err(e) };
        let a = &ctx.sketch.arcs[r];
        let sp = crate::geometry::arc_start_pos(a);
        let ep = crate::geometry::arc_end_pos(a);
        let mut s = format!("{}: center=({:.4},{:.4}) r={:.4} angles={:.1}..{:.1} start=({:.4},{:.4}) end=({:.4},{:.4}) {}",
            a.name, a.center.value.x, a.center.value.y, a.radius.value,
            a.start_angle.value.to_degrees(), a.end_angle.value.to_degrees(),
            sp.x, sp.y, ep.x, ep.y,
            if a.closed { "[circle]" } else { "" });
        let cstrs = constraints_for(&ctx.sketch, name);
        if !cstrs.is_empty() { s += &format!("\n  constraints: {}", cstrs.join(", ")); }
        ok(s)
    } else if name.starts_with('d') {
        if let Some(d) = ctx.sketch.dimensions.iter().find(|d| d.name == name) {
            let expr = d.expr_str.as_deref().unwrap_or("(numeric)");
            let flags = match (d.derived, d.broken) {
                (true, true) => " derived broken",
                (true, false) => " derived",
                (false, true) => " broken",
                (false, false) => "",
            };
            ok(format!("{}: value={:.4} expr={} offset={:.2} along={:.2}{}",
                d.name, d.value, expr, d.offset.y, d.text_along, flags))
        } else {
            err(format!("Unknown dimension: {}", name))
        }
    } else {
        if let Some(p) = ctx.sketch.user_params.iter().find(|p| p.name == name) {
            ok(format!("{}: value={:.4} expr={}{}", p.name, p.value, p.expr_str,
                if p.broken { " broken" } else { "" }))
        } else {
            err(format!("Unknown entity: {}", name))
        }
    }
}

fn cmd_list(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let filter = args.trim();
    if !filter.is_empty() && !matches!(filter, "all" | "lines" | "points" | "arcs" | "dims" | "params" | "constraints") {
        return err(format!("Unknown filter: {}. Use: all, lines, points, arcs, dims, params, constraints", filter));
    }
    let mut lines = Vec::new();
    let show_all = filter.is_empty() || filter == "all";

    if show_all || filter == "lines" {
        for r in ctx.sketch.lines.refs() {
            let l = &ctx.sketch.lines[r];
            let len = ((l.p2.value.x - l.p1.value.x).powi(2) + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
            lines.push(format!("{}: ({:.2},{:.2})-({:.2},{:.2}) len={:.2}",
                l.name, l.p1.value.x, l.p1.value.y, l.p2.value.x, l.p2.value.y, len));
        }
    }
    if show_all || filter == "points" {
        for r in ctx.sketch.points.refs() {
            let p = &ctx.sketch.points[r];
            if p.helper { continue; }
            lines.push(format!("{}: ({:.2},{:.2})", p.name, p.pos.value.x, p.pos.value.y));
        }
    }
    if show_all || filter == "arcs" {
        for r in ctx.sketch.arcs.refs() {
            let a = &ctx.sketch.arcs[r];
            if a.closed {
                lines.push(format!("{}: center=({:.2},{:.2}) r={:.2} [circle]",
                    a.name, a.center.value.x, a.center.value.y, a.radius.value));
            } else {
                let sp = crate::geometry::arc_start_pos(a);
                let ep = crate::geometry::arc_end_pos(a);
                lines.push(format!("{}: center=({:.2},{:.2}) r={:.2} start=({:.2},{:.2}) end=({:.2},{:.2})",
                    a.name, a.center.value.x, a.center.value.y, a.radius.value,
                    sp.x, sp.y, ep.x, ep.y));
            }
        }
    }
    if show_all || filter == "dims" {
        for d in &ctx.sketch.dimensions {
            let expr = d.expr_str.as_deref().unwrap_or("");
            let tag = if d.derived { " derived" } else { "" };
            lines.push(format!("{}: {:.4} {}{}", d.name, d.value, expr, tag));
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
        ok("(empty)")
    } else {
        ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Undo/Redo/History
// ---------------------------------------------------------------------------

fn cmd_undo(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let n: usize = args.trim().parse().unwrap_or(1);
    for _ in 0..n {
        if let Some(s) = ctx.history.undo() {
            ctx.sketch = s;
        } else {
            return ok("Nothing to undo");
        }
    }
    ctx.sketch.solve();
    ok(format!("Undone {} step(s)", n))
}

fn cmd_redo(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let n: usize = args.trim().parse().unwrap_or(1);
    for _ in 0..n {
        if let Some(s) = ctx.history.redo() {
            ctx.sketch = s;
        } else {
            return ok("Nothing to redo");
        }
    }
    ctx.sketch.solve();
    ok(format!("Redone {} step(s)", n))
}

fn cmd_history(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let n: usize = args.trim().parse().unwrap_or(usize::MAX);
    let groups = ctx.history.group_list();
    let total = groups.len();
    let start = if n < total { total - n } else { 0 };
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
    ok(lines.join("\n"))
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

fn cmd_center(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    if args.is_empty() {
        // Fit all -- delegate to pending_fit
        ctx.pending_fit = true;
        return ok("Fitting all");
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
                } else { return err(format!("Unknown: {}", args)); }
            } else { return err(format!("Cannot resolve: {}", args)); }
        }
    };
    // Center the view on pos
    ctx.offset_x = 400.0 - pos.x as f32 * ctx.scale;
    ctx.offset_y = 300.0 + pos.y as f32 * ctx.scale;
    ok(format!("Centered on ({:.2},{:.2})", pos.x, pos.y))
}

fn cmd_zoom(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    match args {
        "+" => { ctx.scale *= 1.5; ok(format!("Zoom: {:.1}", ctx.scale)) }
        "-" => { ctx.scale /= 1.5; ok(format!("Zoom: {:.1}", ctx.scale)) }
        _ => {
            if let Ok(v) = args.parse::<f32>() {
                ctx.scale = v.clamp(1e-4, 1e7);
                ok(format!("Zoom: {:.1}", ctx.scale))
            } else {
                err("Usage: zoom +  or  zoom -  or  zoom 2.0")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Additional geometry
// ---------------------------------------------------------------------------

fn cmd_add_arc(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let nocursor = tokens.last() == Some(&"nocursor");
    if nocursor { tokens.pop(); }
    let noconnect = tokens.last() == Some(&"noconnect");
    if noconnect { tokens.pop(); }
    if tokens.len() != 3 { return err("Usage: add_arc x1,y1 x2,y2 xm,ym [noconnect] [nocursor]"); }
    let p1 = match parse_coord(ctx, tokens[0], ctx.cursor) { Ok(p) => p, Err(e) => return err(e) };
    let p2 = match parse_coord(ctx, tokens[1], Some(p1)) { Ok(p) => p, Err(e) => return err(e) };
    let pm = match parse_coord(ctx, tokens[2], None) { Ok(p) => p, Err(e) => return err(e) };
    ctx.begin_group();
    ctx.exec(Action::AddArc { start: p1, end: p2, mid: pm, swapped: false });
    let arc_ref = ctx.sketch.arcs.refs().last().unwrap();
    let name = ctx.sketch.arcs[arc_ref].name.clone();
    if !nocursor { ctx.cursor = Some(p2); }
    ctx.session_names.insert("_".into(), name.clone());
    let mut msg = format!("Added {}", name);
    if !noconnect {
        let connected = auto_coincident_arc(ctx, arc_ref, false);
        if !connected.is_empty() {
            msg += &format!(" [connected: {}]", connected.join(", "));
        }
    }
    ok(msg)
}

fn cmd_offset_line(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: offset_line L0 distance"); }
    let line = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let d = match eval_expr(&ctx.sketch, tokens[1]) { Ok(v) => v, Err(e) => return err(e) };
    let l = &ctx.sketch.lines[line];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 { return err("Zero-length line"); }
    let nx = -dy / len * d;
    let ny = dx / len * d;
    let p1 = vect2d::new(l.p1.value.x + nx, l.p1.value.y + ny);
    let p2 = vect2d::new(l.p2.value.x + nx, l.p2.value.y + ny);
    ctx.begin_group();
    ctx.exec(Action::AddLine { p1, p2 });
    let name = ctx.sketch.lines.refs().last().map(|r| ctx.sketch.lines[r].name.clone()).unwrap_or_default();
    ctx.cursor = Some(p2);
    ctx.session_names.insert("_".into(), name.clone());
    ok(format!("Added {} (offset of {} by {})", name, tokens[0], d))
}

// ---------------------------------------------------------------------------
// Additional constraints
// ---------------------------------------------------------------------------

fn cmd_midpoint(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: midpoint P0 L0  or  midpoint L0.p1 L1"); }
    let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let line = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let s = &ctx.sketch;
    let exists = match ep {
        EndpointRef::Point(p) => s.midpoint.iter().any(|c| c.point == p && c.line == line),
        EndpointRef::LineP1(l) => s.midpoint_lp1.iter().any(|c| c.line == l && c.target == line),
        EndpointRef::LineP2(l) => s.midpoint_lp2.iter().any(|c| c.line == l && c.target == line),
        EndpointRef::ArcStart(a) => s.midpoint_arc_start.iter().any(|c| c.arc == a && c.line == line),
        EndpointRef::ArcEnd(a) => s.midpoint_arc_end.iter().any(|c| c.arc == a && c.line == line),
        _ => false,
    };
    if exists { return err("Midpoint constraint already exists"); }
    let action = match ep {
        EndpointRef::Point(p) => Action::ApplyMidpoint { point: p, line },
        EndpointRef::LineP1(l) => Action::ApplyMidpointLP1 { line: l, target: line },
        EndpointRef::LineP2(l) => Action::ApplyMidpointLP2 { line: l, target: line },
        EndpointRef::ArcStart(a) => Action::ApplyMidpointArcStart { arc: a, line },
        EndpointRef::ArcEnd(a) => Action::ApplyMidpointArcEnd { arc: a, line },
        _ => return err("First arg must be a point or endpoint"),
    };
    ctx.begin_group();
    ctx.exec(action);
    ok_or_status(ctx, "Applied midpoint")
}

/// Resolve a name to a Ref<Point>, creating a helper point + coincident if it's an endpoint ref.
fn resolve_as_point(ctx: &mut CommandContext, name: &str) -> Result<Ref<Point>, String> {
    // Try as a standalone point first
    if let Ok(r) = resolve_point(&ctx.sketch, name) {
        return Ok(r);
    }
    // Try as endpoint ref — create helper point + coincident constraint
    let ep = resolve_endpoint_ref(&ctx.sketch, name)?;
    let pos = resolve_endpoint_pos(&ctx.sketch, name)?;
    let hp = ctx.sketch.add_helper_point(pos);
    match ep {
        EndpointRef::Point(p) => {
            ctx.exec(Action::ApplyCoincidentPP { a: hp, b: p });
        }
        EndpointRef::LineP1(l) => {
            ctx.exec(Action::ApplyCoincidentLP1 { line: l, point: hp });
        }
        EndpointRef::LineP2(l) => {
            ctx.exec(Action::ApplyCoincidentLP2 { line: l, point: hp });
        }
        EndpointRef::ArcCenter(a) => {
            ctx.exec(Action::ApplyCoincidentArcCenter { point: hp, arc: a });
        }
        EndpointRef::ArcStart(a) => {
            ctx.exec(Action::ApplyCoincidentArcStart { point: hp, arc: a });
        }
        EndpointRef::ArcEnd(a) => {
            ctx.exec(Action::ApplyCoincidentArcEnd { point: hp, arc: a });
        }
    }
    Ok(hp)
}

fn cmd_symmetry(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 3 { return err("Usage: symmetry L0 L1 L2 | symmetry P0 L0 P1 | symmetry L0.p1 L1 L0.p2"); }
    // Try point/endpoint + line + point/endpoint symmetry
    let mid_is_line = resolve_line(&ctx.sketch, tokens[1]).is_ok();
    let first_is_pointlike = resolve_point(&ctx.sketch, tokens[0]).is_ok()
        || resolve_endpoint_ref(&ctx.sketch, tokens[0]).is_ok();
    let third_is_pointlike = resolve_point(&ctx.sketch, tokens[2]).is_ok()
        || resolve_endpoint_ref(&ctx.sketch, tokens[2]).is_ok();
    if mid_is_line && first_is_pointlike && third_is_pointlike {
        // Note: duplicate check is hard here because resolve_as_point creates helper points.
        // We skip duplicate check for symmetry_pp — the solver handles redundancy gracefully.
        ctx.begin_group();
        let a = match resolve_as_point(ctx, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let line = resolve_line(&ctx.sketch, tokens[1]).unwrap();
        let c = match resolve_as_point(ctx, tokens[2]) { Ok(r) => r, Err(e) => return err(e) };
        ctx.exec(Action::ApplySymmetryPP { a, line, c });
        return ok_or_status(ctx, "Applied point symmetry");
    }
    // Fall back to line-line-line symmetry
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let c = match resolve_line(&ctx.sketch, tokens[2]) { Ok(r) => r, Err(e) => return err(e) };
    if ctx.sketch.symmetry_ll.iter().any(|s|
        s.b == b && ((s.a == a && s.c == c) || (s.a == c && s.c == a))) {
        return err("Symmetry constraint already exists");
    }
    ctx.begin_group();
    ctx.exec(Action::ApplySymmetryLL { a, b, c });
    ok_or_status(ctx, "Applied symmetry")
}

/// Which arc endpoint a helper point bridges to.
enum ArcEp { Center, Start, End }

/// Check if an arc endpoint already has a point_on_line constraint via a helper point.
fn has_arc_endpoint_on_line(s: &Sketch, arc: Ref<Arc>, ep: ArcEp, line: Ref<Line>) -> bool {
    // Find helper points bridged to this arc endpoint
    let bridged_points: Vec<Ref<Point>> = match ep {
        ArcEp::Center => s.coincident_arc_center.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::Start => s.coincident_arc_start.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::End => s.coincident_arc_end.iter()
            .filter(|c| c.arc == arc).map(|c| c.point).collect(),
    };
    // Check if any of those points are on the target line
    bridged_points.iter().any(|p| s.point_on_line.iter().any(|c| c.point == *p && c.line == line))
}

/// Check if an arc endpoint already has a point_on_arc constraint via a helper point.
fn has_arc_endpoint_on_arc(s: &Sketch, src: Ref<Arc>, ep: ArcEp, target: Ref<Arc>) -> bool {
    let bridged_points: Vec<Ref<Point>> = match ep {
        ArcEp::Center => s.coincident_arc_center.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::Start => s.coincident_arc_start.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::End => s.coincident_arc_end.iter()
            .filter(|c| c.arc == src).map(|c| c.point).collect(),
    };
    bridged_points.iter().any(|p| s.point_on_arc.iter().any(|c| c.point == *p && c.arc == target))
}

fn cmd_point_on(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: point_on P0 L0  or  point_on L0.p1 A0"); }
    let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let target = tokens[1];
    if target.starts_with('L') {
        let line = match resolve_line(&ctx.sketch, target) { Ok(r) => r, Err(e) => return err(e) };
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.point_on_line.iter().any(|c| c.point == p && c.line == line),
            EndpointRef::LineP1(l) => s.line_p1_on_line.iter().any(|c| c.a == l && c.b == line),
            EndpointRef::LineP2(l) => s.line_p2_on_line.iter().any(|c| c.a == l && c.b == line),
            EndpointRef::ArcCenter(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::Center, line),
            EndpointRef::ArcStart(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::Start, line),
            EndpointRef::ArcEnd(arc) => has_arc_endpoint_on_line(s, arc, ArcEp::End, line),
        };
        if exists { return err("Point-on-line constraint already exists"); }
        ctx.begin_group();
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyPointOnLine { point: p, line },
            EndpointRef::LineP1(l) => Action::ApplyLineP1OnLine { a: l, b: line },
            EndpointRef::LineP2(l) => Action::ApplyLineP2OnLine { a: l, b: line },
            EndpointRef::ArcCenter(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcCenter(a), line },
            EndpointRef::ArcStart(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcStart(a), line },
            EndpointRef::ArcEnd(a) => Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcEnd(a), line },
        };
        ctx.exec(action);
        ok_or_status(ctx, "Applied point-on-line")
    } else if target.starts_with('A') {
        let arc = match resolve_arc(&ctx.sketch, target) { Ok(r) => r, Err(e) => return err(e) };
        let s = &ctx.sketch;
        let exists = match ep {
            EndpointRef::Point(p) => s.point_on_arc.iter().any(|c| c.point == p && c.arc == arc),
            EndpointRef::LineP1(l) => s.line_p1_on_arc.iter().any(|c| c.line == l && c.arc == arc),
            EndpointRef::LineP2(l) => s.line_p2_on_arc.iter().any(|c| c.line == l && c.arc == arc),
            EndpointRef::ArcCenter(src) => has_arc_endpoint_on_arc(s, src, ArcEp::Center, arc),
            EndpointRef::ArcStart(src) => has_arc_endpoint_on_arc(s, src, ArcEp::Start, arc),
            EndpointRef::ArcEnd(src) => has_arc_endpoint_on_arc(s, src, ArcEp::End, arc),
        };
        if exists { return err("Point-on-arc constraint already exists"); }
        ctx.begin_group();
        let action = match ep {
            EndpointRef::Point(p) => Action::ApplyPointOnArc { point: p, arc },
            EndpointRef::LineP1(l) => Action::ApplyLineP1OnArc { line: l, arc },
            EndpointRef::LineP2(l) => Action::ApplyLineP2OnArc { line: l, arc },
            EndpointRef::ArcCenter(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcCenter(a), arc },
            EndpointRef::ArcStart(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcStart(a), arc },
            EndpointRef::ArcEnd(a) => Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcEnd(a), arc },
        };
        ctx.exec(action);
        ok_or_status(ctx, "Applied point-on-arc")
    } else {
        err("Second arg must be a line (L0) or arc (A0)")
    }
}

// ---------------------------------------------------------------------------
// Additional dimensions
// ---------------------------------------------------------------------------

fn cmd_angle(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    if is_derived { tokens.pop(); }

    // Compute current angle to decide which sector (direct or supplement) is closest
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

    if tokens.len() == 2 && is_derived {
        let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let (current_deg, _) = compute_angle(ctx, a, b);
        let kind = DimensionKind::Angle(a, b, false);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: current_deg, expr: None });
            return ok_or_status(ctx, format!("Updated {} derived angle = ({:.4})", name, current_deg));
        }
        ctx.exec(Action::AddDimension { kind, value: current_deg, expr: None, derived: true });
        return ok_or_status(ctx, format!("Derived angle {} {} = ({:.4})", tokens[0], tokens[1], current_deg));
    }

    if tokens.len() != 3 { return err("Usage: angle L0 L1 45 [derived]"); }
    let a = match resolve_line(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let b = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let val_str = tokens[2].trim().trim_matches('"');
    let (current_deg, supplement_deg) = compute_angle(ctx, a, b);

    if let Ok(value) = val_str.parse::<f64>() {
        let supplement = (value - supplement_deg).abs() < (value - current_deg).abs();
        let kind = DimensionKind::Angle(a, b, supplement);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value, expr: None });
            let sector = if supplement { "supplement" } else { "direct" };
            return ok_or_status(ctx, format!("Updated {} angle = {} ({})", name, value, sector));
        }
        ctx.exec(Action::AddDimension { kind, value, expr: None, derived: is_derived });
        let sector = if supplement { "supplement" } else { "direct" };
        let prefix = if is_derived { "Derived" } else { "Set" };
        ok_or_status(ctx, format!("{} angle {} {} = {} ({})", prefix, tokens[0], tokens[1], value, sector))
    } else {
        let value = eval_expr(&ctx.sketch, val_str).unwrap_or(0.0);
        let supplement = (value - supplement_deg).abs() < (value - current_deg).abs();
        let kind = DimensionKind::Angle(a, b, supplement);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: 0.0, expr: Some(val_str.to_string()) });
            let sector = if supplement { "supplement" } else { "direct" };
            return ok_or_status(ctx, format!("Updated {} angle = {} ({})", name, val_str, sector));
        }
        ctx.exec(Action::AddDimension { kind, value: 0.0, expr: Some(val_str.to_string()), derived: is_derived });
        let sector = if supplement { "supplement" } else { "direct" };
        ok_or_status(ctx, format!("Set angle {} {} = {} ({})", tokens[0], tokens[1], val_str, sector))
    }
}

fn cmd_distance(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let is_derived = tokens.last() == Some(&"derived");
    if is_derived { tokens.pop(); }

    fn to_dim_ep(ep: EndpointRef) -> DimensionEndpoint {
        match ep {
            EndpointRef::Point(p) => DimensionEndpoint::Point(p),
            EndpointRef::LineP1(l) => DimensionEndpoint::LineP1(l),
            EndpointRef::LineP2(l) => DimensionEndpoint::LineP2(l),
            EndpointRef::ArcCenter(a) => DimensionEndpoint::ArcCenter(a),
            EndpointRef::ArcStart(a) => DimensionEndpoint::ArcStart(a),
            EndpointRef::ArcEnd(a) => DimensionEndpoint::ArcEnd(a),
        }
    }

    // "distance L0.p1 L1.p2 derived" — measure-only, no value
    if tokens.len() == 2 && is_derived {
        let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let pa = resolve_endpoint_pos(&ctx.sketch, tokens[0]).unwrap();
        let pb = resolve_endpoint_pos(&ctx.sketch, tokens[1]).unwrap();
        let dx = pa.x - pb.x; let dy = pa.y - pb.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let kind = DimensionKind::PointPointDistance(to_dim_ep(ep_a), to_dim_ep(ep_b));
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: dist, expr: None });
            return ok_or_status(ctx, format!("Updated {} derived distance = ({:.4})", name, dist));
        }
        ctx.exec(Action::AddDimension { kind, value: dist, expr: None, derived: true });
        return ok_or_status(ctx, format!("Derived distance {} {} = ({:.4})", tokens[0], tokens[1], dist));
    }

    if tokens.len() != 3 { return err("Usage: distance L0.p1 L1.p2 5.0 [derived]  or  distance P0 L0 3.0 [derived]"); }
    let val_str = tokens[2].trim().trim_matches('"');
    let value = val_str.parse::<f64>().ok();
    let expr = if value.is_none() { Some(val_str.to_string()) } else { None };
    let val = value.unwrap_or(0.0);

    // Try point-line distance
    if (tokens[0].starts_with('P') || tokens[0].contains('.')) && tokens[1].starts_with('L') && !tokens[1].contains('.') {
        let ep = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
        let line = match resolve_line(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
        let kind = DimensionKind::PointLineDistance(to_dim_ep(ep), line);
        ctx.begin_group();
        if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
            let name = ctx.sketch.dimensions[idx].name.clone();
            ctx.exec(Action::UpdateDimension { index: idx, value: val, expr });
            return ok_or_status(ctx, format!("Updated {} distance = {}", name, tokens[2]));
        }
        ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived });
        let prefix = if is_derived { "Derived distance" } else { "Set distance" };
        return ok_or_status(ctx, format!("{} = {}", prefix, tokens[2]));
    }

    // Point-point distance
    let ep_a = match resolve_endpoint_ref(&ctx.sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
    let ep_b = match resolve_endpoint_ref(&ctx.sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
    let kind = DimensionKind::PointPointDistance(to_dim_ep(ep_a), to_dim_ep(ep_b));
    ctx.begin_group();
    if let Some(idx) = find_existing_dimension(&ctx.sketch, &kind) {
        let name = ctx.sketch.dimensions[idx].name.clone();
        ctx.exec(Action::UpdateDimension { index: idx, value: val, expr });
        return ok_or_status(ctx, format!("Updated {} distance = {}", name, tokens[2]));
    }
    ctx.exec(Action::AddDimension { kind, value: val, expr, derived: is_derived });
    let prefix = if is_derived { "Derived distance" } else { "Set distance" };
    ok_or_status(ctx, format!("{} = {}", prefix, tokens[2]))
}

fn cmd_remove_dim(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    if let Some(idx) = ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        ctx.begin_group();
        ctx.exec(Action::RemoveDimension { index: idx });
        ok(format!("Removed dimension {}", name))
    } else {
        err(format!("Unknown dimension: {}", name))
    }
}

/// Remove a coincident constraint between two endpoint refs.
fn remove_coincident(sketch: &mut Sketch, a: EndpointRef, b: EndpointRef) -> bool {
    use EndpointRef::*;
    macro_rules! retain_rm {
        ($vec:expr, $pred:expr) => {{ let b = $vec.len(); $vec.retain($pred); $vec.len() < b }};
    }
    match (a, b) {
        (Point(a), Point(b)) => retain_rm!(sketch.coincident_pp, |c| !((c.a == a && c.b == b) || (c.a == b && c.b == a))),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => retain_rm!(sketch.coincident_lp1, |c| !(c.line == l && c.point == p)),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => retain_rm!(sketch.coincident_lp2, |c| !(c.line == l && c.point == p)),
        (LineP1(a), LineP1(b)) => retain_rm!(sketch.coincident_ll11, |c| !((c.a == a && c.b == b) || (c.a == b && c.b == a))),
        (LineP1(a), LineP2(b)) => {
            retain_rm!(sketch.coincident_ll12, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_ll21, |c| !(c.a == b && c.b == a))
        }
        (LineP2(a), LineP1(b)) => {
            retain_rm!(sketch.coincident_ll21, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_ll12, |c| !(c.a == b && c.b == a))
        }
        (LineP2(a), LineP2(b)) => retain_rm!(sketch.coincident_ll22, |c| !((c.a == a && c.b == b) || (c.a == b && c.b == a))),
        (Point(p), ArcCenter(arc)) | (ArcCenter(arc), Point(p)) => retain_rm!(sketch.coincident_arc_center, |c| !(c.point == p && c.arc == arc)),
        (Point(p), ArcStart(arc)) | (ArcStart(arc), Point(p)) => retain_rm!(sketch.coincident_arc_start, |c| !(c.point == p && c.arc == arc)),
        (Point(p), ArcEnd(arc)) | (ArcEnd(arc), Point(p)) => retain_rm!(sketch.coincident_arc_end, |c| !(c.point == p && c.arc == arc)),
        (LineP1(line), ArcCenter(arc)) | (ArcCenter(arc), LineP1(line)) => retain_rm!(sketch.coincident_lp1_arc_center, |c| !(c.line == line && c.arc == arc)),
        (LineP2(line), ArcCenter(arc)) | (ArcCenter(arc), LineP2(line)) => retain_rm!(sketch.coincident_lp2_arc_center, |c| !(c.line == line && c.arc == arc)),
        (LineP1(line), ArcStart(arc)) | (ArcStart(arc), LineP1(line)) => retain_rm!(sketch.coincident_lp1_arc_start, |c| !(c.line == line && c.arc == arc)),
        (LineP2(line), ArcStart(arc)) | (ArcStart(arc), LineP2(line)) => retain_rm!(sketch.coincident_lp2_arc_start, |c| !(c.line == line && c.arc == arc)),
        (LineP1(line), ArcEnd(arc)) | (ArcEnd(arc), LineP1(line)) => retain_rm!(sketch.coincident_lp1_arc_end, |c| !(c.line == line && c.arc == arc)),
        (LineP2(line), ArcEnd(arc)) | (ArcEnd(arc), LineP2(line)) => retain_rm!(sketch.coincident_lp2_arc_end, |c| !(c.line == line && c.arc == arc)),
        // Arc-arc: check both constraint direction variants
        (ArcCenter(a), ArcStart(b)) => {
            retain_rm!(sketch.coincident_arc_center_start, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_arc_start_center, |c| !(c.a == b && c.b == a))
        }
        (ArcStart(a), ArcCenter(b)) => {
            retain_rm!(sketch.coincident_arc_start_center, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_arc_center_start, |c| !(c.a == b && c.b == a))
        }
        (ArcCenter(a), ArcEnd(b)) => {
            retain_rm!(sketch.coincident_arc_center_end, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_arc_end_center, |c| !(c.a == b && c.b == a))
        }
        (ArcEnd(a), ArcCenter(b)) => {
            retain_rm!(sketch.coincident_arc_end_center, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_arc_center_end, |c| !(c.a == b && c.b == a))
        }
        (ArcStart(a), ArcStart(b)) => retain_rm!(sketch.coincident_arc_start_start, |c| !((c.a == a && c.b == b) || (c.a == b && c.b == a))),
        (ArcStart(a), ArcEnd(b)) => {
            retain_rm!(sketch.coincident_arc_start_end, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_arc_end_start, |c| !(c.a == b && c.b == a))
        }
        (ArcEnd(a), ArcStart(b)) => {
            retain_rm!(sketch.coincident_arc_end_start, |c| !(c.a == a && c.b == b))
            || retain_rm!(sketch.coincident_arc_start_end, |c| !(c.a == b && c.b == a))
        }
        (ArcEnd(a), ArcEnd(b)) => retain_rm!(sketch.coincident_arc_end_end, |c| !((c.a == a && c.b == b) || (c.a == b && c.b == a))),
        _ => false,
    }
}

/// Remove a point-on-line constraint. For arc endpoints, removes the helper point's constraint.
fn remove_point_on_line(sketch: &mut Sketch, ep: EndpointRef, line: Ref<Line>) -> bool {
    use EndpointRef::*;
    match ep {
        Point(p) => { let b = sketch.point_on_line.len(); sketch.point_on_line.retain(|c| !(c.point == p && c.line == line)); sketch.point_on_line.len() < b }
        LineP1(l) => { let b = sketch.line_p1_on_line.len(); sketch.line_p1_on_line.retain(|c| !(c.a == l && c.b == line)); sketch.line_p1_on_line.len() < b }
        LineP2(l) => { let b = sketch.line_p2_on_line.len(); sketch.line_p2_on_line.retain(|c| !(c.a == l && c.b == line)); sketch.line_p2_on_line.len() < b }
        ArcCenter(arc) => remove_arc_ep_on_line(sketch, arc, ArcEp::Center, line),
        ArcStart(arc) => remove_arc_ep_on_line(sketch, arc, ArcEp::Start, line),
        ArcEnd(arc) => remove_arc_ep_on_line(sketch, arc, ArcEp::End, line),
    }
}

/// Remove a point-on-arc constraint. For arc endpoints, removes the helper point's constraint.
fn remove_point_on_arc(sketch: &mut Sketch, ep: EndpointRef, arc: Ref<Arc>) -> bool {
    use EndpointRef::*;
    match ep {
        Point(p) => { let b = sketch.point_on_arc.len(); sketch.point_on_arc.retain(|c| !(c.point == p && c.arc == arc)); sketch.point_on_arc.len() < b }
        LineP1(l) => { let b = sketch.line_p1_on_arc.len(); sketch.line_p1_on_arc.retain(|c| !(c.line == l && c.arc == arc)); sketch.line_p1_on_arc.len() < b }
        LineP2(l) => { let b = sketch.line_p2_on_arc.len(); sketch.line_p2_on_arc.retain(|c| !(c.line == l && c.arc == arc)); sketch.line_p2_on_arc.len() < b }
        ArcCenter(src) => remove_arc_ep_on_arc(sketch, src, ArcEp::Center, arc),
        ArcStart(src) => remove_arc_ep_on_arc(sketch, src, ArcEp::Start, arc),
        ArcEnd(src) => remove_arc_ep_on_arc(sketch, src, ArcEp::End, arc),
    }
}

/// Remove point_on_line for an arc endpoint that uses a helper point.
/// Finds bridged helper points, removes the point_on_line constraint on them.
/// cleanup_helper_points() (called by the caller) will remove orphan helpers.
fn remove_arc_ep_on_line(sketch: &mut Sketch, arc: Ref<Arc>, ep: ArcEp, line: Ref<Line>) -> bool {
    let bridged: Vec<Ref<Point>> = match ep {
        ArcEp::Center => sketch.coincident_arc_center.iter().filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::Start => sketch.coincident_arc_start.iter().filter(|c| c.arc == arc).map(|c| c.point).collect(),
        ArcEp::End => sketch.coincident_arc_end.iter().filter(|c| c.arc == arc).map(|c| c.point).collect(),
    };
    let before = sketch.point_on_line.len();
    sketch.point_on_line.retain(|c| !(bridged.contains(&c.point) && c.line == line));
    sketch.point_on_line.len() < before
}

/// Remove point_on_arc for an arc endpoint that uses a helper point.
fn remove_arc_ep_on_arc(sketch: &mut Sketch, src: Ref<Arc>, ep: ArcEp, target: Ref<Arc>) -> bool {
    let bridged: Vec<Ref<Point>> = match ep {
        ArcEp::Center => sketch.coincident_arc_center.iter().filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::Start => sketch.coincident_arc_start.iter().filter(|c| c.arc == src).map(|c| c.point).collect(),
        ArcEp::End => sketch.coincident_arc_end.iter().filter(|c| c.arc == src).map(|c| c.point).collect(),
    };
    let before = sketch.point_on_arc.len();
    sketch.point_on_arc.retain(|c| !(bridged.contains(&c.point) && c.arc == target));
    sketch.point_on_arc.len() < before
}

fn cmd_remove_constraint(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() < 2 { return err("Usage: remove_constraint L0 horizontal | remove_constraint L0 L1 parallel"); }

    let ctype = tokens.last().unwrap();
    let sketch = &mut ctx.sketch;
    let removed = match *ctype {
        "horizontal" => {
            let r = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            sketch.lines[r].constraints.horizontal = false;
            true
        }
        "vertical" => {
            let r = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            sketch.lines[r].constraints.vertical = false;
            true
        }
        "parallel" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let before = sketch.parallel.len();
            sketch.parallel.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
            sketch.parallel.len() < before
        }
        "perpendicular" | "perp" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let before = sketch.perpendicular.len();
            sketch.perpendicular.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
            sketch.perpendicular.len() < before
        }
        "equal" | "equal_length" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let before = sketch.equal_length.len();
            sketch.equal_length.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
            sketch.equal_length.len() < before
        }
        "collinear" if tokens.len() >= 3 => {
            let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let before = sketch.collinear.len();
            sketch.collinear.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
            sketch.collinear.len() < before
        }
        "tangent" if tokens.len() >= 3 => {
            if tokens[0].starts_with('L') && tokens[1].starts_with('A') {
                let line = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let arc = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let before = sketch.tangent_la.len();
                sketch.tangent_la.retain(|c| !(c.line == line && c.arc == arc));
                sketch.tangent_la.len() < before
            } else if tokens[0].starts_with('A') && tokens[1].starts_with('A') {
                let a = match resolve_arc(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let b = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let before = sketch.tangent_aa.len();
                sketch.tangent_aa.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
                sketch.tangent_aa.len() < before
            } else { false }
        }
        "concentric" if tokens.len() >= 3 => {
            let a = match resolve_arc(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let before = sketch.concentric.len();
            sketch.concentric.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
            sketch.concentric.len() < before
        }
        "lock" => {
            let ep = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            match ep {
                EndpointRef::Point(p) => {
                    sketch.points[p].constraints.has_fix_x = false;
                    sketch.points[p].constraints.has_fix_y = false;
                    true
                }
                EndpointRef::LineP1(l) => {
                    let val = sketch.lines[l].p1.value;
                    sketch.lines[l].p1 = arael::model::Param::new(val);
                    true
                }
                EndpointRef::LineP2(l) => {
                    let val = sketch.lines[l].p2.value;
                    sketch.lines[l].p2 = arael::model::Param::new(val);
                    true
                }
                _ => false,
            }
        }
        "equal_radius" if tokens.len() >= 3 => {
            let a = match resolve_arc(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_arc(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            let before = sketch.equal_radius.len();
            sketch.equal_radius.retain(|c| !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
            sketch.equal_radius.len() < before
        }
        "coincident" if tokens.len() >= 3 => {
            let a = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let b = match resolve_endpoint_ref(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            remove_coincident(sketch, a, b)
        }
        "point_on" if tokens.len() >= 3 => {
            let ep = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let target = tokens[1];
            if target.starts_with('L') || target.starts_with('l') {
                let line = match resolve_line(sketch, target) { Ok(r) => r, Err(e) => return err(e) };
                remove_point_on_line(sketch, ep, line)
            } else if target.starts_with('A') || target.starts_with('a') {
                let arc = match resolve_arc(sketch, target) { Ok(r) => r, Err(e) => return err(e) };
                remove_point_on_arc(sketch, ep, arc)
            } else { false }
        }
        "symmetry" if tokens.len() >= 4 => {
            // Try point-line-point: symmetry_pp has fields a (point), c (point), line
            let ep_a = resolve_endpoint_ref(sketch, tokens[0]);
            let ep_c = resolve_endpoint_ref(sketch, tokens[2]);
            if let (Ok(EndpointRef::Point(a)), Ok(EndpointRef::Point(c))) = (ep_a, ep_c) {
                let line = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let before = sketch.symmetry_pp.len();
                sketch.symmetry_pp.retain(|s| !((s.a == a && s.c == c && s.line == line) || (s.a == c && s.c == a && s.line == line)));
                sketch.symmetry_pp.len() < before
            } else {
                // 3-line form: a symmetric to c about b
                let a = match resolve_line(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
                let b = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
                let c = match resolve_line(sketch, tokens[2]) { Ok(r) => r, Err(e) => return err(e) };
                let before = sketch.symmetry_ll.len();
                sketch.symmetry_ll.retain(|s| !(s.b == b && ((s.a == a && s.c == c) || (s.a == c && s.c == a))));
                sketch.symmetry_ll.len() < before
            }
        }
        "midpoint" if tokens.len() >= 3 => {
            let ep = match resolve_endpoint_ref(sketch, tokens[0]) { Ok(r) => r, Err(e) => return err(e) };
            let line = match resolve_line(sketch, tokens[1]) { Ok(r) => r, Err(e) => return err(e) };
            use EndpointRef::*;
            match ep {
                Point(p) => { let b = sketch.midpoint.len(); sketch.midpoint.retain(|c| !(c.point == p && c.line == line)); sketch.midpoint.len() < b }
                LineP1(l) => { let b = sketch.midpoint_lp1.len(); sketch.midpoint_lp1.retain(|c| !(c.line == l && c.target == line)); sketch.midpoint_lp1.len() < b }
                LineP2(l) => { let b = sketch.midpoint_lp2.len(); sketch.midpoint_lp2.retain(|c| !(c.line == l && c.target == line)); sketch.midpoint_lp2.len() < b }
                ArcStart(a) => { let b = sketch.midpoint_arc_start.len(); sketch.midpoint_arc_start.retain(|c| !(c.arc == a && c.line == line)); sketch.midpoint_arc_start.len() < b }
                ArcEnd(a) => { let b = sketch.midpoint_arc_end.len(); sketch.midpoint_arc_end.retain(|c| !(c.arc == a && c.line == line)); sketch.midpoint_arc_end.len() < b }
                ArcCenter(_) => false,
            }
        }
        _ => { return err(format!("Unknown constraint type: {}. Use: horizontal, vertical, parallel, perpendicular, equal, equal_radius, collinear, tangent, concentric, coincident, point_on, symmetry, midpoint, lock", ctype)); }
    };
    if removed {
        sketch.cleanup_helper_points();
        sketch.solve();
        ok(format!("Removed {} constraint", ctype))
    } else {
        err(format!("Constraint not found"))
    }
}

// ---------------------------------------------------------------------------
// Rename param
// ---------------------------------------------------------------------------

fn cmd_rename_param(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 2 { return err("Usage: rename_param old_name new_name"); }
    let old = tokens[0];
    let new = tokens[1];
    if let Some(idx) = ctx.sketch.user_params.iter().position(|p| p.name == old) {
        if let Err(e) = ctx.sketch.validate_param_name(new, Some(idx)) {
            return err(e);
        }
        let expr = ctx.sketch.user_params[idx].expr_str.clone();
        ctx.begin_group();
        ctx.exec(Action::UpdateUserParam { index: idx, name: new.to_string(), expr_str: expr });
        ok(format!("Renamed {} -> {}", old, new))
    } else {
        err(format!("Unknown parameter: {}", old))
    }
}

// ---------------------------------------------------------------------------
// Find
// ---------------------------------------------------------------------------

fn cmd_find(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.is_empty() { return err("Usage: find x,y [radius]"); }
    let pos = match parse_coord(ctx, tokens[0], None) { Ok(p) => p, Err(e) => return err(e) };
    let radius = if tokens.len() > 1 {
        tokens[1].parse::<f64>().unwrap_or(1.0)
    } else { 1.0 };
    let r2 = radius * radius;
    let mut found = Vec::new();
    for r in ctx.sketch.points.refs() {
        let p = &ctx.sketch.points[r];
        if p.helper { continue; }
        let d2 = (p.pos.value.x - pos.x).powi(2) + (p.pos.value.y - pos.y).powi(2);
        if d2 <= r2 { found.push(format!("{} ({:.2},{:.2})", p.name, p.pos.value.x, p.pos.value.y)); }
    }
    for r in ctx.sketch.lines.refs() {
        let l = &ctx.sketch.lines[r];
        // Point-to-segment distance
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let len2 = dx * dx + dy * dy;
        let dist = if len2 < 1e-24 {
            ((l.p1.value.x - pos.x).powi(2) + (l.p1.value.y - pos.y).powi(2)).sqrt()
        } else {
            let t = (((pos.x - l.p1.value.x) * dx + (pos.y - l.p1.value.y) * dy) / len2).clamp(0.0, 1.0);
            let cx = l.p1.value.x + t * dx - pos.x;
            let cy = l.p1.value.y + t * dy - pos.y;
            (cx * cx + cy * cy).sqrt()
        };
        if dist <= radius { found.push(format!("{} (d={:.2})", l.name, dist)); }
    }
    for r in ctx.sketch.arcs.refs() {
        let a = &ctx.sketch.arcs[r];
        let dc = ((a.center.value.x - pos.x).powi(2) + (a.center.value.y - pos.y).powi(2)).sqrt();
        let dist_to_curve = (dc - a.radius.value).abs();
        if dist_to_curve <= radius || dc <= radius {
            found.push(format!("{} (d={:.2})", a.name, dist_to_curve.min(dc)));
        }
    }
    if found.is_empty() {
        ok(format!("Nothing found within {:.1} of ({:.2},{:.2})", radius, pos.x, pos.y))
    } else {
        ok(found.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Goto history position
// ---------------------------------------------------------------------------

fn cmd_goto(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let target_group: usize = match args.trim().parse() {
        Ok(v) => v,
        Err(_) => return err("Usage: goto <group_number> (see 'history')"),
    };
    let groups = ctx.history.group_list();
    let target_pos = if target_group == 0 {
        0
    } else if target_group <= groups.len() {
        groups[target_group - 1].1
    } else {
        return err(format!("Group {} does not exist (max {})", target_group, groups.len()));
    };
    if let Some(s) = ctx.history.goto(target_pos) {
        ctx.sketch = s;
    }
    ok(format!("Moved to group {} (position {})", target_group, ctx.history.cursor))
}

// ---------------------------------------------------------------------------
// Session variables
// ---------------------------------------------------------------------------

fn cmd_let(ctx: &mut CommandContext, args: &str) -> CommandResult {
    // let name = expr
    let args = args.trim();
    let (name, expr) = match args.split_once('=') {
        Some((n, e)) => (n.trim(), e.trim()),
        None => return err("Usage: let name = expression"),
    };
    if name.is_empty() { return err("Variable name cannot be empty"); }
    // Try as coordinate (geo function or endpoint ref)
    if let Some(result) = eval_geo_coord(&ctx.sketch, expr) {
        match result {
            Ok(v) => {
                ctx.session_vecs.insert(name.to_string(), v);
                ctx.session_vars.insert(format!("{}.x", name), v.x);
                ctx.session_vars.insert(format!("{}.y", name), v.y);
                return ok(format!("{} = ({:.4}, {:.4})", name, v.x, v.y));
            }
            Err(e) => return err(e),
        }
    }
    if let Ok(pos) = resolve_endpoint_pos(&ctx.sketch, expr) {
        ctx.session_vecs.insert(name.to_string(), pos);
        ctx.session_vars.insert(format!("{}.x", name), pos.x);
        ctx.session_vars.insert(format!("{}.y", name), pos.y);
        return ok(format!("{} = ({:.4}, {:.4})", name, pos.x, pos.y));
    }
    // Try as scalar
    if let Some(result) = eval_geo_scalar(&ctx.sketch, expr) {
        match result {
            Ok(v) => { ctx.session_vars.insert(name.to_string(), v); return ok(format!("{} = {:.6}", name, v)); }
            Err(e) => return err(e),
        }
    }
    match eval_expr_with(&ctx.sketch, expr, &ctx.session_vars) {
        Ok(v) => { ctx.session_vars.insert(name.to_string(), v); ok(format!("{} = {:.6}", name, v)) }
        Err(e) => err(format!("Eval error: {}", e)),
    }
}

// ---------------------------------------------------------------------------
// File operations
// ---------------------------------------------------------------------------

fn cmd_save(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let path = args.trim();
    if path.is_empty() { return err("Usage: save path.json"); }
    match serde_json::to_string_pretty(&ctx.sketch) {
        Ok(json) => match std::fs::write(path, &json) {
            Ok(_) => ok(format!("Saved to {}", path)),
            Err(e) => err(format!("Write error: {}", e)),
        },
        Err(e) => err(format!("Serialize error: {}", e)),
    }
}

fn cmd_load(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let path = args.trim();
    if path.is_empty() { return err("Usage: load path.json"); }
    match std::fs::read_to_string(path) {
        Ok(json) => match serde_json::from_str::<Sketch>(&json) {
            Ok(mut sketch) => {
                sketch.solve();
                ctx.history = crate::history::History::new(&sketch);
                ctx.sketch = sketch;
                ok(format!("Loaded {}", path))
            }
            Err(e) => err(format!("Parse error: {}", e)),
        },
        Err(e) => err(format!("Read error: {}", e)),
    }
}

// ---------------------------------------------------------------------------
// Cursor
// ---------------------------------------------------------------------------

fn cmd_cursor(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let args = args.trim();
    if args.is_empty() {
        // Show cursor position
        return match ctx.cursor {
            Some(p) => ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y)),
            None => ok("Cursor: off"),
        };
    }
    match args {
        "off" | "hide" => { ctx.cursor = None; return ok("Cursor hidden"); }
        "on" | "show" => {
            if ctx.cursor.is_none() { ctx.cursor = Some(vect2d::new(0.0, 0.0)); }
            let p = ctx.cursor.unwrap();
            return ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y));
        }
        _ => {}
    }
    // Set cursor to coordinate (absolute, relative, endpoint ref, etc.)
    match parse_coord(ctx, args, ctx.cursor) {
        Ok(p) => { ctx.cursor = Some(p); ok(format!("Cursor: ({:.4}, {:.4})", p.x, p.y)) }
        Err(e) => err(e),
    }
}

// ---------------------------------------------------------------------------
// Dimension text position
// ---------------------------------------------------------------------------

fn cmd_dim_pos(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.len() != 3 { return err("Usage: dim_pos d0 offset 1.5  or  dim_pos d0 along 0.3"); }
    let dim_name = tokens[0];
    let field = tokens[1];
    let val_str = tokens[2];
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == dim_name) {
        Some(i) => i,
        None => return err(format!("Unknown dimension: {}", dim_name)),
    };
    let is_relative = val_str.starts_with('@');
    let val_str = val_str.strip_prefix('@').unwrap_or(val_str);
    let val = match eval_expr(&ctx.sketch, val_str) { Ok(v) => v, Err(e) => return err(e) };
    match field {
        "offset" => {
            if is_relative {
                ctx.sketch.dimensions[idx].offset.y += val;
            } else {
                ctx.sketch.dimensions[idx].offset.y = val;
            }
            ok(format!("{} offset = {:.4}", dim_name, ctx.sketch.dimensions[idx].offset.y))
        }
        "along" => {
            if is_relative {
                ctx.sketch.dimensions[idx].text_along += val;
            } else {
                ctx.sketch.dimensions[idx].text_along = val;
            }
            ok(format!("{} along = {:.4}", dim_name, ctx.sketch.dimensions[idx].text_along))
        }
        _ => err(format!("Unknown field '{}'. Use: offset, along", field)),
    }
}

// ---------------------------------------------------------------------------
// Derived dimensions
// ---------------------------------------------------------------------------

fn cmd_set_derived(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let name = args.trim();
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        Some(i) => i,
        None => return err(format!("Unknown dimension: {}", name)),
    };
    if ctx.sketch.dimensions[idx].derived {
        return ok(format!("{} is already derived", name));
    }
    // Remove the underlying constraint
    ctx.begin_group();
    // Use RemoveDimension logic to remove constraint, then re-add as derived
    let dim = ctx.sketch.dimensions[idx].clone();
    ctx.exec(Action::RemoveDimension { index: idx });
    // Re-create as derived (no constraint)
    ctx.sketch.dimensions.push(Dimension {
        kind: dim.kind, value: dim.value, offset: dim.offset, text_along: dim.text_along,
        name: dim.name.clone(), expr_str: dim.expr_str, broken: dim.broken, derived: true,
    });
    ctx.sketch.solve();
    ctx.sketch.update_expr_dim_values();
    ok(format!("{} is now derived (reference only)", name))
}

fn cmd_set_driven(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let tokens: Vec<&str> = args.splitn(2, char::is_whitespace).collect();
    if tokens.is_empty() { return err("Usage: set_driven d0 [value]"); }
    let name = tokens[0].trim();
    let idx = match ctx.sketch.dimensions.iter().position(|d| d.name == name) {
        Some(i) => i,
        None => return err(format!("Unknown dimension: {}", name)),
    };
    if !ctx.sketch.dimensions[idx].derived {
        return ok(format!("{} is already driven (constraining)", name));
    }
    let (new_value, new_expr) = if tokens.len() > 1 {
        let val_str = tokens[1].trim().trim_matches('"');
        if let Ok(v) = val_str.parse::<f64>() {
            (v, None)
        } else {
            // Expression: evaluate for initial value, store as expr
            let v = match eval_expr(&ctx.sketch, val_str) { Ok(v) => v, Err(e) => return err(e) };
            (v, Some(val_str.to_string()))
        }
    } else {
        (ctx.sketch.dimensions[idx].value, ctx.sketch.dimensions[idx].expr_str.clone())
    };
    // Remove derived dim and re-add as driven
    let dim = ctx.sketch.dimensions[idx].clone();
    ctx.sketch.dimensions.remove(idx);
    ctx.begin_group();
    ctx.exec(Action::AddDimension {
        kind: dim.kind, value: new_value, expr: new_expr.clone(), derived: false,
    });
    // Restore visual properties
    if let Some(d) = ctx.sketch.dimensions.last_mut() {
        d.offset = dim.offset;
        d.text_along = dim.text_along;
        d.name = dim.name.clone();
    }
    ctx.sketch.solve();
    if let Some(expr) = new_expr {
        ok(format!("{} is now driven (constraining) = {}", name, expr))
    } else {
        ok(format!("{} is now driven (constraining) = {:.4}", name, new_value))
    }
}

// ---------------------------------------------------------------------------
// DOF analysis
// ---------------------------------------------------------------------------

/// Compute DOF count only. Delegates to compute_dof_sync and discards descriptions.
fn compute_dof_count(sketch: &mut Sketch) -> usize {
    compute_dof_sync(sketch).0
}

/// Compute DOF synchronously with free direction analysis.
fn compute_dof_sync(sketch: &mut Sketch) -> (usize, Vec<String>) {
    use arael::simple_lm::LmProblem;
    use arael_sketch_solver::SymbolBag;

    sketch.prepare_expr_constraints();

    let saved_drift = sketch.drift_isigma;
    sketch.drift_isigma = 0.0;

    let mut params = Vec::new();
    sketch.serialize64(&mut params);
    let n = params.len();
    if n == 0 {
        sketch.drift_isigma = saved_drift;
        return (0, Vec::new());
    }

    let bag = SymbolBag::build(sketch);
    let mut idx_to_name: Vec<String> = vec![String::new(); n];
    for (name, &idx) in &bag.param_indices {
        let i = idx as usize;
        if i < n && idx_to_name[i].is_empty() {
            idx_to_name[i] = name.clone();
        }
    }

    let mut grad = vec![0.0f64; n];
    let mut hessian = vec![0.0f64; n * n];
    sketch.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    sketch.drift_isigma = saved_drift;

    let h = nalgebra::DMatrix::from_row_slice(n, n, &hessian);
    let eigen = nalgebra::SymmetricEigen::new(h);
    let threshold = 1e-6;

    let mut free_dirs: Vec<String> = Vec::new();
    for col in 0..n {
        if eigen.eigenvalues[col].abs() > threshold { continue; }
        let ev = eigen.eigenvectors.column(col);

        let max_comp = ev.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        if max_comp < 1e-10 { continue; }
        let comp_threshold = max_comp * 0.1;

        let mut parts: Vec<(String, f64)> = Vec::new();
        for i in 0..n {
            if ev[i].abs() > comp_threshold {
                let name = if idx_to_name[i].is_empty() {
                    format!("param[{}]", i)
                } else {
                    idx_to_name[i].clone()
                };
                parts.push((name, ev[i]));
            }
        }
        if parts.is_empty() { continue; }

        free_dirs.push(classify_free_direction(&parts));
    }

    (free_dirs.len(), free_dirs)
}

fn cmd_dof(ctx: &mut CommandContext, args: &str) -> CommandResult {
    let arg = args.trim();
    if !arg.is_empty() && arg != "analyze" {
        return err("Usage: dof | dof analyze");
    }

    let (dof, free_dirs) = compute_dof_sync(&mut ctx.sketch);

    if arg == "analyze" {
        let mut lines = vec![format!("DOF: {}", dof)];
        for (i, desc) in free_dirs.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, desc));
        }
        ok(lines.join("\n"))
    } else {
        ok(format!("DOF: {}", dof))
    }
}

/// Classify a free direction from its eigenvector components.
fn classify_free_direction(parts: &[(String, f64)]) -> String {
    // Single parameter free
    if parts.len() == 1 {
        return format!("{} is free", parts[0].0);
    }

    // Collect entity names and check motion patterns
    let mut entities = std::collections::BTreeSet::new();
    let mut all_x = true;
    let mut all_y = true;
    let mut has_non_xy = false;

    for (name, _val) in parts {
        // Extract entity name (e.g., "L0" from "L0.p1.x")
        let entity = name.split('.').next().unwrap_or(name);
        entities.insert(entity.to_string());

        if name.ends_with(".x") {
            all_y = false;
        } else if name.ends_with(".y") {
            all_x = false;
        } else {
            // radius, angle, etc.
            all_x = false;
            all_y = false;
            has_non_xy = false;
        }
    }

    let entity_list: Vec<&str> = entities.iter().map(|s| s.as_str()).collect();
    let entity_str = if entity_list.len() <= 9 {
        entity_list.join(", ")
    } else {
        format!("{} entities", entity_list.len())
    };

    // Check for pure translation
    if all_x && !has_non_xy {
        return format!("translate X: {}", entity_str);
    }
    if all_y && !has_non_xy {
        return format!("translate Y: {}", entity_str);
    }

    // Check for uniform translation (all x components equal AND all y components equal)
    let x_vals: Vec<f64> = parts.iter()
        .filter(|(n, _)| n.ends_with(".x"))
        .map(|(_, v)| *v).collect();
    let y_vals: Vec<f64> = parts.iter()
        .filter(|(n, _)| n.ends_with(".y"))
        .map(|(_, v)| *v).collect();

    if !x_vals.is_empty() && !y_vals.is_empty() && y_vals.iter().all(|v| v.abs() < 1e-6) {
        // All Y near zero, only X moves
        return format!("translate X: {}", entity_str);
    }
    if !x_vals.is_empty() && !y_vals.is_empty() && x_vals.iter().all(|v| v.abs() < 1e-6) {
        return format!("translate Y: {}", entity_str);
    }

    // Check for rotation: x and y components should follow tangent pattern
    // For rotation around centroid, dx_i ~ -(y_i - cy), dy_i ~ (x_i - cx)
    // Simplified: if x and y components are mixed and entities share the motion, call it rotation
    if !x_vals.is_empty() && !y_vals.is_empty() && x_vals.len() == y_vals.len() && !has_non_xy {
        // Check if all translation components are equal (pure translation)
        let all_x_equal = x_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        let all_y_equal = y_vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        if all_x_equal && all_y_equal {
            return format!("translate: {}", entity_str);
        }
        return format!("rotate: {}", entity_str);
    }

    // Check for single-entity multi-param freedom
    if entities.len() == 1 {
        let param_list: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
        return format!("{} free: {}", entity_list[0], param_list.join(", "));
    }

    // Fallback: list participating entities
    format!("coupled motion: {}", entity_str)
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

fn cmd_help(args: &str) -> CommandResult {
    if args.trim() == "full" {
        return CommandResult {
            output: include_str!("../docs/COMMANDS.md").to_string(),
            is_error: false, no_echo: false, markdown: true,
        };
    }
    if args.is_empty() {
        ok("Commands: add_line add_point add_circle add_arc offset_line delete horizontal vertical \
            parallel perpendicular equal collinear tangent coincident concentric midpoint \
            symmetry point_on length radius angle distance remove_dim set_derived set_driven \
            lock unlock param del_param rename_param style select deselect print info list \
            find dof cost undo redo history goto center zoom cursor dim_pos clear let save load \
            remove_constraint(rc) help\n\
            Type 'help <command>' for details. 'help full' for complete reference.")
    } else {
        let msg = match args.trim() {
            "add_line" => "add_line x1,y1 x2,y2 | add_line @dx,dy (from last point) | add_line x,y (from last point)",
            "add_point" => "add_point x,y [nocursor]",
            "add_circle" => "add_circle cx,cy radius",
            "delete" => "delete L0 | delete P0 | delete A0",
            "horizontal" => "horizontal L0 [L1 ...]",
            "vertical" => "vertical L0 [L1 ...]",
            "parallel" => "parallel L0 L1",
            "perpendicular" => "perpendicular L0 L1",
            "equal" => "equal L0 L1 (length) | equal A0 A1 (radius)",
            "collinear" => "collinear L0 L1",
            "tangent" => "tangent L0 A0 | tangent A0 A1",
            "coincident" => "coincident L0.p2 L1.p1 (any endpoint pair: P0, L0.p1/p2, A0.center/start/end)",
            "concentric" => "concentric A0 A1",
            "midpoint" => "midpoint P0 L0 | midpoint L0.p1 L1",
            "symmetry" => "symmetry L0 L1 L2 (L0,L2 symmetric about L1)",
            "point_on" => "point_on P0 L0 | point_on L0.p1 A0",
            "length" => "length L0 5.0 [derived] | length L0 derived | length L0 \"width * 2\"",
            "radius" => "radius A0 1.5 [derived] | radius A0 derived | radius A0 \"expr\"",
            "sweep" => "sweep A0 180 [derived] | sweep A0 derived | sweep A0 \"expr\"",
            "angle" => "angle L0 L1 45 [derived] | angle L0 L1 derived",
            "distance" => "distance L0.p1 L1.p2 5.0 [derived] | distance P0 L0 3.0 [derived]",
            "remove_dim" => "remove_dim d0",
            "set_derived" => "set_derived d0 (make dimension display-only)",
            "set_driven" => "set_driven d0 [value|\"expr\"] (make dimension constraining)",
            "lock" => "lock P0 | lock L0.p1 | lock L0.p1 x,y",
            "unlock" => "unlock P0 | unlock L0.p1",
            "param" => "param name value | param name \"expr\" (creates or updates)",
            "del_param" => "del_param name",
            "rename_param" => "rename_param old_name new_name",
            "style" => "style L0 [solid|dashed|dashdot]",
            "select" => "select L0 [L1 P0 ...]",
            "deselect" => "deselect (clears selection)",
            "print" => "print <expression> (evaluate and display)",
            "info" => "info L0 | info P0 | info A0 | info d0 | info paramname",
            "list" => "list [all|lines|points|arcs|dims|params|constraints]",
            "find" => "find x,y [radius] (list nearby entities)",
            "undo" => "undo [n]",
            "redo" => "redo [n]",
            "history" => "history [n] (show last n entries)",
            "goto" => "goto <position> (jump to history position)",
            "center" => "center L0 | center x,y | center (fit all)",
            "zoom" => "zoom + | zoom - | zoom 2.0",
            "msg" => "msg text — print message to history (supports markdown, \\n for newlines)",
            "cursor" => "cursor [x,y | @dx,dy | on | off] — show/set/hide command cursor",
            "dim_pos" => "dim_pos d0 offset 1.5 | dim_pos d0 along 0.3 (@ for relative)",
            "clear" => "clear (new empty sketch)",
            "add_arc" => "add_arc x1,y1 x2,y2 xm,ym (start, end, midpoint)",
            "offset_line" | "offset" => "offset_line L0 distance (create parallel line offset by distance)",
            "let" => "let name = expression (session variable, scalar or coordinate)",
            "save" => "save path.json",
            "load" => "load path.json",
            "perp" => "alias for perpendicular",
            other => return err(format!("help: unknown command: {}. Usage: help | help <command> | help full", other)),
        };
        ok(msg.to_string())
    }
}

// ---------------------------------------------------------------------------
// Autocomplete
// ---------------------------------------------------------------------------

const COMMAND_NAMES: &[&str] = &[
    "add_line", "add_point", "add_circle", "add_arc", "offset_line", "offset",
    "delete", "horizontal", "vertical", "parallel", "perpendicular", "perp",
    "equal", "collinear", "tangent", "coincident", "concentric", "midpoint",
    "symmetry", "point_on", "length", "radius", "sweep", "angle", "distance",
    "remove_dim", "remove_constraint", "rc", "set_derived", "set_driven",
    "lock", "unlock", "param", "del_param", "rename_param", "style",
    "select", "deselect", "print", "info", "list", "find", "let",
    "dof", "cost", "undo", "redo", "history", "goto", "center", "zoom",
    "cursor", "dim_pos", "clear", "save", "load", "help", "msg",
];

const GEO_FUNCTIONS: &[&str] = &[
    "intersect", "midpoint", "project", "along", "arc_point",
    "rotate", "mirror", "tangent", "normal", "dist", "angle",
];

const MATH_FUNCTIONS: &[&str] = &[
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "ln", "log2", "log10",
    "sqrt", "abs", "pow", "pi",
];

/// Generate autocomplete suggestions for the command input.
/// Returns completions for the word at `cursor_pos` in `input`.
pub fn complete(
    sketch: &Sketch,
    session_names: &HashMap<String, String>,
    input: &str,
    cursor_pos: usize,
) -> Vec<String> {
    let input = &input[..cursor_pos.min(input.len())];
    let current_line = input.lines().last().unwrap_or("");
    let word_start = current_line.rfind(|c: char| c.is_whitespace()).map(|i| i + 1).unwrap_or(0);
    let current_word = &current_line[word_start..];
    let is_first_token = current_line[..word_start].trim().is_empty();

    // No completions when nothing typed on first token (would show all commands)
    if current_word.is_empty() && is_first_token { return Vec::new(); }

    // Dot completion (context-independent)
    if let Some(dot_pos) = current_word.rfind('.') {
        let before_dot = &current_word[..dot_pos];
        let after_dot = &current_word[dot_pos + 1..];
        let mut r = complete_after_dot(sketch, before_dot, after_dot);
        r.retain(|s| s != current_word);
        return r;
    }

    let mut results = Vec::new();

    // First token: command names only
    if is_first_token {
        add_matching(&mut results, current_word, COMMAND_NAMES);
        results.sort();
        results.dedup();
        results.truncate(20);
        return results;
    }

    // Non-first token: command-specific completions
    let first_cmd = current_line.split_whitespace().next().unwrap_or("");
    let token_index = current_line[..word_start].split_whitespace().count();
    // token_index: 1 = arg1, 2 = arg2, 3 = arg3

    // Collect already-completed args (excluding current word being typed)
    let typed_args: Vec<&str> = current_line[..word_start].split_whitespace().skip(1).collect();

    match first_cmd {
        // Variadic line commands: exclude already-typed lines
        "horizontal" | "vertical" => {
            add_lines_excluding(sketch, &mut results, current_word, &typed_args);
        }

        // Two-line commands: no suggestions after 2 args
        "parallel" | "perpendicular" | "perp" | "collinear" => {
            if token_index <= 2 {
                add_lines(sketch, &mut results, current_word);
            }
        }

        // Arc-only, exactly 2 args
        "concentric" => {
            if token_index <= 2 {
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Equal: match type of first arg, exactly 2 args
        "equal" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                let arg1 = current_line.split_whitespace().nth(1).unwrap_or("");
                if arg1.starts_with('L') {
                    add_lines(sketch, &mut results, current_word);
                } else if arg1.starts_with('A') {
                    add_arcs(sketch, &mut results, current_word);
                }
            }
        }

        // Tangent: line+arc or arc+arc, exactly 2 args
        "tangent" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Variadic entity commands: exclude already-typed
        "delete" | "select" => {
            add_all_entities_excluding(sketch, &mut results, current_word, &typed_args);
        }

        // Single entity arg
        "info" | "center" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }

        // Endpoint commands: exactly 2 args
        "coincident" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }
        "lock" | "unlock" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            }
        }

        // Midpoint: arg1=point/endpoint, arg2=line
        "midpoint" => {
            if token_index == 1 {
                add_points(sketch, &mut results, current_word);
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_lines(sketch, &mut results, current_word);
            }
        }

        // Point_on: arg1=point/endpoint, arg2=line or arc
        "point_on" => {
            if token_index == 1 {
                add_points(sketch, &mut results, current_word);
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            }
        }

        // Symmetry: arg1=entity, arg2=line(mirror), arg3=entity
        "symmetry" => {
            if token_index <= 3 {
                if token_index == 2 {
                    add_lines(sketch, &mut results, current_word);
                } else {
                    add_all_entities(sketch, &mut results, current_word);
                }
            }
        }

        // Dimension: length (arg1=line, arg2=value/derived)
        "length" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: radius (arg1=arc, arg2=value/derived)
        "radius" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: sweep (arg1=arc, arg2=value/derived)
        "sweep" => {
            if token_index == 1 {
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["derived"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: angle (arg1=line, arg2=line, arg3=value/derived)
        "angle" => {
            if token_index <= 2 {
                add_lines(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension: distance (arg1=endpoint, arg2=endpoint/line, arg3=value/derived)
        "distance" => {
            if token_index <= 2 {
                add_all_entities(sketch, &mut results, current_word);
            } else if token_index == 3 {
                add_matching(&mut results, current_word, &["derived"]);
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Dimension management: single dim arg
        "remove_dim" | "set_derived" | "set_driven" => {
            if token_index == 1 {
                add_dimensions(sketch, &mut results, current_word);
            }
        }

        // dim_pos: arg1=dim, arg2=offset/along, arg3=value
        "dim_pos" => {
            if token_index == 1 {
                add_dimensions(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["offset", "along"]);
            }
        }

        // Style: arg1=entity, arg2=style value
        "style" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
                add_arcs(sketch, &mut results, current_word);
            } else if token_index == 2 {
                add_matching(&mut results, current_word, &["solid", "dashed", "dashdot"]);
            }
        }

        // List: single filter arg
        "list" => {
            if token_index == 1 {
                add_matching(&mut results, current_word,
                    &["all", "lines", "points", "arcs", "dims", "params", "constraints"]);
            }
        }

        // Help: single arg (full or command name)
        "help" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["full"]);
                add_matching(&mut results, current_word, COMMAND_NAMES);
            }
        }

        // Cursor: single keyword arg
        "cursor" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["on", "off", "show", "hide"]);
            }
        }

        // Remove constraint
        "remove_constraint" | "rc" => {
            if token_index == 1 {
                add_all_entities(sketch, &mut results, current_word);
            } else if token_index <= 3 {
                add_all_entities(sketch, &mut results, current_word);
                add_matching(&mut results, current_word,
                    &["horizontal", "vertical", "parallel", "perpendicular",
                      "equal", "equal_length", "equal_radius", "collinear",
                      "tangent", "concentric", "coincident", "point_on",
                      "symmetry", "midpoint", "lock"]);
            }
        }

        // Param commands: single param name
        "param" | "del_param" | "rename_param" => {
            if token_index == 1 {
                add_params(sketch, &mut results, current_word);
            }
        }

        // Offset: arg1=line, arg2=expression
        "offset_line" | "offset" => {
            if token_index == 1 {
                add_lines(sketch, &mut results, current_word);
            } else {
                add_expression_completions(sketch, session_names, &mut results, current_word);
            }
        }

        // Geometry creation: position-aware completions
        // add_line: [coord1] [coord2] [noconnect] [nocursor]
        // add_point: [coord] (no flags)
        // add_circle: [center] [radius] [noconnect] [nocursor]
        // add_arc: [start] [end] [mid] [noconnect] [nocursor]
        "add_line" => {
            let max_coords = if typed_args.iter().any(|a| *a == "noconnect" || *a == "nocursor") { 0 } else { 2 };
            let coord_args = typed_args.iter().filter(|a| **a != "noconnect" && **a != "nocursor").count();
            if coord_args < max_coords {
                // Still entering coordinates
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            // Offer flags not already typed (after at least 1 coord)
            if coord_args >= 1 {
                if !typed_args.contains(&"noconnect") {
                    add_matching(&mut results, current_word, &["noconnect"]);
                }
                if !typed_args.contains(&"nocursor") {
                    add_matching(&mut results, current_word, &["nocursor"]);
                }
            }
        }
        "add_point" => {
            let coord_args = typed_args.iter().filter(|a| **a != "nocursor").count();
            if coord_args < 1 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 1 && !typed_args.contains(&"nocursor") {
                add_matching(&mut results, current_word, &["nocursor"]);
            }
        }
        "add_circle" => {
            let coord_args = typed_args.iter().filter(|a| **a != "noconnect" && **a != "nocursor").count();
            if coord_args < 2 {
                // arg1=center coord, arg2=radius (expression)
                if coord_args == 0 {
                    add_matching(&mut results, current_word, &["cursor"]);
                    add_all_entities(sketch, &mut results, current_word);
                    add_session_names(session_names, &mut results, current_word);
                } else {
                    // radius: expression context
                    add_expression_completions(sketch, session_names, &mut results, current_word);
                }
            }
            if coord_args >= 2 {
                if !typed_args.contains(&"noconnect") {
                    add_matching(&mut results, current_word, &["noconnect"]);
                }
                if !typed_args.contains(&"nocursor") {
                    add_matching(&mut results, current_word, &["nocursor"]);
                }
            }
        }
        "add_arc" => {
            let coord_args = typed_args.iter().filter(|a| **a != "noconnect" && **a != "nocursor").count();
            if coord_args < 3 {
                add_matching(&mut results, current_word, &["cursor"]);
                add_all_entities(sketch, &mut results, current_word);
                add_session_names(session_names, &mut results, current_word);
            }
            if coord_args >= 3 {
                if !typed_args.contains(&"noconnect") {
                    add_matching(&mut results, current_word, &["noconnect"]);
                }
                if !typed_args.contains(&"nocursor") {
                    add_matching(&mut results, current_word, &["nocursor"]);
                }
            }
        }

        // Expression-only: print, let
        "print" => {
            add_expression_completions(sketch, session_names, &mut results, current_word);
            add_all_entities(sketch, &mut results, current_word);
        }
        "let" => {
            add_expression_completions(sketch, session_names, &mut results, current_word);
            add_all_entities(sketch, &mut results, current_word);
        }

        // No completions for these
        // dof: single arg
        "dof" => {
            if token_index == 1 {
                add_matching(&mut results, current_word, &["analyze"]);
            }
        }

        "undo" | "redo" | "history" | "goto" | "cost" | "clear"
        | "deselect" | "save" | "load" | "msg" | "find" | "zoom" => {}

        _ => {}
    }

    results.sort();
    results.dedup();
    results.truncate(20);
    results
}

// --- Completion helpers ---

fn add_matching(results: &mut Vec<String>, prefix: &str, candidates: &[&str]) {
    for &c in candidates {
        if c.starts_with(prefix) && c != prefix {
            results.push(c.to_string());
        }
    }
}

fn add_lines(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

fn add_arcs(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.arcs.refs() {
        let name = &sketch.arcs[r].name;
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

fn add_points(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        if p.helper { continue; }
        if p.name.starts_with(prefix) && p.name != prefix {
            results.push(p.name.clone());
        }
    }
}

fn add_all_entities(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    add_lines(sketch, results, prefix);
    add_points(sketch, results, prefix);
    add_arcs(sketch, results, prefix);
}

fn add_lines_excluding(sketch: &Sketch, results: &mut Vec<String>, prefix: &str, exclude: &[&str]) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
}

fn add_all_entities_excluding(sketch: &Sketch, results: &mut Vec<String>, prefix: &str, exclude: &[&str]) {
    for r in sketch.lines.refs() {
        let name = &sketch.lines[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
    for r in sketch.points.refs() {
        let p = &sketch.points[r];
        if p.helper { continue; }
        if p.name.starts_with(prefix) && p.name != prefix && !exclude.contains(&p.name.as_str()) {
            results.push(p.name.clone());
        }
    }
    for r in sketch.arcs.refs() {
        let name = &sketch.arcs[r].name;
        if name.starts_with(prefix) && name != prefix && !exclude.contains(&name.as_str()) {
            results.push(name.clone());
        }
    }
}

fn add_dimensions(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for d in &sketch.dimensions {
        if d.name.starts_with(prefix) && d.name != prefix {
            results.push(d.name.clone());
        }
    }
}

fn add_params(sketch: &Sketch, results: &mut Vec<String>, prefix: &str) {
    for p in &sketch.user_params {
        if p.name.starts_with(prefix) && p.name != prefix {
            results.push(p.name.clone());
        }
    }
}

fn add_session_names(session_names: &HashMap<String, String>, results: &mut Vec<String>, prefix: &str) {
    for (name, _) in session_names {
        if name == "_" { continue; }
        if name.starts_with(prefix) && name != prefix {
            results.push(name.clone());
        }
    }
}

fn add_expression_completions(sketch: &Sketch, session_names: &HashMap<String, String>, results: &mut Vec<String>, prefix: &str) {
    add_dimensions(sketch, results, prefix);
    add_params(sketch, results, prefix);
    add_session_names(session_names, results, prefix);
    add_matching(results, prefix, GEO_FUNCTIONS);
    add_matching(results, prefix, MATH_FUNCTIONS);
}

/// Complete after a dot: "L0." → ["p1", "p2"], "A0." → ["center", "start", "end"], etc.
fn complete_after_dot(sketch: &Sketch, before_dot: &str, after_dot: &str) -> Vec<String> {
    let mut results = Vec::new();

    // Check for double-dot: "L0.p1." → x, y
    if let Some(first_dot) = before_dot.rfind('.') {
        let entity = &before_dot[..first_dot];
        let prop = &before_dot[first_dot + 1..];
        // L<N>.p1. or L<N>.p2. → x, y
        if entity.starts_with('L') && (prop == "p1" || prop == "p2") {
            for &s in &["x", "y"] {
                if s.starts_with(after_dot) {
                    results.push(format!("{}.{}", before_dot, s));
                }
            }
        }
        // A<N>.center. → x, y
        if entity.starts_with('A') && prop == "center" {
            for &s in &["x", "y"] {
                if s.starts_with(after_dot) {
                    results.push(format!("{}.{}", before_dot, s));
                }
            }
        }
        // P<N>. after P<N>.pos or similar
        return results;
    }

    // Single dot: entity.suffix
    if before_dot.starts_with('L') && sketch.lines.refs().any(|r| sketch.lines[r].name == before_dot) {
        for &s in &["p1", "p2", "length", "angle"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    } else if before_dot.starts_with('A') && sketch.arcs.refs().any(|r| sketch.arcs[r].name == before_dot) {
        for &s in &["center", "start", "end", "radius", "start_angle", "end_angle"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    } else if before_dot.starts_with('P') && sketch.points.refs().any(|r| sketch.points[r].name == before_dot) {
        for &s in &["x", "y"] {
            if s.starts_with(after_dot) {
                results.push(format!("{}.{}", before_dot, s));
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run(ctx: &mut CommandContext, cmd: &str) -> CommandResult {
        let results = execute(ctx, cmd);
        assert!(!results.is_empty());
        results.into_iter().next().unwrap()
    }

    fn run_ok(ctx: &mut CommandContext, cmd: &str) -> String {
        let r = run(ctx, cmd);
        assert!(!r.is_error, "Command '{}' failed: {}", cmd, r.output);
        r.output
    }

    fn run_err(ctx: &mut CommandContext, cmd: &str) -> String {
        let r = run(ctx, cmd);
        assert!(r.is_error, "Command '{}' should have failed but got: {}", cmd, r.output);
        r.output
    }

    fn line_len(ctx: &CommandContext, name: &str) -> f64 {
        let r = resolve_line(&ctx.sketch, name).unwrap();
        let l = &ctx.sketch.lines[r];
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        (dx * dx + dy * dy).sqrt()
    }

    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 0.1 }

    // -- Geometry creation --

    #[test]
    fn test_add_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.sketch.lines.refs().count(), 1);
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 0.0));
        assert!(near(ctx.sketch.lines[r].p2.value.x, 5.0));
    }

    #[test]
    fn test_add_line_chaining() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line @0,3");
        let r = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
        assert!(near(ctx.sketch.lines[r].p2.value.y, 3.0));
    }

    #[test]
    fn test_add_line_single_arg() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line 5,3");
        let r = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
        assert!(near(ctx.sketch.lines[r].p2.value.y, 3.0));
    }

    #[test]
    fn test_add_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 3,4");
        assert!(ctx.sketch.points.refs().count() >= 1);
    }

    #[test]
    fn test_add_circle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2");
        assert_eq!(ctx.sketch.arcs.refs().count(), 1);
    }

    // -- Coordinate parsing --

    #[test]
    fn test_coord_endpoint_ref() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "add_line L0.p2 10,0");
        let r = resolve_line(&ctx.sketch, "L1").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
    }

    #[test]
    fn test_coord_relative() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 @3,4");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p2.value.x, 3.0));
        assert!(near(ctx.sketch.lines[r].p2.value.y, 4.0));
    }

    #[test]
    fn test_coord_expression() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param w 5");
        run_ok(&mut ctx, "add_line 0,0 w,0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p2.value.x, 5.0));
    }

    #[test]
    fn test_coord_geo_function() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        run_ok(&mut ctx, "add_point midpoint(L0)");
        let p = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[p].pos.value.x, 2.0));
    }

    // -- Constraints --

    #[test]
    fn test_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1");
        run_ok(&mut ctx, "horizontal L0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_vertical() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 1,5");
        run_ok(&mut ctx, "vertical L0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.vertical);
    }

    #[test]
    fn test_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "parallel L0 L1");
        assert!(!ctx.sketch.parallel.is_empty());
    }

    #[test]
    fn test_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,0.1 10,0");
        run_ok(&mut ctx, "coincident L0.p2 L1.p1");
        assert!(!ctx.sketch.coincident_ll21.is_empty());
    }

    // -- Dimensions --

    #[test]
    fn test_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        // Solve happened inside exec
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    #[test]
    fn test_remove_dim() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        run_ok(&mut ctx, "remove_dim d0");
        assert_eq!(ctx.sketch.dimensions.len(), 0);
    }

    // -- Parameters --

    #[test]
    fn test_param_create() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param width 10");
        assert_eq!(ctx.sketch.user_params.len(), 1);
        assert_eq!(ctx.sketch.user_params[0].name, "width");
    }

    #[test]
    fn test_param_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param w 5");
        run_ok(&mut ctx, "param w 10");
        assert_eq!(ctx.sketch.user_params[0].value, 10.0);
    }

    #[test]
    fn test_del_param() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param w 5; del_param w");
        assert!(ctx.sketch.user_params.is_empty());
    }

    #[test]
    fn test_rename_param() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param width 5; rename_param width w");
        assert_eq!(ctx.sketch.user_params[0].name, "w");
    }

    // -- Style --

    #[test]
    fn test_style_set() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "style L0 dashed");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert_eq!(ctx.sketch.lines[r].style, LineStyle::Dashed);
    }

    #[test]
    fn test_style_query() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "style L0");
        assert!(out.contains("solid"));
    }

    // -- Introspection --

    #[test]
    fn test_print_expr() {
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "print 2+3");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_print_entity() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        let out = run_ok(&mut ctx, "print L0.length");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_info_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "info L0");
        assert!(out.contains("L0"));
    }

    #[test]
    fn test_list() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_point 3,4");
        let out = run_ok(&mut ctx, "list");
        assert!(out.contains("L0"));
        assert!(out.contains("P0"));
    }

    #[test]
    fn test_list_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
        let out = run_ok(&mut ctx, "list constraints");
        assert!(out.contains("horizontal"));
    }

    #[test]
    fn test_find() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 10,0");
        let out = run_ok(&mut ctx, "find 5,0 1");
        assert!(out.contains("L0"));
    }

    // -- Geometric functions --

    #[test]
    fn test_intersect() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line -1,-1 1,1; add_line -1,1 1,-1");
        run_ok(&mut ctx, "add_point intersect(L0,L1)");
        let p = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[p].pos.value.x, 0.0));
        assert!(near(ctx.sketch.points[p].pos.value.y, 0.0));
    }

    #[test]
    fn test_midpoint_func() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0; add_point midpoint(L0)");
        let p = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[p].pos.value.x, 2.0));
    }

    #[test]
    fn test_tangent_normal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 4,0");
        let out = run_ok(&mut ctx, "print tangent(L0)");
        assert!(out.contains("1.0"));
        let out = run_ok(&mut ctx, "print normal(L0)");
        assert!(out.contains("1.0")); // normal is (0,1), output has "1.0"
    }

    #[test]
    fn test_dist_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        let out = run_ok(&mut ctx, "print dist(L0.p1,L0.p2)");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_dist_pl() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,3; add_line 0,0 10,0");
        let out = run_ok(&mut ctx, "print dist(P0,L0)");
        assert!(out.contains("3"));
    }

    // -- Session variables --

    #[test]
    fn test_let_scalar() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "let d = 5");
        let out = run_ok(&mut ctx, "print d");
        assert!(out.contains("5"));
    }

    #[test]
    fn test_let_vec() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        run_ok(&mut ctx, "let p = L0.p2");
        let out = run_ok(&mut ctx, "print p");
        assert!(out.contains("3") && out.contains("4"));
    }

    #[test]
    fn test_let_in_coord() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4; let p = L0.p2");
        run_ok(&mut ctx, "add_point p");
        let pt = ctx.sketch.points.refs().last().unwrap();
        assert!(near(ctx.sketch.points[pt].pos.value.x, 3.0));
    }

    // -- Selection --

    #[test]
    fn test_select() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; select L0");
        assert_eq!(ctx.selection.len(), 1);
    }

    #[test]
    fn test_deselect() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; select L0; deselect");
        assert!(ctx.selection.is_empty());
    }

    // -- Undo/redo --

    #[test]
    fn test_undo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.sketch.lines.refs().count(), 1);
        run_ok(&mut ctx, "undo");
        assert_eq!(ctx.sketch.lines.refs().count(), 0);
    }

    #[test]
    fn test_redo() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; undo; redo");
        assert_eq!(ctx.sketch.lines.refs().count(), 1);
    }

    #[test]
    fn test_history() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 1,1 2,2");
        let out = run_ok(&mut ctx, "history");
        assert!(out.contains("Add line"));
    }

    // -- Remove constraint --

    #[test]
    fn test_remove_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
        run_ok(&mut ctx, "rc L0 horizontal");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(!ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_remove_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1; parallel L0 L1");
        assert!(!ctx.sketch.parallel.is_empty());
        run_ok(&mut ctx, "rc L0 L1 parallel");
        assert!(ctx.sketch.parallel.is_empty());
    }

    // -- Multi-command --

    #[test]
    fn test_semicolon() {
        let mut ctx = CommandContext::new();
        let results = execute(&mut ctx, "add_line 0,0 5,0; horizontal L0");
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_error);
        assert!(!results[1].is_error);
    }

    // -- Error handling --

    #[test]
    fn test_unknown_command() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "foobar"); }

    #[test]
    fn test_unknown_entity() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "info L99"); }

    #[test]
    fn test_bad_coord() { let mut ctx = CommandContext::new(); run_err(&mut ctx, "add_line abc xyz"); }

    #[test]
    fn test_help() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "help"); }

    #[test]
    fn test_help_command() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "help add_line"); }

    #[test]
    fn test_dof() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "dof"); }

    #[test]
    fn test_cost() { let mut ctx = CommandContext::new(); run_ok(&mut ctx, "cost"); }

    // -- Entity name capture --

    #[test]
    fn test_auto_underscore() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1");
        assert!(ctx.session_names.contains_key("_"));
        run_ok(&mut ctx, "vertical _");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.vertical);
    }

    #[test]
    fn test_auto_underscore_updates() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert_eq!(ctx.session_names["_"], "L0");
        run_ok(&mut ctx, "add_line 1,1 2,2");
        assert_eq!(ctx.session_names["_"], "L1");
    }

    #[test]
    fn test_assign_entity_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "base = add_line 0,0 5,1");
        assert_eq!(ctx.session_names.get("base").unwrap(), "L0");
        run_ok(&mut ctx, "horizontal base");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_let_entity_name() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "let l = add_line 0,0 5,1");
        assert_eq!(ctx.session_names.get("l").unwrap(), "L0");
        run_ok(&mut ctx, "horizontal l");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
    }

    #[test]
    fn test_let_entity_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "a = add_line 0,0 5,0");
        run_ok(&mut ctx, "b = add_line 5,0.1 10,0");
        run_ok(&mut ctx, "coincident a.p2 b.p1");
        assert!(!ctx.sketch.coincident_ll21.is_empty());
    }

    #[test]
    fn test_let_entity_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "l = add_line 0,0 5,0");
        run_ok(&mut ctx, "length l 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    #[test]
    fn test_let_entity_info() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "l = add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "info l");
        assert!(out.contains("L0"));
    }

    #[test]
    fn test_underscore_chain() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; horizontal _; length _ 3");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(ctx.sketch.lines[r].constraints.horizontal);
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    // -- Auto-coincident --

    #[test]
    fn test_auto_coincident() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_line 5,0 5,3");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        // L1.p1==L0.p2 -> coincident_ll12 (a.p1 == b.p2 where a=L1, b=L0)
        let has_coincident = !ctx.sketch.coincident_ll12.is_empty()
            || !ctx.sketch.coincident_ll21.is_empty()
            || !ctx.sketch.coincident_ll11.is_empty()
            || !ctx.sketch.coincident_ll22.is_empty();
        assert!(has_coincident, "Should have coincident constraint");
    }

    #[test]
    fn test_noconnect() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_line 5,0 5,3 noconnect");
        assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
        assert!(ctx.sketch.coincident_ll21.is_empty());
    }

    // -- Auto-coincident for arcs/circles --

    #[test]
    fn test_auto_coincident_circle_to_line_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_circle 5,0 1");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(out.contains("A0.center=L0.p2"), "Should mention A0.center=L0.p2: {}", out);
        assert!(!ctx.sketch.coincident_lp2_arc_center.is_empty(),
            "Should have coincident_lp2_arc_center");
    }

    #[test]
    fn test_auto_coincident_circle_to_point() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 3,3");
        let out = run_ok(&mut ctx, "add_circle 3,3 1");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(!ctx.sketch.coincident_arc_center.is_empty(),
            "Should have coincident_arc_center");
    }

    #[test]
    fn test_auto_coincident_circle_concentric() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2");
        let out = run_ok(&mut ctx, "add_circle 0,0 3");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(out.contains("A1.center=A0.center"), "Should mention concentric: {}", out);
        assert!(!ctx.sketch.concentric.is_empty(), "Should have concentric constraint");
    }

    #[test]
    fn test_auto_coincident_line_to_arc_center() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 5,3 1");
        let out = run_ok(&mut ctx, "add_line 0,0 5,3");
        assert!(out.contains("connected"), "Should auto-connect: {}", out);
        assert!(out.contains("L0.p2=A0.center"), "Should mention A0.center: {}", out);
        assert!(!ctx.sketch.coincident_lp2_arc_center.is_empty(),
            "Should have coincident_lp2_arc_center");
    }

    #[test]
    fn test_noconnect_circle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_circle 5,0 1 noconnect");
        assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
        assert!(ctx.sketch.coincident_lp2_arc_center.is_empty());
    }

    #[test]
    fn test_noconnect_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "add_arc 5,0 5,3 6,1.5 noconnect");
        assert!(!out.contains("connected"), "Should NOT auto-connect: {}", out);
    }

    // -- Duplicate constraint rejection --

    #[test]
    fn test_duplicate_horizontal() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        let e = run_err(&mut ctx, "horizontal L0");
        assert!(e.contains("already horizontal"));
    }

    #[test]
    fn test_duplicate_vertical() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 0,5");
        run_ok(&mut ctx, "vertical L0");
        let e = run_err(&mut ctx, "vertical L0");
        assert!(e.contains("already vertical"));
    }

    #[test]
    fn test_duplicate_parallel() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "parallel L0 L1");
        let e = run_err(&mut ctx, "parallel L0 L1");
        assert!(e.contains("already exists"));
        let e = run_err(&mut ctx, "parallel L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_perpendicular() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
        run_ok(&mut ctx, "perpendicular L0 L1");
        let e = run_err(&mut ctx, "perpendicular L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_equal_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "equal L0 L1");
        let e = run_err(&mut ctx, "equal L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_equal_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 3");
        run_ok(&mut ctx, "equal A0 A1");
        let e = run_err(&mut ctx, "equal A1 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_collinear() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 6,0 10,0");
        run_ok(&mut ctx, "collinear L0 L1");
        let e = run_err(&mut ctx, "collinear L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_tangent_la() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 2.5,1 1");
        run_ok(&mut ctx, "tangent L0 A0");
        let e = run_err(&mut ctx, "tangent L0 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_tangent_aa() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 2");
        run_ok(&mut ctx, "tangent A0 A1");
        let e = run_err(&mut ctx, "tangent A1 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_coincident_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
        run_ok(&mut ctx, "coincident L0.p2 L1.p1");
        let e = run_err(&mut ctx, "coincident L0.p2 L1.p1");
        assert!(e.contains("already exists"));
        // Cross-type: L0.p2=L1.p1 is same as L1.p1=L0.p2 (swapped order)
        let e = run_err(&mut ctx, "coincident L1.p1 L0.p2");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_concentric() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 2; add_circle 5,0 3");
        run_ok(&mut ctx, "concentric A0 A1");
        let e = run_err(&mut ctx, "concentric A1 A0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_point_on_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2.5,0; add_line 0,0 5,0");
        run_ok(&mut ctx, "point_on P0 L0");
        let e = run_err(&mut ctx, "point_on P0 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_midpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2.5,0; add_line 0,0 5,0");
        run_ok(&mut ctx, "midpoint P0 L0");
        let e = run_err(&mut ctx, "midpoint P0 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_duplicate_symmetry_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line -2,0 -2,3; add_line 0,0 0,5; add_line 2,0 2,3");
        run_ok(&mut ctx, "symmetry L0 L1 L2");
        let e = run_err(&mut ctx, "symmetry L2 L1 L0");
        assert!(e.contains("already exists"));
    }

    #[test]
    fn test_self_reference_rejected() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 0,0 2");
        let e = run_err(&mut ctx, "parallel L0 L0");
        assert!(e.contains("itself"));
        let e = run_err(&mut ctx, "equal L0 L0");
        assert!(e.contains("itself"));
        let e = run_err(&mut ctx, "concentric A0 A0");
        assert!(e.contains("itself"));
    }

    // -- Info with constraints --

    #[test]
    fn test_info_shows_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,1; horizontal L0");
        let out = run_ok(&mut ctx, "info L0");
        assert!(out.contains("horizontal"), "info should show constraints: {}", out);
    }

    #[test]
    fn test_info_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "info L0.p1");
        assert!(out.contains("0.0000"), "info L0.p1 should show position: {}", out);
    }

    // -- Select endpoints --

    #[test]
    fn test_select_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "select L0.p1");
        assert_eq!(ctx.selection.len(), 1);
        assert!(matches!(ctx.selection[0], Selection::LineP1(_)));
    }

    // -- Param shows value --

    #[test]
    fn test_param_shows_value() {
        let mut ctx = CommandContext::new();
        let out = run_ok(&mut ctx, "param kala 12+3*4");
        assert!(out.contains("24"), "Should show evaluated value: {}", out);
    }

    // -- Cursor --

    #[test]
    fn test_cursor_set() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 5,3");
        assert!(ctx.cursor.is_some());
        assert!(near(ctx.cursor.unwrap().x, 5.0));
        assert!(near(ctx.cursor.unwrap().y, 3.0));
    }

    #[test]
    fn test_cursor_from_add_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        assert!(ctx.cursor.is_some());
        assert!(near(ctx.cursor.unwrap().x, 5.0));
        assert!(near(ctx.cursor.unwrap().y, 0.0));
    }

    #[test]
    fn test_cursor_relative() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 1,1");
        run_ok(&mut ctx, "cursor @2,3");
        assert!(near(ctx.cursor.unwrap().x, 3.0));
        assert!(near(ctx.cursor.unwrap().y, 4.0));
    }

    #[test]
    fn test_cursor_as_coord() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 5,0");
        run_ok(&mut ctx, "add_line cursor 5,3");
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(near(ctx.sketch.lines[r].p1.value.x, 5.0));
        assert!(near(ctx.sketch.lines[r].p1.value.y, 0.0));
    }

    #[test]
    fn test_cursor_off() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 5,3");
        assert!(ctx.cursor.is_some());
        run_ok(&mut ctx, "cursor off");
        assert!(ctx.cursor.is_none());
    }

    #[test]
    fn test_cursor_nocursor() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 1,1");
        run_ok(&mut ctx, "add_line 0,0 5,0 nocursor");
        // Cursor should still be at 1,1, not moved to 5,0
        assert!(near(ctx.cursor.unwrap().x, 1.0));
    }

    #[test]
    fn test_cursor_endpoint_ref() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "cursor L0.p1");
        assert!(near(ctx.cursor.unwrap().x, 0.0));
    }

    #[test]
    fn test_cursor_query() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "cursor 3,7");
        let out = run_ok(&mut ctx, "cursor");
        assert!(out.contains("3.0000") && out.contains("7.0000"));
    }

    // -- Dimension text position --

    #[test]
    fn test_dim_pos_offset() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        run_ok(&mut ctx, "dim_pos d0 offset 2.0");
        assert!(near(ctx.sketch.dimensions[0].offset.y, 2.0));
    }

    #[test]
    fn test_dim_pos_along() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        run_ok(&mut ctx, "dim_pos d0 along 0.5");
        assert!(near(ctx.sketch.dimensions[0].text_along, 0.5));
    }

    #[test]
    fn test_dim_info_shows_position() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        let out = run_ok(&mut ctx, "info d0");
        assert!(out.contains("offset=") && out.contains("along="));
    }

    // -- Point symmetry command --

    #[test]
    fn test_cmd_symmetry_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 3,2; add_point 7,2; add_line 5,0 5,10");
        run_ok(&mut ctx, "symmetry P0 L0 P1");
        assert!(!ctx.sketch.symmetry_pp.is_empty());
    }

    // -- Derived dimensions --

    #[test]
    fn test_cmd_derived_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 5 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        // Derived should NOT constrain — line length should stay at original ~5
        // (no has_length constraint set)
        let r = resolve_line(&ctx.sketch, "L0").unwrap();
        assert!(!ctx.sketch.lines[r].constraints.has_length);
    }

    #[test]
    fn test_cmd_set_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; length L0 3");
        assert!(!ctx.sketch.dimensions[0].derived);
        run_ok(&mut ctx, "set_derived d0");
        // Find the derived dim (might be re-added with same name)
        let dim = ctx.sketch.dimensions.iter().find(|d| d.name == "d0");
        assert!(dim.is_some());
        assert!(dim.unwrap().derived);
    }

    #[test]
    fn test_cmd_set_driven() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        // Create derived dim first
        run_ok(&mut ctx, "length L0 5 derived");
        assert!(ctx.sketch.dimensions[0].derived);
        run_ok(&mut ctx, "set_driven d0 3");
        // Should now be driven
        let dim = ctx.sketch.dimensions.last().unwrap();
        assert!(!dim.derived);
        assert!(near(line_len(&ctx, "L0"), 3.0));
    }

    #[test]
    fn test_cmd_derived_length_measure() {
        // "length L0 derived" should measure current geometry
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,4");
        run_ok(&mut ctx, "length L0 derived");
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 5.0));
    }

    #[test]
    fn test_cmd_derived_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3");
        run_ok(&mut ctx, "radius A0 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 3.0));
    }

    #[test]
    fn test_cmd_derived_angle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 0,5");
        run_ok(&mut ctx, "angle L0 L1 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 90.0));
    }

    #[test]
    fn test_cmd_derived_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 4,0 7,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 derived");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
        assert!(near(ctx.sketch.dimensions[0].value, 1.0));
    }

    // -- Helper point display and cleanup tests --

    fn has_helper_points(ctx: &CommandContext) -> bool {
        ctx.sketch.points.refs().any(|r| ctx.sketch.points[r].helper)
    }

    fn list_constraints_output(ctx: &mut CommandContext) -> String {
        run_ok(ctx, "list constraints")
    }

    // 6A: Display tests -- list constraints shows no Pc names

    #[test]
    fn test_list_no_pc_distance_ll_endpoints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
        assert!(out.contains("distance"), "should list distance constraint: {}", out);
    }

    #[test]
    fn test_list_no_pc_distance_arc_endpoints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center A1.center 10");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
        assert!(out.contains("distance") && out.contains("A0.center") && out.contains("A1.center"),
            "should show semantic names: {}", out);
    }

    #[test]
    fn test_list_no_pc_distance_mixed_arc_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center L0.p1 10");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    }

    #[test]
    fn test_list_no_pc_distance_point_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    }

    #[test]
    fn test_list_no_pc_symmetry_pp_endpoints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
        run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
        assert!(out.contains("symmetry") && out.contains("L0.p1") && out.contains("L0.p2"),
            "should show semantic names: {}", out);
    }

    #[test]
    fn test_list_no_pc_symmetry_pp_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 5,-5 5,5; add_circle 10,0 3");
        run_ok(&mut ctx, "symmetry A0.center L0 A1.center");
        let out = list_constraints_output(&mut ctx);
        assert!(!out.contains("Pc"), "list should not contain Pc: {}", out);
    }

    #[test]
    fn test_list_no_bridge_constraints() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
        let out = list_constraints_output(&mut ctx);
        // Should not contain bridge coincident entries
        let lines: Vec<&str> = out.lines().collect();
        for line in &lines {
            if line.starts_with("coincident") {
                assert!(!line.contains("Pc"), "bridge constraint should be hidden: {}", line);
            }
        }
    }

    // 6B: Cleanup on object deletion
    // Note: Line-Line endpoint distances (DistanceLL*) don't create helpers.
    // Helpers are created for Arc endpoint distances and PointLineDistance
    // with non-Point endpoints.

    #[test]
    fn test_cleanup_delete_line_removes_symmetry_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
        run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
        assert!(has_helper_points(&ctx), "should have helpers after symmetry");
        run_ok(&mut ctx, "delete L0");
        assert!(!has_helper_points(&ctx), "helpers should be cleaned up after delete L0");
        assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry_pp should be empty");
    }

    #[test]
    fn test_cleanup_delete_arc_removes_distance_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center A1.center 10");
        assert!(has_helper_points(&ctx), "should have helpers");
        run_ok(&mut ctx, "delete A0");
        assert!(!has_helper_points(&ctx), "helpers should be cleaned up");
        assert!(ctx.sketch.distance_pp.is_empty(), "distance_pp should be empty");
    }

    #[test]
    fn test_cleanup_delete_arc_removes_symmetry_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 5,-5 5,5; add_circle 10,0 3");
        run_ok(&mut ctx, "symmetry A0.center L0 A1.center");
        assert!(has_helper_points(&ctx), "should have helpers");
        run_ok(&mut ctx, "delete A0");
        assert!(!has_helper_points(&ctx), "helpers should be cleaned up");
        assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry_pp should be empty");
    }

    #[test]
    fn test_cleanup_delete_mirror_line_removes_symmetry_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 3,-5 3,5");
        run_ok(&mut ctx, "symmetry L0.p1 L1 L0.p2");
        assert!(!ctx.sketch.symmetry_pp.is_empty());
        run_ok(&mut ctx, "delete L1");
        assert!(ctx.sketch.symmetry_pp.is_empty(), "symmetry gone after mirror line deleted");
        assert!(!has_helper_points(&ctx), "helpers cleaned up");
    }

    #[test]
    fn test_cleanup_delete_line_removes_pl_distance_helpers() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        // distance from line endpoint to other line creates a helper
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        assert!(has_helper_points(&ctx), "should have helper for L0.p1");
        run_ok(&mut ctx, "delete L0");
        assert!(!has_helper_points(&ctx), "helpers cleaned up");
        assert!(ctx.sketch.distance_pl.is_empty(), "distance_pl should be empty");
    }

    // 6C: Cleanup on dimension removal

    #[test]
    fn test_cleanup_remove_dim_distance_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_circle 10,0 2");
        run_ok(&mut ctx, "distance A0.center A1.center 10");
        assert!(has_helper_points(&ctx));
        run_ok(&mut ctx, "remove_dim d0");
        assert!(!has_helper_points(&ctx), "helpers cleaned up after remove_dim");
    }

    #[test]
    fn test_cleanup_remove_dim_distance_point_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,3 5,3");
        run_ok(&mut ctx, "distance L0.p1 L1 3");
        assert!(has_helper_points(&ctx));
        run_ok(&mut ctx, "remove_dim d0");
        assert!(!has_helper_points(&ctx), "helpers cleaned up after remove_dim");
    }

    #[test]
    fn test_cleanup_remove_dim_distance_arc_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 0,5 5,5");
        run_ok(&mut ctx, "distance A0.center L0 5");
        assert!(has_helper_points(&ctx));
        run_ok(&mut ctx, "remove_dim d0");
        assert!(!has_helper_points(&ctx), "helpers cleaned up after remove_dim");
    }

    #[test]
    fn test_cleanup_remove_dim_distance_arc_mixed() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 3; add_line 10,0 15,0");
        run_ok(&mut ctx, "distance A0.center L0.p1 10");
        assert!(has_helper_points(&ctx));
        run_ok(&mut ctx, "remove_dim d0");
        assert!(!has_helper_points(&ctx), "helpers cleaned up after remove_dim");
    }

    // 6D: No Pc in distance constraints that don't need helpers (regression)

    #[test]
    fn test_no_helpers_for_line_line_distance() {
        // Line-Line endpoint distances use specialized constraints, no helpers
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 3,0; add_line 5,0 8,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 2");
        assert!(!has_helper_points(&ctx), "DistanceLL should not create helpers");
    }

    // -- Autocomplete tests --

    fn setup_complete_ctx() -> CommandContext {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");   // L0
        run_ok(&mut ctx, "add_line 5,0 5,5");   // L1
        run_ok(&mut ctx, "add_point 2,3");       // P0
        run_ok(&mut ctx, "add_circle 3,3 2");    // A0
        run_ok(&mut ctx, "length L0 5");         // d0
        run_ok(&mut ctx, "param width 10");
        ctx
    }

    fn completions(ctx: &CommandContext, input: &str) -> Vec<String> {
        complete(&ctx.sketch, &ctx.session_names, input, input.len())
    }

    // -- DOF check on constraints --

    #[test]
    fn test_dof_check_accepts_valid() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        // horizontal removes 1 DOF, should succeed
    }

    #[test]
    fn test_dof_check_force_overrides() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "horizontal L0");
        // Adding a redundant perpendicular to a non-existent L1 won't work,
        // but we can test force with a degenerate scenario:
        // Two parallel lines, then add parallel again with force
        run_ok(&mut ctx, "add_line 0,1 5,1");
        run_ok(&mut ctx, "parallel L0 L1");
        // collinear on already-parallel lines is degenerate — might not reduce DOF
        // Try with force
        run_ok(&mut ctx, "collinear L0 L1 force");
    }

    // -- DOF analysis --

    #[test]
    fn test_dof_analyze_unconstrained_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        let out = run_ok(&mut ctx, "dof analyze");
        assert!(out.contains("DOF: 4"), "Unconstrained line should have 4 DOF: {}", out);
    }

    #[test]
    fn test_dof_analyze_constrained() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; horizontal L0; length L0 5; lock L0.p1 0,0");
        let out = run_ok(&mut ctx, "dof analyze");
        assert!(out.contains("DOF: 0"), "Fully constrained should be DOF 0: {}", out);
    }

    #[test]
    fn test_dof_analyze_partial() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; horizontal L0; length L0 5");
        let out = run_ok(&mut ctx, "dof analyze");
        // 4 DOF - 1 (horizontal) - 1 (length) = 2 DOF (translate X, Y)
        assert!(out.contains("DOF: 2"), "Should have 2 DOF: {}", out);
        assert!(out.contains("translate"), "Should identify translation: {}", out);
    }

    #[test]
    fn test_dof_analyze_empty() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "clear");
        let out = run_ok(&mut ctx, "dof analyze");
        assert!(out.contains("DOF: 0"), "Empty sketch should be 0 DOF: {}", out);
    }

    // -- point_on with arc endpoints --

    #[test]
    fn test_point_on_arc_center_on_line_no_duplicate() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_circle 2,1 1");
        run_ok(&mut ctx, "point_on A0.center L0");
        let arc_center_count = ctx.sketch.coincident_arc_center.len();
        let pol_count = ctx.sketch.point_on_line.len();
        eprintln!("After first: arc_center={}, point_on_line={}", arc_center_count, pol_count);
        let out2 = run_err(&mut ctx, "point_on A0.center L0");
        eprintln!("Second attempt: {}", out2);
        assert!(out2.contains("already exists"), "Should reject duplicate: {}", out2);
    }

    #[test]
    fn test_point_on_arc_center_on_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0.5 2; add_line -5,0 5,0");
        let out = run_ok(&mut ctx, "point_on A0.center L0");
        assert!(out.contains("point-on-line"), "Should succeed: {}", out);
        // Verify helper point was created and point_on_line constraint exists
        assert!(!ctx.sketch.point_on_line.is_empty(),
            "Should have point_on_line constraint");
    }

    #[test]
    fn test_point_on_arc_center_on_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_circle 4.5,0 1");
        let out = run_ok(&mut ctx, "point_on A1.center A0");
        assert!(out.contains("point-on-arc"), "Should succeed: {}", out);
        assert!(!ctx.sketch.point_on_arc.is_empty(),
            "Should have point_on_arc constraint");
    }

    // -- Dimension update (no duplicates) --

    #[test]
    fn test_dimension_update_length() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0");
        run_ok(&mut ctx, "length L0 5");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "length L0 10");
        assert!(out.contains("Updated"), "Should update existing: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1, "Should still be 1 dimension");
    }

    #[test]
    fn test_dimension_update_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "radius A0 10");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_update_radius_to_expr() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5");
        run_ok(&mut ctx, "param scale 2");
        let out = run_ok(&mut ctx, "radius A0 \"5*scale\"");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_update_angle() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,0 3,4");
        run_ok(&mut ctx, "angle L0 L1 45");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "angle L0 L1 90");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_dimension_update_distance() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 7,0 10,0");
        run_ok(&mut ctx, "distance L0.p2 L1.p1 3");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "distance L0.p2 L1.p1 5");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    #[test]
    fn test_dimension_expr_constrains() {
        // Exact reproduction from bug report
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "clear");
        run_ok(&mut ctx, "param scale 1");
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5*scale");
        // Check that expr_constraints were created
        ctx.sketch.solve();
        assert!(!ctx.sketch.expr_constraints.is_empty(),
            "Expression dimension should create expr_constraint, got none. dims: {:?}",
            ctx.sketch.dimensions.iter().map(|d| (&d.name, &d.expr_str, d.value)).collect::<Vec<_>>());
        // Verify it actually constrains the radius
        let r = ctx.sketch.arcs.refs().next().unwrap();
        let radius = ctx.sketch.arcs[r].radius.value;
        assert!((radius - 5.0).abs() < 0.1,
            "radius should be 5*1=5, got {}", radius);
    }

    #[test]
    fn test_dimension_expr_constrains_fresh() {
        // Fresh expression dim without prior numeric — exact bug report scenario
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "param scale 1");
        run_ok(&mut ctx, "add_circle 0,0 5");
        // No prior "radius A0 5" — go straight to expression
        run_ok(&mut ctx, "radius A0 5*scale");
        // Check dimension was created with expression
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert_eq!(ctx.sketch.dimensions[0].expr_str.as_deref(), Some("5*scale"));
        // Solve and check expr constraints are built
        let result = ctx.sketch.solve();
        assert!(!ctx.sketch.expr_constraints.is_empty(),
            "Should have expr_constraints after solve");
        // Check radius is actually constrained to 5
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!((ctx.sketch.arcs[r].radius.value - 5.0).abs() < 0.1,
            "radius should be 5*1=5, got {}", ctx.sketch.arcs[r].radius.value);
        assert!(result.end_cost < 0.01, "cost should be near zero, got {}", result.end_cost);
    }

    #[test]
    fn test_dimension_expr_update_constrains() {
        // Updating numeric dim to expression should constrain
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        run_ok(&mut ctx, "radius A0 5");
        run_ok(&mut ctx, "param scale 3");
        run_ok(&mut ctx, "radius A0 \"2*scale\"");
        ctx.sketch.solve();
        assert!(!ctx.sketch.expr_constraints.is_empty(),
            "Updated expression should create expr_constraint");
        let r = ctx.sketch.arcs.refs().next().unwrap();
        let radius = ctx.sketch.arcs[r].radius.value;
        assert!((radius - 6.0).abs() < 0.1,
            "radius should be 2*3=6, got {}", radius);
    }

    #[test]
    fn test_dimension_no_cross_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "length L0 5");
        run_ok(&mut ctx, "length L1 3");
        assert_eq!(ctx.sketch.dimensions.len(), 2, "Different entities should have separate dims");
    }

    // -- Autocomplete tests --

    #[test]
    fn test_complete_empty_input() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "").is_empty());
    }

    #[test]
    fn test_complete_first_token_commands() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_");
        assert!(c.contains(&"add_line".to_string()));
        assert!(c.contains(&"add_point".to_string()));
        assert!(c.contains(&"add_circle".to_string()));
        // Should NOT contain entity names
        assert!(!c.iter().any(|s| s.starts_with('L')));
    }

    #[test]
    fn test_complete_list_filters_not_entities() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "list l");
        assert!(c.contains(&"lines".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "list should not offer entity names: {:?}", c);
    }

    #[test]
    fn test_complete_cursor_keywords() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "cursor o");
        assert!(c.contains(&"on".to_string()));
        assert!(c.contains(&"off".to_string()));
    }

    #[test]
    fn test_complete_add_line_cursor() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line curs");
        assert!(c.contains(&"cursor".to_string()));
    }

    #[test]
    fn test_complete_horizontal_lines_only() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "horizontal L");
        assert!(c.contains(&"L0".to_string()));
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('A')), "horizontal should not offer arcs");
        assert!(!c.iter().any(|s| s.starts_with('P')), "horizontal should not offer points");
    }

    #[test]
    fn test_complete_concentric_arcs_only() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "concentric A");
        assert!(c.contains(&"A0".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "concentric should not offer lines");
    }

    #[test]
    fn test_complete_style_values() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "style L0 d");
        assert!(c.contains(&"dashed".to_string()));
        assert!(c.contains(&"dashdot".to_string()));
        assert!(!c.iter().any(|s| s.starts_with("d0")), "style arg2 should not offer dimensions");
    }

    #[test]
    fn test_complete_remove_dim() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "remove_dim d");
        assert!(c.contains(&"d0".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "remove_dim should not offer lines");
    }

    #[test]
    fn test_complete_length_arg2_derived() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "length L0 d");
        assert!(c.contains(&"derived".to_string()));
        // Should offer dimension refs in expression context
        assert!(c.contains(&"d0".to_string()));
        // Should NOT offer lines
        assert!(!c.contains(&"L0".to_string()), "length arg2 should not offer L0");
    }

    #[test]
    fn test_complete_equal_type_matching() {
        let ctx = setup_complete_ctx();
        // After "equal L0", should only offer lines
        let c = completions(&ctx, "equal L0 L");
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('A')), "equal with L0 should not offer arcs");
    }

    #[test]
    fn test_complete_dim_pos() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "dim_pos d0 o");
        assert!(c.contains(&"offset".to_string()));
        let c = completions(&ctx, "dim_pos d0 a");
        assert!(c.contains(&"along".to_string()));
    }

    #[test]
    fn test_complete_no_arg_commands() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "dof x").is_empty());
        assert!(completions(&ctx, "cost x").is_empty());
    }

    #[test]
    fn test_complete_del_param() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "del_param w");
        assert!(c.contains(&"width".to_string()));
    }

    #[test]
    fn test_complete_rc_constraint_types() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "rc L0 h");
        assert!(c.contains(&"horizontal".to_string()));
    }

    #[test]
    fn test_complete_dot_line() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "info L0.");
        assert!(c.contains(&"L0.p1".to_string()));
        assert!(c.contains(&"L0.p2".to_string()));
    }

    #[test]
    fn test_complete_dot_arc() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "info A0.");
        assert!(c.contains(&"A0.center".to_string()));
        assert!(c.contains(&"A0.start".to_string()));
        assert!(c.contains(&"A0.end".to_string()));
    }

    #[test]
    fn test_complete_midpoint_arg2_lines_only() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "midpoint P0 L");
        assert!(c.contains(&"L0".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('A')), "midpoint arg2 should not offer arcs");
    }

    #[test]
    fn test_complete_offset_line_arg1() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "offset L");
        assert!(c.contains(&"L0".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('A')));
    }

    #[test]
    fn test_complete_list_space_shows_options() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "list ");
        assert!(c.contains(&"lines".to_string()));
        assert!(c.contains(&"constraints".to_string()));
    }

    #[test]
    fn test_complete_horizontal_space_shows_lines() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "horizontal ");
        assert!(c.contains(&"L0".to_string()));
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('A')));
    }

    #[test]
    fn test_complete_cursor_space_shows_keywords() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "cursor ");
        assert!(c.contains(&"on".to_string()));
        assert!(c.contains(&"off".to_string()));
    }

    #[test]
    fn test_complete_style_space_shows_entities() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "style ");
        assert!(c.contains(&"L0".to_string()));
    }

    #[test]
    fn test_complete_style_entity_space_shows_values() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "style L0 ");
        assert!(c.contains(&"solid".to_string()));
        assert!(c.contains(&"dashed".to_string()));
    }

    #[test]
    fn test_complete_empty_first_token_no_suggestions() {
        let ctx = setup_complete_ctx();
        // Just a space or empty — no suggestions
        assert!(completions(&ctx, "").is_empty());
    }

    #[test]
    fn test_complete_add_line_after_coords_only_flags() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line 0,0 5,0 ");
        assert!(c.contains(&"noconnect".to_string()));
        assert!(c.contains(&"nocursor".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "Should not offer entities after coords: {:?}", c);
    }

    #[test]
    fn test_complete_add_line_flag_excludes_typed() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line 0,0 5,0 nocursor ");
        assert!(c.contains(&"noconnect".to_string()));
        assert!(!c.contains(&"nocursor".to_string()), "Should not re-offer nocursor");
    }

    #[test]
    fn test_complete_add_line_first_coord() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_line curs");
        assert!(c.contains(&"cursor".to_string()));
    }

    #[test]
    fn test_complete_add_point_after_coord() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_point 0,0 ");
        assert!(c.contains(&"nocursor".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')), "Should not offer entities: {:?}", c);
    }

    #[test]
    fn test_complete_add_circle_radius_position() {
        let ctx = setup_complete_ctx();
        // After center coord, radius is expression context
        let c = completions(&ctx, "add_circle 0,0 w");
        assert!(c.contains(&"width".to_string()));
        assert!(!c.contains(&"cursor".to_string()));
    }

    #[test]
    fn test_complete_add_arc_after_3_coords() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "add_arc 0,0 5,0 2,3 ");
        assert!(c.contains(&"noconnect".to_string()));
        assert!(!c.iter().any(|s| s.starts_with('L')));
    }

    #[test]
    fn test_complete_help_full() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "help f");
        assert!(c.contains(&"full".to_string()));
    }

    #[test]
    fn test_complete_list_all_keyword() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "list a");
        assert!(c.contains(&"all".to_string()));
        assert!(c.contains(&"arcs".to_string()));
    }

    #[test]
    fn test_complete_list_no_second_arg() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "list lines ").is_empty());
    }

    #[test]
    fn test_complete_horizontal_excludes_typed() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "horizontal L0 L");
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.contains(&"L0".to_string()), "Should exclude already-typed L0");
    }

    #[test]
    fn test_complete_select_excludes_typed() {
        let ctx = setup_complete_ctx();
        let c = completions(&ctx, "select L0 P0 L");
        assert!(c.contains(&"L1".to_string()));
        assert!(!c.contains(&"L0".to_string()), "Should exclude already-typed L0");
    }

    #[test]
    fn test_complete_cursor_no_second_arg() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "cursor on ").is_empty());
    }

    #[test]
    fn test_complete_parallel_no_third_arg() {
        let ctx = setup_complete_ctx();
        assert!(completions(&ctx, "parallel L0 L1 ").is_empty());
    }

    // -- sweep tests --

    #[test]
    fn test_sweep_basic() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        let out = run_ok(&mut ctx, "sweep A0 180");
        assert!(out.contains("Set") || out.contains("sweep"), "Should succeed: {}", out);
        assert!(ctx.sketch.arcs.refs().next().map(|r| ctx.sketch.arcs[r].constraints.has_target_sweep).unwrap_or(false));
        // Solve and check sweep is close to 180 degrees
        ctx.sketch.solve();
        let r = ctx.sketch.arcs.refs().next().unwrap();
        let sweep = (ctx.sketch.arcs[r].end_angle.value - ctx.sketch.arcs[r].start_angle.value).to_degrees();
        assert!((sweep - 180.0).abs() < 1.0, "Sweep should be ~180, got {}", sweep);
    }

    #[test]
    fn test_sweep_update() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        run_ok(&mut ctx, "sweep A0 180");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let out = run_ok(&mut ctx, "sweep A0 90");
        assert!(out.contains("Updated"), "Should update: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
    }

    #[test]
    fn test_sweep_derived() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        let out = run_ok(&mut ctx, "sweep A0 derived");
        assert!(out.contains("Derived"), "Should be derived: {}", out);
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        assert!(ctx.sketch.dimensions[0].derived);
    }

    #[test]
    fn test_sweep_expression() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        run_ok(&mut ctx, "param n 2");
        let out = run_ok(&mut ctx, "sweep A0 \"90*n\"");
        assert!(out.contains("Set") || out.contains("sweep"), "Should succeed: {}", out);
    }

    #[test]
    fn test_sweep_full_circle_rejected() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5");
        let e = run_err(&mut ctx, "sweep A0 180");
        assert!(e.contains("full circle"), "Should reject: {}", e);
    }

    #[test]
    fn test_sweep_remove() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        run_ok(&mut ctx, "sweep A0 180");
        assert_eq!(ctx.sketch.dimensions.len(), 1);
        let name = ctx.sketch.dimensions[0].name.clone();
        run_ok(&mut ctx, &format!("remove_dim {}", name));
        assert_eq!(ctx.sketch.dimensions.len(), 0);
        let r = ctx.sketch.arcs.refs().next().unwrap();
        assert!(!ctx.sketch.arcs[r].constraints.has_target_sweep);
    }

    // -- arc derived properties in expressions --

    #[test]
    fn test_print_arc_start_end() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_arc -5,0 5,0 0,5");
        let out = run_ok(&mut ctx, "print A0.start.x");
        // Should return a number, not an error
        assert!(out.parse::<f64>().is_ok() || out.trim().parse::<f64>().is_ok(),
            "A0.start.x should be a number: {}", out);
        run_ok(&mut ctx, "print A0.start.y");
        run_ok(&mut ctx, "print A0.end.x");
        run_ok(&mut ctx, "print A0.end.y");
        run_ok(&mut ctx, "print A0.sweep");
        run_ok(&mut ctx, "print A0.diameter");
    }

    // -- remove_constraint tests --

    #[test]
    fn test_remove_constraint_coincident_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 0,0; add_point 1,0");
        run_ok(&mut ctx, "coincident P0 P1");
        assert_eq!(ctx.sketch.coincident_pp.len(), 1);
        run_ok(&mut ctx, "remove_constraint P0 P1 coincident");
        assert_eq!(ctx.sketch.coincident_pp.len(), 0);
    }

    #[test]
    fn test_remove_constraint_coincident_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
        run_ok(&mut ctx, "coincident L0.p2 L1.p1");
        assert_eq!(ctx.sketch.coincident_ll21.len(), 1);
        run_ok(&mut ctx, "remove_constraint L0.p2 L1.p1 coincident");
        assert_eq!(ctx.sketch.coincident_ll21.len(), 0);
    }

    #[test]
    fn test_remove_constraint_coincident_not_found() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 5,1 10,1");
        let e = run_err(&mut ctx, "remove_constraint L0.p2 L1.p1 coincident");
        assert!(e.contains("not found"), "{}", e);
    }

    #[test]
    fn test_remove_constraint_point_on_line() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2,0.5; add_line 0,0 5,0");
        run_ok(&mut ctx, "point_on P0 L0");
        assert_eq!(ctx.sketch.point_on_line.len(), 1);
        run_ok(&mut ctx, "remove_constraint P0 L0 point_on");
        assert_eq!(ctx.sketch.point_on_line.len(), 0);
    }

    #[test]
    fn test_remove_constraint_point_on_line_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line 0,1 5,1");
        run_ok(&mut ctx, "point_on L0.p1 L1");
        assert_eq!(ctx.sketch.line_p1_on_line.len(), 1);
        run_ok(&mut ctx, "remove_constraint L0.p1 L1 point_on");
        assert_eq!(ctx.sketch.line_p1_on_line.len(), 0);
    }

    #[test]
    fn test_remove_constraint_point_on_arc() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 5,0; add_circle 0,0 5");
        run_ok(&mut ctx, "point_on P0 A0");
        assert_eq!(ctx.sketch.point_on_arc.len(), 1);
        run_ok(&mut ctx, "remove_constraint P0 A0 point_on");
        assert_eq!(ctx.sketch.point_on_arc.len(), 0);
    }

    #[test]
    fn test_remove_constraint_point_on_line_arc_endpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0.5 2; add_line -5,0 5,0");
        run_ok(&mut ctx, "point_on A0.center L0");
        assert!(!ctx.sketch.point_on_line.is_empty());
        run_ok(&mut ctx, "remove_constraint A0.center L0 point_on");
        // The point_on_line constraint on the helper should be removed
        // cleanup_helper_points removes orphan helpers
        assert!(ctx.sketch.point_on_line.is_empty() || ctx.sketch.points.refs().all(|p| !ctx.sketch.points[p].helper));
    }

    #[test]
    fn test_remove_constraint_symmetry_pp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point -3,0; add_line 0,-5 0,5; add_point 3,0");
        run_ok(&mut ctx, "symmetry P0 L0 P1");
        assert_eq!(ctx.sketch.symmetry_pp.len(), 1);
        run_ok(&mut ctx, "remove_constraint P0 L0 P1 symmetry");
        assert_eq!(ctx.sketch.symmetry_pp.len(), 0);
    }

    #[test]
    fn test_remove_constraint_symmetry_ll() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line -2,0 -2,3; add_line 0,0 0,5; add_line 2,0 2,3");
        run_ok(&mut ctx, "symmetry L0 L1 L2");
        assert_eq!(ctx.sketch.symmetry_ll.len(), 1);
        run_ok(&mut ctx, "remove_constraint L0 L1 L2 symmetry");
        assert_eq!(ctx.sketch.symmetry_ll.len(), 0);
    }

    #[test]
    fn test_remove_constraint_midpoint() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_point 2.5,0.5; add_line 0,0 5,0");
        run_ok(&mut ctx, "midpoint P0 L0");
        assert_eq!(ctx.sketch.midpoint.len(), 1);
        run_ok(&mut ctx, "remove_constraint P0 L0 midpoint");
        assert_eq!(ctx.sketch.midpoint.len(), 0);
    }

    #[test]
    fn test_remove_constraint_midpoint_lp() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_line 0,0 5,0; add_line -5,0 10,0");
        run_ok(&mut ctx, "midpoint L0.p1 L1");
        assert_eq!(ctx.sketch.midpoint_lp1.len(), 1);
        run_ok(&mut ctx, "remove_constraint L0.p1 L1 midpoint");
        assert_eq!(ctx.sketch.midpoint_lp1.len(), 0);
    }

    #[test]
    fn test_remove_constraint_equal_radius() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_circle 10,0 3");
        run_ok(&mut ctx, "equal A0 A1");
        assert_eq!(ctx.sketch.equal_radius.len(), 1);
        run_ok(&mut ctx, "remove_constraint A0 A1 equal_radius");
        assert_eq!(ctx.sketch.equal_radius.len(), 0);
    }

    #[test]
    fn test_remove_constraint_equal_radius_not_found() {
        let mut ctx = CommandContext::new();
        run_ok(&mut ctx, "add_circle 0,0 5; add_circle 10,0 3");
        let e = run_err(&mut ctx, "remove_constraint A0 A1 equal_radius");
        assert!(e.contains("not found"), "{}", e);
    }
}
