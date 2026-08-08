// Interactive 2D sketch editor with real-time constraint solving.
//
// Tools: Select (drag to solve), Point, Line
// Constraints: Horizontal, Vertical, Coincident (via toolbar)
// Navigation: scroll wheel = zoom, middle mouse drag = pan

mod colors;
mod tools;
mod drawing;
mod app_update;

use arael::simple_lm::RootProblem;
use std::collections::HashMap;
use eframe::egui;
use arael::model::{Param, CrossBlock};
use arael::simple_lm::LmProblem;
use arael::utils::rad2deg;
use arael::vect::vect2d;
use arael::refs::Ref;
use arael_sketch_solver::*;

use colors::ColorScheme;
use tools::*;
use arael_sketch_backend::{Action, History};
use arael_sketch_backend::geometry::*;

pub struct SavedArcLocks {
    pub had_radius: bool,
    pub old_radius: f64,
    pub had_radius_b: bool,
    pub old_radius_b: f64,
    pub rotation_optimize: bool,
    pub had_sweep: bool,
    pub old_sweep: f64,
    pub old_sweep_sign: f64,
    pub start_optimize: bool,
    pub end_optimize: bool,
}
// drawing module methods are accessed through impl blocks on EditorApp

/// Entities involved in a constraint, for highlighting.
/// Whole-entity fields highlight the entire line/arc/point.
/// Endpoint-specific fields highlight just one endpoint.
#[derive(Default)]
pub struct ConstraintEntities {
    pub lines: Vec<Ref<Line>>,
    pub arcs: Vec<Ref<Arc>>,
    pub points: Vec<Ref<Point>>,
    pub line_p1s: Vec<Ref<Line>>,
    pub line_p2s: Vec<Ref<Line>>,
    pub arc_starts: Vec<Ref<Arc>>,
    pub arc_ends: Vec<Ref<Arc>>,
    pub arc_centers: Vec<Ref<Arc>>,
}

/// Spawn an async task without blocking the UI thread.
/// On WASM: uses wasm_bindgen_futures. On native: uses std::thread.
#[cfg(target_arch = "wasm32")]
fn spawn_async<F: std::future::Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_async<F: std::future::Future<Output = ()> + Send + 'static>(f: F) {
    std::thread::spawn(move || pollster::block_on(f));
}

// ---------------------------------------------------------------------------
// DOF computation (Hessian rank analysis)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Editor state
// ---------------------------------------------------------------------------

// Auto-perpendicular snap tolerance: perpendicular distance in screen
// pixels from the cursor to the virtual 90-degree axis before the cursor
// is pulled onto that axis.
const PERP_SNAP_PX: f32 = 10.0;
/// Tolerance on |sin(theta)| for "treat these two lines as parallel" in
/// the interactive dimension tool. ~3 degrees: user clicking precision
/// is coarse, and a false positive self-corrects when the paired
/// Parallel constraint snaps the lines exactly parallel.
const PARALLEL_SELECTION_TOL: f64 = 0.05;

/// Weight of the soft-drag helper's attractor residual. Sits between
/// `drift_isigma` (~1e-3, the per-entity stability regularizer) and
/// `constraint_isigma` (~1e3, hard constraints). At 1.0 the attractor
/// overwhelms background drift so the helper tracks the cursor
/// cleanly on unconstrained DOFs, but yields to any real constraint,
/// leaving the sketch at cost ~ 0 with the dragged endpoint lagging
/// when the cursor target is infeasible.
use arael_sketch_backend::DRAG_PULL_WEIGHT;

/// Maximum distance (in screen pixels) the drag cursor may pull the
/// drag helper beyond its last-solved feasible position. The GUI
/// projects any cursor position outside this ball back onto the
/// boundary before feeding it to the solver, so a very-far drag on
/// a locked sketch cannot cause the drag attractor to overwhelm the
/// constraints and deform the sketch. The clamp reference moves
/// with the sketch as it tracks the cursor -- free DOFs still
/// follow the mouse pixel-for-pixel; only the excess lag when the
/// sketch runs out of slack is absorbed.
const DRAG_MAX_LAG_PX: f32 = 100.0;

pub struct EditorApp {
    pub sketch: SketchCell,
    // View transform
    pub offset: egui::Vec2,  // pan offset in screen pixels
    pub scale: f32,          // pixels per sketch unit

    // Tools
    pub tool: Tool,
    pub line_draw: Option<LineDrawState>,
    pub circle_draw: Option<CircleDrawState>,
    pub arc_draw: Option<ArcDrawState>,
    pub rect_draw: Option<RectDrawState>,
    /// In-flight fillet from the GUI Fillet tool. Set after the fillet
    /// actions run; kept alive while the radius dim-input overlay is
    /// open so Escape can restore the pre-fillet sketch.
    pub fillet_pending: Option<FilletPending>,

    // Selection and hover
    pub selection: Vec<Selection>,
    pub hovered: Option<Selection>,

    // Drag state
    pub grab: Option<GrabTarget>,
    pub drag_point: Option<Ref<Point>>,  // temporary drag point
    pub drag_point2: Option<Ref<Point>>, // second drag point (line body drag)
    pub drag_offset: vect2d,             // offset from mouse to drag point
    pub drag_offset2: vect2d,            // offset for second drag point
    pub drag_saved_arc_locks: Option<SavedArcLocks>,
    pub drag_dimension: Option<usize>,   // index of dimension being dragged
    /// Live midpoint snap during endpoint drag: (snap position in sketch
    /// coords, snap target). Populated every `update_drag`, cleared by
    /// `end_drag`. Used so the preview render can show the midpoint
    /// marker without re-running find_snap_target.
    pub drag_snap_preview: Option<(vect2d, SnapTarget)>,
    /// Auto-perpendicular hint during line-endpoint drag: the host line
    /// the drawn line will be made perpendicular to, plus the anchor
    /// (opposite endpoint) position for drawing the corner marker.
    /// Populated only when the cursor has no stronger snap and the
    /// drawn-line direction is near 90 degrees to the host.
    pub drag_perp_snap: Option<(Ref<Line>, vect2d)>,
    /// While true, all snap detection and auto-constraint emission is
    /// suppressed -- creation tools place their clicks at the raw cursor
    /// position, drag lets the endpoint land anywhere, and no coincident
    /// or perpendicular constraints are added. Refreshed per frame from
    /// the Shift key state.
    pub snap_disabled: bool,

    // Undo/redo
    pub history: History,

    // Constraint markers (rebuilt each frame for hit testing)
    pub constraint_markers: Vec<ConstraintMarker>,

    // File
    pub pending_load: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    pub pending_fit: bool,

    // Dimension input
    pub dim_input: String,          // text being typed for dimension value
    pub dim_editing: bool,          // true when text input is active
    pub dim_kind: Option<DimensionKind>, // what dimension is being created
    pub dim_placing: bool,          // true when positioning the dimension with mouse
    pub dim_offset: vect2d,         // current offset being placed
    pub dim_text_along: f64,        // text position along line during creation
    pub dim_edit_index: Option<usize>, // index of dimension being edited (for double-click edit)
    pub dim_select_all: bool,           // one-shot: select all text on next frame
    pub dim_derived: bool,              // checkbox state for derived (reference) dimensions
    pub dim_derived_prev: bool,         // previous frame's `dim_derived`, for edge-detect
    pub dim_input_backup: String,       // `dim_input` saved when `dim_derived` went false->true,
                                        // restored when it goes back true->false
    pub dim_select_all_on_uncheck: bool, // one-shot: select all after uncheck restore

    // Display
    pub show_constraints: bool,
    pub show_dimensions: bool,
    pub show_points: bool,

    // Theme
    pub dark_mode: bool,
    pub colors: ColorScheme,

    // Parameters panel
    pub show_params: bool,
    pub param_new_name: String,
    pub param_new_expr: String,
    pub param_edit_index: Option<usize>,
    pub param_edit_name: String,
    pub param_edit_expr: String,
    pub param_focus_new: bool,
    pub param_focus_field: Option<bool>, // None=no focus request, Some(false)=name, Some(true)=expr

    // Command panel
    pub show_command: bool,
    pub help_expand: bool,
    pub help_scroll_top: bool,
    pub command_scroll_to_bottom: bool,
    pub show_hints: bool,
    pub command_input: String,
    pub command_history: Vec<String>,
    pub command_history_pos: usize,
    pub command_output: Vec<(String, bool, bool)>, // (text, is_error, is_markdown)
    pub command_focus: bool,                  // request focus next frame
    pub command_has_focus: bool,              // true if command input had focus last frame
    pub completions: Vec<String>,             // autocomplete suggestions
    pub completion_idx: usize,                // selected suggestion index
    pub completion_suppressed: bool,          // true after Escape dismisses popup; reset on space/dot
    pub command_cursor: Option<vect2d>,   // for chaining add_line
    pub command_cursor_tangent: Option<vect2d>,
    pub session_vars: HashMap<String, f64>,  // session variables from 'let' commands
    pub session_vecs: HashMap<String, vect2d>, // session coordinate variables
    pub session_names: HashMap<String, String>, // entity name aliases

    // Echo command output to stdout (--stdout flag)
    pub echo_stdout: bool,
    // Raw-drag mode (--drag-raw flag): use the old hard-pin drag where a
    // fixed helper point forces the dragged endpoint to the cursor exactly,
    // deforming the sketch when that is infeasible. When false (default),
    // the drag helper is an optimizable Point so the Point drift residual
    // softly anchors it at the cursor -- hard constraints overrule, so the
    // user sees the relaxed cost ~ 0 state while dragging.
    pub drag_raw: bool,
    // Exit requested by the exit command
    pub exit_requested: bool,

    // Constraint conflict error message
    pub status_error: Option<String>,
    /// Flash state: when a constraint is rejected, briefly highlight
    /// the conflicting constraints so the user can see what's blocking
    /// the new one, even if those constraints are normally invisible
    /// (coincident bridges, flag-style H/V). See `start_constraint_flash`.
    pub flash_names: Vec<String>,
    pub flash_start: Option<web_time::Instant>,
    /// Short-lived cache of failed `perp_would_reduce_dof` checks.
    /// On a big sketch a full DOF recompute can be tens of ms, too
    /// much for 60 Hz drag preview. Once we've determined that a
    /// perpendicular between a given `(line, host)` pair wouldn't
    /// reduce DOF, skip re-checking for ~1 second so the drag
    /// remains responsive even if the user keeps hovering near the
    /// same perpendicular anchor.
    pub drag_perp_fail_cache: Option<(u32, u32, web_time::Instant)>,
    /// Classical box-select: when the user starts dragging on empty
    /// canvas (nothing grabbable at the press origin), we enter
    /// box-select mode. The start position is the press-origin in
    /// screen coords so the rect renders correctly across pan/zoom
    /// during the drag. Cleared when the drag completes.
    pub box_select_start: Option<egui::Pos2>,
    /// Pairs (line, host) that were already (directly or implicitly)
    /// perpendicular at drag-start. During drag the geometry is briefly
    /// off-axis so a rank-based re-check would spuriously report the
    /// perp as DOF-reducing; suppressing these pairs for the whole drag
    /// avoids phantom perp hints. Checked once per drag, reused every
    /// frame -- also saves the per-frame DOF recompute that caused the
    /// drag to feel choppy.
    pub drag_perp_already: Vec<(u32, u32)>,
    /// Auto-horizontal / auto-vertical hint during line-endpoint drag.
    /// `(line, true)` means "snap the dragged line to horizontal on
    /// release", `(line, false)` means vertical. Only populated when
    /// the line did not already carry an H or V constraint at drag
    /// start -- the snap turns a free-angle line into an axis-aligned
    /// one, same pattern as the existing auto-perpendicular.
    pub drag_hv_hint: Option<(Ref<Line>, bool)>,
    /// Captured at drag-start: true if the dragged line is already
    /// locked to horizontal / vertical by the existing constraint
    /// system -- either via the direct flag or implicitly through
    /// other constraints (e.g. a triangle side made horizontal by
    /// two parallel constraints + a pinned partner). Suppresses the
    /// auto-H/V hint for the duration of the drag: the line already
    /// IS that axis, so proposing to make it so is noise and would
    /// just be rejected by the DOF check on release.
    pub drag_line_locked_h: bool,
    pub drag_line_locked_v: bool,
    /// Captured at drag-start: true if the dragged line's dragged
    /// endpoint is coincident with / on / tied to some other entity.
    /// Suppresses the auto-H/V hint for interior vertices of a
    /// segmented structure. Snapshotted at drag-start so the
    /// drag-helper coincident pushed a moment later doesn't itself
    /// count as a "connection".
    pub drag_line_endpoint_connected: bool,
    /// Pairs (line, host) already collinear at drag-start. Used the
    /// same way as drag_perp_already: skip the hint / DOF probe for
    /// the duration of the drag when the line is structurally already
    /// collinear with the host, avoiding phantom hints and the
    /// per-frame rank recompute.
    pub drag_collinear_already: Vec<(u32, u32)>,
    /// Auto-collinear hint during line-endpoint drag.
    pub drag_collinear_hint: Option<(Ref<Line>, Ref<Line>)>,
    pub last_cost: f64,
    drag_saved_cost: f64,              // best cost seen during drag
    drag_saved_snapshot: Option<Vec<u8>>, // sketch state at that best cost
    /// State token for the auto-anchor hack (helper Points on every
    /// free chain endpoint), pushed at start_drag and rolled back at
    /// end_drag. See Sketch::add_drag_auto_anchors for the rationale.
    drag_auto_anchors: Option<arael_sketch_solver::DragAutoAnchorState>,

    // DOF (degrees of freedom) computed in background thread.
    // Single worker thread reads latest sketch data from dof_input,
    // computes DOF, writes result to dof_output. Intermediate requests
    // are overwritten -- only the newest sketch state is computed.
    pub dof_display: Option<usize>,    // None = computing or unknown
    dof_input: std::sync::Arc<std::sync::Mutex<Option<Vec<u8>>>>,
    dof_output: std::sync::Arc<std::sync::Mutex<Option<usize>>>,

    // MCP server channel (None when --mcp not used)
    #[cfg(not(target_arch = "wasm32"))]
    pub mcp_rx: Option<tokio::sync::mpsc::Receiver<arael_sketch_backend::mcp_server::McpRequest>>,
    // Shared egui context for the MCP server to wake the GUI thread.
    // Populated the first frame update() is called; captured inside the
    // wake callback handed to the backend MCP server.
    #[cfg(not(target_arch = "wasm32"))]
    egui_ctx: std::sync::Arc<std::sync::Mutex<Option<egui::Context>>>,
}

impl EditorApp {
    fn demo() -> Self {
        let mut sketch = Sketch::new();

        // Isoceles triangle with a circle at the apex
        // L0: apex (0,0) -> bottom-left (-1.5,-5)
        // L1: bottom-left -> bottom-right (1.5,-5), length=3
        // L2: bottom-right -> apex
        let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(-1.5, -5.0));
        let l1 = sketch.add_line(vect2d::new(-1.5, -5.0), vect2d::new(1.5, -5.0));
        let l2 = sketch.add_line(vect2d::new(1.5, -5.0), vect2d::new(0.0, 0.0));

        // Connect corners: L0.p2=L1.p1, L1.p2=L2.p1, L2.p2=L0.p1
        sketch.coincident_ll21.push(CoincidentLL21 { a: l0, b: l1, nid: 0, cid: 0, hb: CrossBlock::new() });
        sketch.coincident_ll21.push(CoincidentLL21 { a: l1, b: l2, nid: 0, cid: 0, hb: CrossBlock::new() });
        sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l0, nid: 0, cid: 0, hb: CrossBlock::new() });



        // Circle (full arc, r=1.5) centered at apex
        let a0 = sketch.add_arc(vect2d::new(0.0, 0.0), 1.5, 0.0, std::f64::consts::TAU, true);

        // Equal length: L0 = L2 (isoceles)
        sketch.equal_length.push(EqualLength { a: l2, b: l0, nid: 0, cid: 0, hb: CrossBlock::new() });

        // Arc center = L0.p1 (apex)
        sketch.coincident_lp1_arc_center.push(CoincidentLP1ArcCenter { line: l0, arc: a0, nid: 0, cid: 0, hb: CrossBlock::new() });

        // User parameter for the base length
        sketch.user_params.push(UserParam {
            name: "base_length".into(), expr_str: "3".into(), value: 3.0, broken: false,
        });

        // Dimensions (via Action, then adjust offsets for nice layout)
        Action::AddDimension { kind: DimensionKind::LineLength(l1), value: 0.0, expr: Some("base_length".into()), derived: false, range: None }.apply(&mut sketch);
        sketch.dimensions.last_mut().unwrap().offset = vect2d::new(0.0, -0.32);
        sketch.dimensions.last_mut().unwrap().text_along = -0.27;

        Action::AddDimension {
            kind: DimensionKind::PointLineDistance(DimensionEndpoint::LineP1(l0), l1), value: 5.0,
            expr: None, derived: false, range: None,
        }.apply(&mut sketch);
        sketch.dimensions.last_mut().unwrap().offset = vect2d::new(0.0, 1.72);
        sketch.dimensions.last_mut().unwrap().text_along = 0.20;

        Action::AddDimension { kind: DimensionKind::ArcRadius(a0), value: 1.5, expr: None, derived: false, range: None }.apply(&mut sketch);
        sketch.dimensions.last_mut().unwrap().offset = vect2d::new(0.91, 0.0);

        let result = sketch.solve();
        let last_cost = result.end_cost;
        let history = History::new(&sketch);

        EditorApp {
            sketch: sketch.into(),
            offset: egui::Vec2::new(400.0, 300.0),
            scale: 80.0,
            tool: Tool::Select,
            line_draw: None,
            circle_draw: None,
            arc_draw: None,
            rect_draw: None,
            fillet_pending: None,
            selection: Vec::new(),
            hovered: None,
            grab: None,
            drag_point: None,
            drag_point2: None,
            drag_offset: vect2d::new(0.0, 0.0),
            drag_offset2: vect2d::new(0.0, 0.0),
            drag_saved_arc_locks: None,
            drag_dimension: None,
            drag_snap_preview: None,
            drag_perp_snap: None,
            snap_disabled: false,
            history,
            constraint_markers: Vec::new(),
            pending_load: std::sync::Arc::new(std::sync::Mutex::new(None)),
            pending_fit: true,
            dim_input: String::new(),
            dim_editing: false,
            dim_kind: None,
            dim_placing: false,
            dim_offset: vect2d::new(0.0, 1.0),
            dim_text_along: 0.0,
            dim_edit_index: None,
            dim_select_all: false,
            dim_derived: false,
            dim_derived_prev: false,
            dim_input_backup: String::new(),
            dim_select_all_on_uncheck: false,
            show_constraints: true,
            show_dimensions: true,
            show_points: true,
            show_params: false,
            param_new_name: String::new(),
            param_new_expr: String::new(),
            param_edit_index: None,
            param_edit_name: String::new(),
            param_edit_expr: String::new(),
            param_focus_new: false,
            param_focus_field: None,
            show_command: false,
            help_expand: false,
            help_scroll_top: false,
            command_scroll_to_bottom: false,
            show_hints: true,
            command_input: String::new(),
            command_history: Vec::new(),
            command_history_pos: 0,
            command_output: Vec::new(),
            command_focus: false,
            command_has_focus: false,
            completions: Vec::new(),
            completion_idx: 0,
            completion_suppressed: false,
            command_cursor: None,
            command_cursor_tangent: None,
            session_vars: HashMap::new(),
            session_vecs: HashMap::new(),
            session_names: HashMap::new(),
            dark_mode: cfg!(target_arch = "wasm32"),
            colors: if cfg!(target_arch = "wasm32") { ColorScheme::dark() } else { ColorScheme::light() },
            echo_stdout: false,
            drag_raw: false,
            exit_requested: false,
            status_error: None,
            flash_names: Vec::new(),
            flash_start: None,
            drag_perp_fail_cache: None,
            box_select_start: None,
            drag_perp_already: Vec::new(),
            drag_hv_hint: None,
            drag_line_locked_h: false,
            drag_line_locked_v: false,
            drag_line_endpoint_connected: false,
            drag_collinear_already: Vec::new(),
            drag_collinear_hint: None,
            last_cost,
            drag_saved_cost: 0.0,
            drag_saved_snapshot: None,
            drag_auto_anchors: None,
            dof_display: None,
            dof_input: std::sync::Arc::new(std::sync::Mutex::new(None)),
            dof_output: std::sync::Arc::new(std::sync::Mutex::new(None)),
            #[cfg(not(target_arch = "wasm32"))]
            mcp_rx: None,
            #[cfg(not(target_arch = "wasm32"))]
            egui_ctx: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

impl Default for EditorApp {
    fn default() -> Self {
        let mut app = Self::demo();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let input = std::sync::Arc::clone(&app.dof_input);
            let output = std::sync::Arc::clone(&app.dof_output);
            std::thread::spawn(move || {
                loop {
                    let data = input.lock().unwrap().take();
                    if let Some(data) = data {
                        if let Ok(mut sketch) = bincode::deserialize::<Sketch>(&data) {
                            let dof = match sketch.compute_dof(false) {
                                Ok(r) => r.dof,
                                Err(_) => { continue; }
                            };
                            *output.lock().unwrap() = Some(dof);
                        }
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            });
        }
        app.compute_dof_async();
        app
    }
}

impl EditorApp {
    // Sketch coords -> screen coords
    pub fn to_screen(&self, p: vect2d) -> egui::Pos2 {
        egui::Pos2::new(
            p.x as f32 * self.scale + self.offset.x,
            -p.y as f32 * self.scale + self.offset.y, // y flipped
        )
    }

    // Screen coords -> sketch coords
    pub fn to_sketch(&self, p: egui::Pos2) -> vect2d {
        vect2d::new(
            ((p.x - self.offset.x) / self.scale) as f64,
            (-(p.y - self.offset.y) / self.scale) as f64,
        )
    }

    // Hit test: find nearest grabbable target within threshold
    /// Classical box-select: take two screen points (press origin
    /// and release point), build their axis-aligned bounding box in
    /// sketch coords, then add every entity inside or crossing that
    /// box to the selection. `additive` keeps the existing selection
    /// (shift-drag); otherwise the prior selection is replaced.
    pub fn apply_box_select(&mut self, a: egui::Pos2, b: egui::Pos2, additive: bool) {
        let a_s = self.to_sketch(a);
        let b_s = self.to_sketch(b);
        let min = vect2d::new(a_s.x.min(b_s.x), a_s.y.min(b_s.y));
        let max = vect2d::new(a_s.x.max(b_s.x), a_s.y.max(b_s.y));
        let in_rect = |p: vect2d| -> bool {
            p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y
        };
        let segment_crosses = |p1: vect2d, p2: vect2d| -> bool {
            if in_rect(p1) || in_rect(p2) { return true; }
            // Any of the four rect edges crossed by p1-p2?
            let corners = [
                vect2d::new(min.x, min.y),
                vect2d::new(max.x, min.y),
                vect2d::new(max.x, max.y),
                vect2d::new(min.x, max.y),
            ];
            let orient = |a: vect2d, b: vect2d, c: vect2d| -> f64 {
                (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
            };
            let segs_cross = |a: vect2d, b: vect2d, c: vect2d, d: vect2d| -> bool {
                let o1 = orient(a, b, c);
                let o2 = orient(a, b, d);
                let o3 = orient(c, d, a);
                let o4 = orient(c, d, b);
                (o1 * o2 < 0.0) && (o3 * o4 < 0.0)
            };
            for i in 0..4 {
                if segs_cross(p1, p2, corners[i], corners[(i + 1) % 4]) { return true; }
            }
            false
        };
        let circle_bbox_hits = |center: vect2d, r: f64| -> bool {
            // Cheap AABB/AABB overlap: circle bbox [c-r, c+r].
            let cmin = vect2d::new(center.x - r, center.y - r);
            let cmax = vect2d::new(center.x + r, center.y + r);
            cmin.x <= max.x && cmax.x >= min.x
                && cmin.y <= max.y && cmax.y >= min.y
        };

        if !additive { self.selection.clear(); }
        let add = |sel: &mut Vec<Selection>, s: Selection| {
            if !sel.contains(&s) { sel.push(s); }
        };
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            if in_rect(p.pos.value) { add(&mut self.selection, Selection::Point(r)); }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if segment_crosses(l.p1.value, l.p2.value) {
                add(&mut self.selection, Selection::Line(r));
            }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            // Bounding-box hit is a deliberate over-approximation for
            // a selection tool: a marquee that grazes the bbox but
            // misses the curve itself is rare in practice, and the
            // user-friendly answer when in doubt is "include it".
            let r_eff = if a.is_ellipse {
                a.radius.value.max(a.radius_b.value)
            } else {
                a.radius.value
            };
            if circle_bbox_hits(a.center.value, r_eff) {
                add(&mut self.selection, Selection::Arc(r));
            }
        }
    }

    /// Double-click chain selection: starting from `seed` (a Line or
    /// Arc), walk the graph of coincident endpoints and add every
    /// line / arc reachable via an unbranched chain. "Unbranched"
    /// means every endpoint class we follow has exactly one other
    /// entity attached; as soon as we hit a fork (three or more
    /// entity-endpoints at a class) or a dead end (zero others), the
    /// chain stops from that side. The seed itself is always added.
    pub fn select_chain(&mut self, seed: Selection) {
        use std::collections::{HashMap, HashSet, VecDeque};

        /// A geometric endpoint slot. `Point` is a bare sketch point
        /// which acts as a junction; unioning everything coincident
        /// to it keeps the chain logic independent of how the
        /// connection was recorded (direct LL* coincident vs via
        /// bare Point).
        #[derive(PartialEq, Eq, Hash, Clone, Copy)]
        enum Ep {
            Point(Ref<Point>),
            LineP1(Ref<Line>),
            LineP2(Ref<Line>),
            ArcStart(Ref<Arc>),
            ArcEnd(Ref<Arc>),
        }

        // ---- Build equivalence classes via union-find ----
        fn find(p: &mut HashMap<Ep, Ep>, e: Ep) -> Ep {
            let mut cur = e;
            while let Some(&parent) = p.get(&cur) {
                if parent == cur { break; }
                cur = parent;
            }
            // Path compression.
            let mut walker = e;
            while let Some(&parent) = p.get(&walker) {
                if parent == cur { break; }
                p.insert(walker, cur);
                walker = parent;
            }
            cur
        }
        fn ensure(p: &mut HashMap<Ep, Ep>, e: Ep) {
            p.entry(e).or_insert(e);
        }
        fn union(p: &mut HashMap<Ep, Ep>, a: Ep, b: Ep) {
            ensure(p, a); ensure(p, b);
            let ra = find(p, a);
            let rb = find(p, b);
            if ra != rb { p.insert(ra, rb); }
        }

        let mut parent: HashMap<Ep, Ep> = HashMap::new();

        // Seed every relevant endpoint slot so lone endpoints still
        // appear in the map (chain ends need empty classes too).
        // Construction lines / arcs are skipped -- they exist for
        // reference geometry and the user's mental chain is the
        // solid outline. They don't end up in classes or in the
        // walk's visitable set.
        for r in self.sketch.lines.refs() {
            if self.sketch.lines[r].construction { continue; }
            ensure(&mut parent, Ep::LineP1(r));
            ensure(&mut parent, Ep::LineP2(r));
        }
        for r in self.sketch.arcs.refs() {
            if self.sketch.arcs[r].closed { continue; }
            if self.sketch.arcs[r].construction { continue; }
            ensure(&mut parent, Ep::ArcStart(r));
            ensure(&mut parent, Ep::ArcEnd(r));
        }
        for r in self.sketch.points.refs() {
            if self.sketch.points[r].helper { continue; }
            ensure(&mut parent, Ep::Point(r));
        }
        // Skip construction-tied coincidents below via these
        // predicates so unions never touch a construction entity.
        let line_ok = |r: Ref<Line>| -> bool { !self.sketch.lines[r].construction };
        let arc_ok = |r: Ref<Arc>| -> bool { !self.sketch.arcs[r].construction };

        // Line-line coincidents.
        for c in &self.sketch.coincident_ll11 {
            if line_ok(c.a) && line_ok(c.b) {
                union(&mut parent, Ep::LineP1(c.a), Ep::LineP1(c.b));
            }
        }
        for c in &self.sketch.coincident_ll12 {
            if line_ok(c.a) && line_ok(c.b) {
                union(&mut parent, Ep::LineP1(c.a), Ep::LineP2(c.b));
            }
        }
        for c in &self.sketch.coincident_ll21 {
            if line_ok(c.a) && line_ok(c.b) {
                union(&mut parent, Ep::LineP2(c.a), Ep::LineP1(c.b));
            }
        }
        for c in &self.sketch.coincident_ll22 {
            if line_ok(c.a) && line_ok(c.b) {
                union(&mut parent, Ep::LineP2(c.a), Ep::LineP2(c.b));
            }
        }
        // Line endpoint <-> bare point.
        for c in &self.sketch.coincident_lp1 {
            if line_ok(c.line) && !self.sketch.points[c.point].helper {
                union(&mut parent, Ep::LineP1(c.line), Ep::Point(c.point));
            }
        }
        for c in &self.sketch.coincident_lp2 {
            if line_ok(c.line) && !self.sketch.points[c.point].helper {
                union(&mut parent, Ep::LineP2(c.line), Ep::Point(c.point));
            }
        }
        // Point-point coincidents keep a junction "one" node even if
        // the sketch author used two separate points.
        for c in &self.sketch.coincident_pp {
            if !self.sketch.points[c.a].helper && !self.sketch.points[c.b].helper {
                union(&mut parent, Ep::Point(c.a), Ep::Point(c.b));
            }
        }
        // Line endpoint <-> arc endpoint (direct).
        for c in &self.sketch.coincident_lp1_arc_start {
            if line_ok(c.line) && arc_ok(c.arc) {
                union(&mut parent, Ep::LineP1(c.line), Ep::ArcStart(c.arc));
            }
        }
        for c in &self.sketch.coincident_lp2_arc_start {
            if line_ok(c.line) && arc_ok(c.arc) {
                union(&mut parent, Ep::LineP2(c.line), Ep::ArcStart(c.arc));
            }
        }
        for c in &self.sketch.coincident_lp1_arc_end {
            if line_ok(c.line) && arc_ok(c.arc) {
                union(&mut parent, Ep::LineP1(c.line), Ep::ArcEnd(c.arc));
            }
        }
        for c in &self.sketch.coincident_lp2_arc_end {
            if line_ok(c.line) && arc_ok(c.arc) {
                union(&mut parent, Ep::LineP2(c.line), Ep::ArcEnd(c.arc));
            }
        }
        // Arc endpoint <-> bare point.
        for c in &self.sketch.coincident_arc_start {
            if arc_ok(c.arc) && !self.sketch.points[c.point].helper {
                union(&mut parent, Ep::ArcStart(c.arc), Ep::Point(c.point));
            }
        }
        for c in &self.sketch.coincident_arc_end {
            if arc_ok(c.arc) && !self.sketch.points[c.point].helper {
                union(&mut parent, Ep::ArcEnd(c.arc), Ep::Point(c.point));
            }
        }
        // Arc-arc endpoint unions (direct).
        for c in &self.sketch.coincident_arc_start_start {
            if arc_ok(c.a) && arc_ok(c.b) {
                union(&mut parent, Ep::ArcStart(c.a), Ep::ArcStart(c.b));
            }
        }
        for c in &self.sketch.coincident_arc_start_end {
            if arc_ok(c.a) && arc_ok(c.b) {
                union(&mut parent, Ep::ArcStart(c.a), Ep::ArcEnd(c.b));
            }
        }
        for c in &self.sketch.coincident_arc_end_start {
            if arc_ok(c.a) && arc_ok(c.b) {
                union(&mut parent, Ep::ArcEnd(c.a), Ep::ArcStart(c.b));
            }
        }
        for c in &self.sketch.coincident_arc_end_end {
            if arc_ok(c.a) && arc_ok(c.b) {
                union(&mut parent, Ep::ArcEnd(c.a), Ep::ArcEnd(c.b));
            }
        }

        // ---- Chain walk ----
        // An entity key distinguishes "self" from "other" when
        // counting neighbours at an endpoint class. ArcCenter and
        // Points aren't entities in the chain sense -- they're
        // junctions or fixtures.
        #[derive(PartialEq, Eq, Hash, Clone, Copy)]
        enum EntKey { Line(Ref<Line>), Arc(Ref<Arc>) }

        let all_eps: Vec<Ep> = parent.keys().copied().collect();
        // Resolve every ep to its root and collect class members.
        let mut class: HashMap<Ep, Vec<Ep>> = HashMap::new();
        for ep in all_eps {
            let r = find(&mut parent, ep);
            class.entry(r).or_default().push(ep);
        }
        // Per-endpoint neighbour: given a concrete (entity, end) pair,
        // return the unique OTHER entity sharing that endpoint's
        // class -- or None if the class is a branch or dead end.
        let neighbour = |ep: Ep| -> Option<(EntKey, Ep)> {
            let r = parent.get(&ep).copied()?;
            let members = class.get(&r)?;
            let self_key = match ep {
                Ep::LineP1(i) | Ep::LineP2(i) => EntKey::Line(i),
                Ep::ArcStart(i) | Ep::ArcEnd(i) => EntKey::Arc(i),
                Ep::Point(_) => return None,
            };
            let mut others: Vec<(EntKey, Ep)> = Vec::new();
            for m in members {
                let k = match *m {
                    Ep::LineP1(i) | Ep::LineP2(i) => EntKey::Line(i),
                    Ep::ArcStart(i) | Ep::ArcEnd(i) => EntKey::Arc(i),
                    Ep::Point(_) => continue, // junctions don't count as entities
                };
                if k == self_key { continue; }
                if !others.iter().any(|(ek, _)| *ek == k) {
                    others.push((k, *m));
                }
            }
            if others.len() == 1 { Some(others[0]) } else { None }
        };

        let seed_key = match seed {
            Selection::Line(r) => EntKey::Line(r),
            Selection::Arc(r) => {
                if self.sketch.arcs[r].closed {
                    // Closed arcs aren't chainable; just select this one.
                    if !self.selection.contains(&seed) { self.selection.push(seed); }
                    return;
                }
                EntKey::Arc(r)
            }
            _ => return,
        };

        let endpoints_of = |k: EntKey| -> [Ep; 2] {
            match k {
                EntKey::Line(i) => [Ep::LineP1(i), Ep::LineP2(i)],
                EntKey::Arc(i) => [Ep::ArcStart(i), Ep::ArcEnd(i)],
            }
        };
        // EntKey carries the raw slot index because the BFS keys on it;
        // resolving back to a Selection goes through the arena, so a stale
        // index yields nothing instead of a ref to whatever now occupies it.
        let key_to_selection = |k: EntKey| -> Selection {
            match k {
                EntKey::Line(r) => Selection::Line(r),
                EntKey::Arc(r) => Selection::Arc(r),
            }
        };

        let mut visited: HashSet<EntKey> = HashSet::new();
        let mut queue: VecDeque<EntKey> = VecDeque::new();
        queue.push_back(seed_key);
        while let Some(cur) = queue.pop_front() {
            if !visited.insert(cur) { continue; }
            let sel = key_to_selection(cur);
            if !self.selection.contains(&sel) { self.selection.push(sel); }
            for ep in endpoints_of(cur) {
                if let Some((nb_key, _nb_ep)) = neighbour(ep)
                    && !visited.contains(&nb_key)
                {
                    queue.push_back(nb_key);
                }
            }
        }
    }

    fn hit_test(&self, sketch_pos: vect2d, threshold: f64) -> Option<GrabTarget> {
        let mut best: Option<(f64, GrabTarget)> = None;

        let mut check = |dist: f64, target: GrabTarget| {
            if dist < threshold
                && (best.is_none() || dist < best.unwrap().0) {
                    best = Some((dist, target));
                }
        };

        // Points (skip helpers)
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d, GrabTarget::Point(r));
        }

        // Line endpoints (priority over line body)
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];

            let d1 = ((l.p1.value.x - sketch_pos.x).powi(2)
                    + (l.p1.value.y - sketch_pos.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - sketch_pos.x).powi(2)
                    + (l.p2.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d1, GrabTarget::LineP1(r));
            check(d2, GrabTarget::LineP2(r));
        }

        // Arc centers and endpoints
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let dc = ((a.center.value.x - sketch_pos.x).powi(2)
                    + (a.center.value.y - sketch_pos.y).powi(2)).sqrt();
            check(dc, GrabTarget::ArcCenter(r));

            if !a.closed {
                let sp = arc_start_pos(a);
                let ep = arc_end_pos(a);
                let ds = ((sp.x - sketch_pos.x).powi(2) + (sp.y - sketch_pos.y).powi(2)).sqrt();
                let de = ((ep.x - sketch_pos.x).powi(2) + (ep.y - sketch_pos.y).powi(2)).sqrt();
                check(ds, GrabTarget::ArcStart(r));
                check(de, GrabTarget::ArcEnd(r));
            }
        }

        // If an endpoint was found, return it (priority over body drag)
        if best.is_some() { return best.map(|(_, t)| t); }

        // Line bodies (lower priority than endpoints)
        let mut body_best: Option<(f64, GrabTarget)> = None;
        let mut check_body = |dist: f64, target: GrabTarget| {
            if dist < threshold
                && (body_best.is_none() || dist < body_best.unwrap().0) {
                    body_best = Some((dist, target));
                }
        };
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            check_body(d, GrabTarget::LineDrag(r));
        }

        // Arc/circle curves (lower priority than endpoints)
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let (d, _) = point_to_arc_dist(sketch_pos, a);
            check_body(d, GrabTarget::ArcDrag(r));
        }

        body_best.map(|(_, t)| t)
    }

    // Find selection target (entity near mouse)
    // Priority: Constraints > Points > Endpoints > Lines/Arcs
    fn hit_test_selection(&self, sketch_pos: vect2d, threshold: f64) -> Option<Selection> {
        // Constraint markers first (screen-space, pick closest)
        let screen_pos = self.to_screen(sketch_pos);
        let mut best_constraint: Option<(f32, ConstraintId)> = None;
        for marker in &self.constraint_markers {
            let dx = screen_pos.x - marker.pos.x;
            let dy = screen_pos.y - marker.pos.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d < 10.0
                && (best_constraint.is_none() || d < best_constraint.unwrap().0) {
                    best_constraint = Some((d, marker.id));
                }
        }
        if let Some((_, id)) = best_constraint {
            return Some(Selection::Constraint(id));
        }

        // Then standalone points (skip helpers, skip if hidden)
        if self.show_points {
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            if d < threshold { return Some(Selection::Point(r)); }
        }
        }

        // Then line endpoints (pseudo-points)
        let mut best_ep: Option<(f64, Selection)> = None;
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];

            let d1 = ((l.p1.value.x - sketch_pos.x).powi(2)
                    + (l.p1.value.y - sketch_pos.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - sketch_pos.x).powi(2)
                    + (l.p2.value.y - sketch_pos.y).powi(2)).sqrt();
            if d1 < threshold
                && (best_ep.is_none() || d1 < best_ep.unwrap().0) {
                    best_ep = Some((d1, Selection::LineP1(r)));
                }
            if d2 < threshold
                && (best_ep.is_none() || d2 < best_ep.unwrap().0) {
                    best_ep = Some((d2, Selection::LineP2(r)));
                }
        }
        // Arc centers and endpoints (same priority as line endpoints)
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let dc = ((a.center.value.x - sketch_pos.x).powi(2)
                    + (a.center.value.y - sketch_pos.y).powi(2)).sqrt();
            if dc < threshold
                && (best_ep.is_none() || dc < best_ep.unwrap().0) {
                    best_ep = Some((dc, Selection::ArcCenter(r)));
                }
            if !a.closed {
                let sp = arc_start_pos(a);
                let ep = arc_end_pos(a);
                let ds = ((sp.x - sketch_pos.x).powi(2) + (sp.y - sketch_pos.y).powi(2)).sqrt();
                let de = ((ep.x - sketch_pos.x).powi(2) + (ep.y - sketch_pos.y).powi(2)).sqrt();
                if ds < threshold
                    && (best_ep.is_none() || ds < best_ep.unwrap().0) {
                        best_ep = Some((ds, Selection::ArcStart(r)));
                    }
                if de < threshold
                    && (best_ep.is_none() || de < best_ep.unwrap().0) {
                        best_ep = Some((de, Selection::ArcEnd(r)));
                    }
            }
        }

        if let Some((_, sel)) = best_ep { return Some(sel); }

        // Dimension annotations (higher priority than line/arc bodies)
        // Check both the text segment AND the dimension arrow line
        if self.show_dimensions {
        let screen_pos = self.to_screen(sketch_pos);
        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
            // Text segment
            let (ts, te) = self.dim_text_segment(dim);
            let dt = Self::screen_point_to_segment_dist(screen_pos, ts, te);
            // Arrow line segment (for angle dimensions, use the arc)
            let da = if matches!(dim.kind, DimensionKind::ArcRadius(_) | DimensionKind::ArcRadiusB(_) | DimensionKind::ArcSweep(_) | DimensionKind::ArcRotation(_) | DimensionKind::Angle(..) | DimensionKind::LineAngle(_) | DimensionKind::HDistance(..) | DimensionKind::VDistance(..) | DimensionKind::ConcentricDistance(..)) {
                dt // for radius/angle/concentric, text check is enough
            } else {
                let (p1, p2) = self.dim_endpoints(&dim.kind);
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                let nx = -dy / len;
                let ny = dx / len;
                let off = dim.offset.y;
                let q1 = vect2d::new(p1.x + nx * off, p1.y + ny * off);
                let q2 = vect2d::new(p2.x + nx * off, p2.y + ny * off);
                let sq1 = self.to_screen(q1);
                let sq2 = self.to_screen(q2);
                Self::screen_point_to_segment_dist(screen_pos, sq1, sq2)
            };
            if dt < 15.0 || da < 8.0 {
                return Some(Selection::Dimension(i));
            }
        }
        }

        // Then lines (distance to segment)
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            if d < threshold { return Some(Selection::Line(r)); }
        }

        // Then arc/circle curves (find closest, not first)
        let mut best_arc: Option<(f64, arael::refs::Ref<arael_sketch_solver::Arc>)> = None;
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let (d, _) = point_to_arc_dist(sketch_pos, a);
            if d < threshold
                && (best_arc.is_none() || d < best_arc.unwrap().0) {
                    best_arc = Some((d, r));
                }
        }
        if let Some((_, r)) = best_arc { return Some(Selection::Arc(r)); }

        None
    }

    /// Build the drag helper's Param. In soft-drag mode (default), the
    /// helper is an optimizable Point so the Point drift residual pulls
    /// it toward `pos` but hard constraints can overrule; the user sees
    /// the sketch's relaxed state. `--drag-raw` returns a fixed Param
    /// that pins the helper exactly at `pos` (legacy hard-pin drag).
    fn drag_helper_param(&self, pos: vect2d) -> Param<vect2d> {
        if self.drag_raw { Param::fixed(pos) } else { Param::new(pos) }
    }

    /// Test whether adding a perpendicular constraint between `a`
    /// and `b` would reduce DOF. Used by drag preview to suppress
    /// auto-perpendicular proposals that would be rejected on release.
    ///
    /// Cheap variant: push one entry, recompute DOF, pop it back.
    /// No bincode round-trip and no solve. Still costs one full SVD
    /// on the Jacobian which can be tens of milliseconds on big
    /// sketches -- too much at 60 Hz during a drag. So: once the
    /// check fails for a given (a, b) pair, cache that failure for
    /// 1 second before re-testing. Successes are never cached
    /// (cheap path: if perp is structurally active, we want the
    /// check to re-confirm promptly if the user maneuvers the
    /// geometry in a way that could change applicability).
    /// True when making `line` horizontal (if `horizontal`) or vertical
    /// would reduce sketch DOF -- i.e. the axis is not already forced
    /// by the existing constraint system. Flag-based short-circuit
    /// first (same orientation flag already set = no reduction), then
    /// fall back to a single rank check. Used by auto-H/V snap gating
    /// at drag-start so a line that's already axis-locked (directly or
    /// via implicit chains) doesn't flash a redundant hint.
    fn hv_would_reduce_dof(&mut self, line: Ref<Line>, horizontal: bool) -> bool {
        let l_c = &self.sketch.lines[line].constraints;
        if horizontal && l_c.horizontal { return false; }
        if !horizontal && l_c.vertical { return false; }
        let Ok(old_dof) = self.sketch.get_mut().dof() else { return true; };
        let saved_cached = self.sketch.cached_dof;
        if horizontal {
            let dx = self.sketch.lines[line].p2.value.x - self.sketch.lines[line].p1.value.x;
            self.sketch.get_mut().lines[line].constraints.h_dir_sign = if dx >= 0.0 { 1.0 } else { -1.0 };
            self.sketch.get_mut().lines[line].constraints.horizontal = true;
        } else {
            let dy = self.sketch.lines[line].p2.value.y - self.sketch.lines[line].p1.value.y;
            self.sketch.get_mut().lines[line].constraints.v_dir_sign = if dy >= 0.0 { 1.0 } else { -1.0 };
            self.sketch.get_mut().lines[line].constraints.vertical = true;
        }
        self.sketch.get_mut().cached_dof = None;
        let new_dof = self.sketch.get_mut().dof().unwrap_or(old_dof);
        if horizontal {
            self.sketch.get_mut().lines[line].constraints.horizontal = false;
        } else {
            self.sketch.get_mut().lines[line].constraints.vertical = false;
        }
        self.sketch.get_mut().cached_dof = saved_cached;
        new_dof < old_dof
    }

    /// True when adding Collinear(a, b) would reduce sketch DOF.
    /// Same pattern as perp_would_reduce_dof: temporarily push the
    /// constraint, recompute DOF, compare, restore. Used by auto-
    /// collinear hint gating at drag-start.
    fn collinear_would_reduce_dof(&mut self, a: Ref<Line>, b: Ref<Line>) -> bool {
        if self.has_collinear_conflict(a, b) { return false; }
        let Ok(old_dof) = self.sketch.get_mut().dof() else { return true; };
        let saved_cached = self.sketch.cached_dof;
        self.sketch.get_mut().collinear.push(Collinear {
            a, b, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
        });
        self.sketch.get_mut().cached_dof = None;
        let new_dof = self.sketch.get_mut().dof().unwrap_or(old_dof);
        self.sketch.get_mut().collinear.pop();
        self.sketch.get_mut().cached_dof = saved_cached;
        new_dof < old_dof
    }

    fn perp_would_reduce_dof(&mut self, a: Ref<Line>, b: Ref<Line>) -> bool {
        let key = (a.index(), b.index());
        if let Some((ka, kb, at)) = self.drag_perp_fail_cache
            && (ka, kb) == key
            && at.elapsed() < std::time::Duration::from_secs(1) {
            return false;
        }
        // Fast path: if one line is horizontal and the other vertical
        // (or vice versa), perpendicularity is already implied by the
        // H/V flags -- no need to run a full jacobian-rank check.
        // During drag the geometry is briefly off-axis so the rank
        // query can wrongly report a DOF reduction; the flags are
        // unambiguous. Suppresses the spurious perp hint on rect
        // corners and avoids the tens-of-ms DOF recompute per frame.
        let la_c = &self.sketch.lines[a].constraints;
        let lb_c = &self.sketch.lines[b].constraints;
        if (la_c.horizontal && lb_c.vertical) || (la_c.vertical && lb_c.horizontal) {
            self.drag_perp_fail_cache = Some((key.0, key.1, web_time::Instant::now()));
            return false;
        }
        // Pre-drag snapshot: pairs that were already perpendicular
        // (directly or via an implicit chain such as equal-length
        // plus collinear neighbours) when the drag started. Stored
        // by start_drag. Covers the general case beyond the H/V flag
        // shortcut above.
        for &(ai, bi) in &self.drag_perp_already {
            if (ai == key.0 && bi == key.1) || (ai == key.1 && bi == key.0) {
                return false;
            }
        }
        let old_dof = match self.sketch.get_mut().dof() { Ok(d) => d, Err(_) => return false };
        let saved_cached = self.sketch.cached_dof;

        let la = &self.sketch.lines[a];
        let lb = &self.sketch.lines[b];
        let dx1 = la.p2.value.x - la.p1.value.x;
        let dy1 = la.p2.value.y - la.p1.value.y;
        let dx2 = lb.p2.value.x - lb.p1.value.x;
        let dy2 = lb.p2.value.y - lb.p1.value.y;
        let cross = dx1 * dy2 - dy1 * dx2;
        let dir_sign = if cross >= 0.0 { 1.0 } else { -1.0 };
        self.sketch.get_mut().perpendicular.push(Perpendicular {
            a, b, dir_sign, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
        });
        self.sketch.get_mut().cached_dof = None;
        let new_dof = self.sketch.get_mut().dof().unwrap_or(old_dof);
        self.sketch.get_mut().perpendicular.pop();
        self.sketch.get_mut().cached_dof = saved_cached;

        let result = new_dof < old_dof;
        if !result {
            self.drag_perp_fail_cache = Some((key.0, key.1, web_time::Instant::now()));
        } else {
            self.drag_perp_fail_cache = None;
        }
        result
    }

    /// Add a drag helper point using the mode-appropriate Param.
    ///
    /// Soft drag uses an optimizable helper point with `drag_pull > 0`
    /// so the dedicated attractor residual tracks the cursor while hard
    /// constraints stay satisfied. Raw drag uses `Param::fixed`, which
    /// reproduces the old hard-pin behaviour.
    fn add_drag_helper(&mut self, pos: vect2d) -> Ref<Point> {
        if self.drag_raw {
            self.sketch.get_mut().add_point_fixed(pos)
        } else {
            let r = self.sketch.get_mut().add_helper_point(pos);
            self.sketch.get_mut().points[r].drag_pull = DRAG_PULL_WEIGHT;
            r
        }
    }

    // Start dragging: create a temporary drag helper point and coincident constraint
    fn start_drag(&mut self, target: GrabTarget, mouse_pos: vect2d) {
        self.show_hints = false;
        // Save pre-drag state before adding drag apparatus
        self.drag_saved_cost = self.last_cost;
        self.drag_saved_snapshot = bincode::serialize(&self.sketch).ok();

        // Pre-compute pairs that are already perpendicular now, so the
        // per-frame auto-perp hint can short-circuit without a rank-based
        // check on off-axis intermediate geometry. See drag_perp_already.
        self.drag_perp_already.clear();
        self.drag_collinear_already.clear();
        self.drag_line_locked_h = false;
        self.drag_line_locked_v = false;
        self.drag_line_endpoint_connected = false;
        if let GrabTarget::LineP1(line) | GrabTarget::LineP2(line) = target {
            let is_p1 = matches!(target, GrabTarget::LineP1(_));
            if let Some(host) = self.find_anchor_host_line_for_drag(line, is_p1) {
                if !self.perp_would_reduce_dof(line, host) {
                    self.drag_perp_already.push((line.index(), host.index()));
                }
                if !self.collinear_would_reduce_dof(line, host) {
                    self.drag_collinear_already.push((line.index(), host.index()));
                }
            }
            // Auto-H/V suppression: if the line's orientation is
            // already pinned (directly or implicitly), skip the hint.
            self.drag_line_locked_h = !self.hv_would_reduce_dof(line, true);
            self.drag_line_locked_v = !self.hv_would_reduce_dof(line, false);
            // Connectedness check must happen BEFORE start_drag pushes
            // its drag-helper coincident (otherwise the helper's own
            // coincident registers as a connection).
            self.drag_line_endpoint_connected = self.line_endpoint_has_connection(line, is_p1);
        }

        // Create a drag helper point at mouse position
        let drag_pt = self.add_drag_helper(mouse_pos);
        self.drag_point = Some(drag_pt);

        // Add coincident constraint between drag point and the grabbed target
        match target {
            GrabTarget::Point(r) => {
                self.sketch.get_mut().coincident_pp.push(CoincidentPP {
                    a: drag_pt, b: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::LineP1(r) => {
                self.sketch.get_mut().coincident_lp1.push(CoincidentLP1 {
                    line: r, point: drag_pt, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::LineP2(r) => {
                self.sketch.get_mut().coincident_lp2.push(CoincidentLP2 {
                    line: r, point: drag_pt, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcCenter(r) => {
                self.sketch.get_mut().coincident_arc_center.push(CoincidentArcCenter {
                    point: drag_pt, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcStart(r) => {
                self.sketch.get_mut().coincident_arc_start.push(CoincidentArcStart {
                    point: drag_pt, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcEnd(r) => {
                self.sketch.get_mut().coincident_arc_end.push(CoincidentArcEnd {
                    point: drag_pt, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::LineDrag(r) => {
                let (p1v, p2v) = {
                    let l = &self.sketch.lines[r];
                    (l.p1.value, l.p2.value)
                };
                self.drag_offset = vect2d::new(p1v.x - mouse_pos.x, p1v.y - mouse_pos.y);
                self.drag_offset2 = vect2d::new(p2v.x - mouse_pos.x, p2v.y - mouse_pos.y);
                // First drag point at p1
                let pos1 = self.drag_helper_param(p1v);
                self.sketch.get_mut().points[drag_pt].pos = pos1;
                self.sketch.get_mut().coincident_lp1.push(CoincidentLP1 {
                    line: r, point: drag_pt, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
                // Second drag point at p2
                let drag_pt2 = self.add_drag_helper(p2v);
                self.drag_point2 = Some(drag_pt2);
                self.sketch.get_mut().coincident_lp2.push(CoincidentLP2 {
                    line: r, point: drag_pt2, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcDrag(r) => {
                let centre = self.sketch.arcs[r].center.value;
                self.drag_offset = vect2d::new(centre.x - mouse_pos.x, centre.y - mouse_pos.y);
                // Drag point at center
                let pos_c = self.drag_helper_param(centre);
                self.sketch.get_mut().points[drag_pt].pos = pos_c;
                self.sketch.get_mut().coincident_arc_center.push(CoincidentArcCenter {
                    point: drag_pt, arc: r, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
                // Lock radius and sweep to prevent shape change
                let a = &self.sketch.arcs[r];
                self.drag_saved_arc_locks = Some(SavedArcLocks {
                    had_radius: a.constraints.has_target_radius,
                    old_radius: a.constraints.target_radius,
                    had_radius_b: a.constraints.has_target_radius_b,
                    old_radius_b: a.constraints.target_radius_b,
                    rotation_optimize: a.rotation.optimize,
                    had_sweep: a.constraints.has_target_sweep,
                    old_sweep: a.constraints.target_sweep,
                    old_sweep_sign: a.constraints.sweep_sign,
                    start_optimize: a.start_angle.optimize,
                    end_optimize: a.end_angle.optimize,
                });
                let a = &mut self.sketch.get_mut().arcs[r];
                a.constraints.has_target_radius = true;
                a.constraints.target_radius = a.radius.value;
                if a.is_ellipse {
                    a.constraints.has_target_radius_b = true;
                    a.constraints.target_radius_b = a.radius_b.value;
                    a.rotation.optimize = false;
                }
                a.constraints.has_target_sweep = true;
                // target_sweep is the positive sweep magnitude; sweep_sign
                // carries the direction. Signed delta here would mismatch
                // sweep_sign on CW arcs and force radius to 0 to zero the
                // residual.
                a.constraints.target_sweep = (a.end_angle.value - a.start_angle.value).abs();
                a.constraints.sweep_sign = if a.ccw { 1.0 } else { -1.0 };
                a.start_angle.optimize = false;
                a.end_angle.optimize = false;
            }
        }
        self.grab = Some(target);
        // Auto-anchor hack stabilises chain-style drags; rolled back
        // at end_drag (and from clone snapshots in update_drag).
        // See Sketch::add_drag_auto_anchors for the full rationale.
        self.drag_auto_anchors = Some(self.sketch.get_mut().add_drag_auto_anchors());
    }

    // Update drag position and re-solve.
    // Track the best (lowest cost) clean state seen during the drag.
    fn update_drag(&mut self, mouse_pos: vect2d, hit_threshold: f64) {
        if let Some(drag_pt) = self.drag_point {
            let is_body_drag = matches!(self.grab, Some(GrabTarget::LineDrag(_) | GrabTarget::ArcDrag(_)));
            // Live snap preview for point-like drags: if the cursor is
            // within snap range of another entity (excluding the dragged
            // entity itself), pull the drag point to the snap location so
            // the user sees where the constraint will land on release --
            // same behavior as during line creation. Body drags
            // (LineDrag/ArcDrag) are excluded.
            let (exclude_line, exclude_arc) = match self.grab {
                Some(GrabTarget::LineP1(r)) | Some(GrabTarget::LineP2(r)) => (Some(r), None),
                Some(GrabTarget::ArcCenter(r)) | Some(GrabTarget::ArcStart(r)) | Some(GrabTarget::ArcEnd(r)) => (None, Some(r)),
                _ => (None, None),
            };
            // Where the dragged entity actually sits after the last
            // solve. Used to gate snap hints -- a snap is only real if
            // the grab can physically reach the target. Without this
            // gate a DOF=0 sketch would flash snap previews as the
            // cursor roams, and a heavily-lagged drag (ball-clamp)
            // would propose unreachable snaps. The probe itself stays
            // at mouse_pos so snaps still *detach* when the user
            // moves the cursor away from the target.
            let grab_pos = match self.grab {
                Some(GrabTarget::Point(r)) => Some(self.sketch.points[r].pos.value),
                Some(GrabTarget::LineP1(r)) => Some(self.sketch.lines[r].p1.value),
                Some(GrabTarget::LineP2(r)) => Some(self.sketch.lines[r].p2.value),
                Some(GrabTarget::ArcCenter(r)) => Some(self.sketch.arcs[r].center.value),
                Some(GrabTarget::ArcStart(r)) => Some(arael_sketch_backend::geometry::arc_start_pos(&self.sketch.arcs[r])),
                Some(GrabTarget::ArcEnd(r)) => Some(arael_sketch_backend::geometry::arc_end_pos(&self.sketch.arcs[r])),
                _ => None,
            };
            let reach = |target: vect2d| -> bool {
                match grab_pos {
                    Some(gp) => {
                        let dx = target.x - gp.x;
                        let dy = target.y - gp.y;
                        (dx * dx + dy * dy).sqrt() < hit_threshold
                    }
                    None => true,
                }
            };
            let (effective_pos, snap_preview) = match self.grab {
                Some(grab @ (GrabTarget::Point(_)
                    | GrabTarget::LineP1(_) | GrabTarget::LineP2(_)
                    | GrabTarget::ArcCenter(_) | GrabTarget::ArcStart(_) | GrabTarget::ArcEnd(_))) => {
                    // Filter in-loop so candidates already attached to the
                    // dragged entity don't hide second-best unattached ones.
                    // This matters when the dragged endpoint is coincident
                    // with a twin: the twin sits at the cursor (dist ~= 0)
                    // and without in-loop filtering would shadow any real
                    // new target within threshold.
                    match self.find_snap_target_filter(
                        mouse_pos, hit_threshold, exclude_line, exclude_arc,
                        |t| !self.has_existing_snap_attachment(grab, *t),
                    ) {
                        Some((p, t)) if reach(p) => (p, Some((p, t))),
                        _ => (mouse_pos, None),
                    }
                }
                _ => (mouse_pos, None),
            };
            self.drag_snap_preview = snap_preview;

            // Auto-perpendicular. Allowed alongside an end-line-body
            // snap (both project to lines, so the unique intersection
            // determines the position and BOTH constraints fire on
            // release). Suppressed for point-like / arc-like snaps that
            // already pin the position.
            let mut effective_pos = effective_pos;
            self.drag_perp_snap = None;
            let perp_eligible = match snap_preview {
                None => true,
                Some((_, SnapTarget::Line(_))) => true,
                _ => false,
            };
            if perp_eligible {
                if let Some(grab) = self.grab {
                    if let GrabTarget::LineP1(line) | GrabTarget::LineP2(line) = grab {
                        let is_p1 = matches!(grab, GrabTarget::LineP1(_));
                        if let Some(host) = self.find_anchor_host_line_for_drag(line, is_p1) {
                            if !self.has_perp_conflict(line, host) {
                                let opp = if is_p1 {
                                    self.sketch.lines[line].p2.value
                                } else {
                                    self.sketch.lines[line].p1.value
                                };
                                let hl = &self.sketch.lines[host];
                                if let Some(p) = self.try_perp_snap(
                                    opp, hl.p1.value, hl.p2.value, mouse_pos, PERP_SNAP_PX,
                                ).filter(|foot| reach(*foot)) {
                                    // Only propose the perpendicular hint if
                                    // applying it would actually reduce DOF.
                                    // If the candidate's row is already in the
                                    // row-span (e.g. parallel already implies
                                    // perp via the rest of the sketch) the
                                    // release would just reject it, so
                                    // suppress the preview.
                                    let hl_p1 = hl.p1.value;
                                    let hl_p2 = hl.p2.value;
                                    if self.perp_would_reduce_dof(line, host) {
                                        self.drag_perp_snap = Some((host, opp));
                                        if let Some((_, SnapTarget::Line(other))) = snap_preview {
                                            let hdx = hl_p2.x - hl_p1.x;
                                            let hdy = hl_p2.y - hl_p1.y;
                                            let ol = &self.sketch.lines[other];
                                            let odx = ol.p2.value.x - ol.p1.value.x;
                                            let ody = ol.p2.value.y - ol.p1.value.y;
                                            let cross = (-hdy) * ody - hdx * odx;
                                            if cross.abs() >= 1e-9 {
                                                let perp_p2 = vect2d::new(opp.x - hdy, opp.y + hdx);
                                                effective_pos = arael_sketch_backend::geometry::line_line_intersection(
                                                    opp, perp_p2, ol.p1.value, ol.p2.value);
                                            }
                                        } else {
                                            effective_pos = p;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Auto-H/V hint for line-endpoint drags. Fires only when
            // no snap or perp claimed the position and the line did
            // not already carry an H or V constraint -- mirroring the
            // line-creation auto-H/V. The snap anchors the opposite
            // endpoint (the fixed end) and pulls the dragged endpoint
            // onto the nearest axis through it, so the user can turn
            // a free-angle line into an axis-aligned one by the same
            // motion that would otherwise drop it free-hand.
            self.drag_hv_hint = None;
            self.drag_collinear_hint = None;
            if !self.snap_disabled
                && self.drag_perp_snap.is_none()
                && snap_preview.is_none()
                && !self.drag_line_endpoint_connected
            {
                if let Some(grab) = self.grab {
                    if let GrabTarget::LineP1(line) | GrabTarget::LineP2(line) = grab {
                        let is_p1 = matches!(grab, GrabTarget::LineP1(_));
                        let anchor = if is_p1 {
                            self.sketch.lines[line].p2.value
                        } else {
                            self.sketch.lines[line].p1.value
                        };
                        // Auto-collinear takes priority over H/V: it's a
                        // stronger structural constraint (same infinite
                        // line as host). Only fire when the host isn't
                        // already structurally collinear with the line
                        // and the dragged-line direction points along
                        // the host, not across it.
                        if let Some((host, foot)) = self.find_best_collinear_host_at(
                            anchor, effective_pos, PERP_SNAP_PX, Some(line),
                        ) {
                            let already = self.drag_collinear_already.iter()
                                .any(|&(la, lb)| la == line.index() && lb == host.index());
                            if !already && !self.has_collinear_conflict(line, host) {
                                effective_pos = foot;
                                self.drag_collinear_hint = Some((line, host));
                            }
                        }
                        // Fall through to H/V only if collinear didn't fire.
                        if self.drag_collinear_hint.is_none()
                            && let Some((horizontal, snapped)) = crate::app_update::hv_snap_from(
                            anchor, effective_pos, self.scale, PERP_SNAP_PX,
                        ) {
                            let already_locked = if horizontal {
                                self.drag_line_locked_h
                            } else {
                                self.drag_line_locked_v
                            };
                            if !already_locked {
                                effective_pos = snapped;
                                self.drag_hv_hint = Some((line, horizontal));
                            }
                        }
                    }
                }
            }

            // Ball-clamp each cursor to a radius of DRAG_MAX_LAG_PX
            // screen pixels around the helper's last-solved feasible
            // position. The solver never sees a cursor further out
            // than that, so on a locked sketch where the helper cannot
            // follow, the drag attractor residual stays bounded and
            // cannot deform the sketch. When the sketch has slack the
            // helper tracks the cursor exactly (delta < R → clamp is a
            // no-op) so free-direction drags feel unchanged.
            let r_world = (DRAG_MAX_LAG_PX / self.scale.max(1e-6)) as f64;
            let clamp = |target: vect2d, center: vect2d| -> vect2d {
                let dx = target.x - center.x;
                let dy = target.y - center.y;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= r_world || dist < 1e-12 {
                    target
                } else {
                    vect2d::new(center.x + dx * r_world / dist,
                                center.y + dy * r_world / dist)
                }
            };
            if is_body_drag {
                let ref1 = self.sketch.points[drag_pt].pos.value;
                let pos1 = clamp(vect2d::new(mouse_pos.x + self.drag_offset.x,
                                              mouse_pos.y + self.drag_offset.y), ref1);
                // Moving the drag helper is a value change: the entities,
                // constraints and dimensions all stand still for the whole
                // drag, so the derived state stays valid and is not rebuilt.
                let p1 = self.drag_helper_param(pos1);
                self.sketch.mutate_values(|s| s.points[drag_pt].pos = p1);
                if let Some(drag_pt2) = self.drag_point2 {
                    let ref2 = self.sketch.points[drag_pt2].pos.value;
                    let pos2 = clamp(vect2d::new(mouse_pos.x + self.drag_offset2.x,
                                                  mouse_pos.y + self.drag_offset2.y), ref2);
                    let p2 = self.drag_helper_param(pos2);
                    self.sketch.mutate_values(|s| s.points[drag_pt2].pos = p2);
                }
            } else {
                let ref_pos = self.sketch.points[drag_pt].pos.value;
                let pos = clamp(effective_pos, ref_pos);
                let p = self.drag_helper_param(pos);
                self.sketch.mutate_values(|s| s.points[drag_pt].pos = p);
            }
            let result = self.sketch.solve();
            self.last_cost = result.end_cost;

            // If cost is good, save a clean snapshot (without drag apparatus)
            if self.last_cost < self.drag_saved_cost + 1e-3
                && let Ok(snap) = bincode::serialize(&self.sketch)
                    && let Ok(mut clean) = bincode::deserialize::<Sketch>(&snap) {
                        // Auto-anchors are present in the live sketch
                        // (and therefore in the serialized clone).
                        // Roll them back from the clone first so the
                        // drag-apparatus pop()s below hit the right
                        // vec entries.
                        if let Some(state) = self.drag_auto_anchors.clone() {
                            clean.remove_drag_auto_anchors(state);
                        }
                        // Remove drag constraint from clone
                        match self.grab {
                            Some(GrabTarget::Point(_)) => { clean.coincident_pp.pop(); }
                            Some(GrabTarget::LineP1(_)) => { clean.coincident_lp1.pop(); }
                            Some(GrabTarget::LineP2(_)) => { clean.coincident_lp2.pop(); }
                            Some(GrabTarget::ArcCenter(_)) => { clean.coincident_arc_center.pop(); }
                            Some(GrabTarget::ArcStart(_)) => { clean.coincident_arc_start.pop(); }
                            Some(GrabTarget::ArcEnd(_)) => { clean.coincident_arc_end.pop(); }
                            Some(GrabTarget::LineDrag(_)) => {
                                clean.coincident_lp1.pop();
                                clean.coincident_lp2.pop();
                                if let Some(dp2) = self.drag_point2 { clean.points.remove(dp2); }
                            }
                            Some(GrabTarget::ArcDrag(r)) => {
                                clean.coincident_arc_center.pop();
                                if let Some(ref saved) = self.drag_saved_arc_locks {
                                    let a = &mut clean.arcs[r];
                                    a.constraints.has_target_radius = saved.had_radius;
                                    a.constraints.target_radius = saved.old_radius;
                                    a.constraints.has_target_radius_b = saved.had_radius_b;
                                    a.constraints.target_radius_b = saved.old_radius_b;
                                    a.rotation.optimize = saved.rotation_optimize;
                                    a.constraints.has_target_sweep = saved.had_sweep;
                                    a.constraints.target_sweep = saved.old_sweep;
                                    a.constraints.sweep_sign = saved.old_sweep_sign;
                                    a.start_angle.optimize = saved.start_optimize;
                                    a.end_angle.optimize = saved.end_optimize;
                                }
                            }
                            None => {}
                        }
                        clean.points.remove(drag_pt);
                        self.drag_saved_cost = self.last_cost;
                        self.drag_saved_snapshot = bincode::serialize(&clean).ok();
                    }
        }
    }

    // Remove the drag apparatus (temp point + constraint) from the sketch.
    fn remove_drag_apparatus(&mut self, drag_pt: arael::refs::Ref<Point>) {
        match self.grab {
            Some(GrabTarget::Point(_)) => { self.sketch.get_mut().coincident_pp.pop(); }
            Some(GrabTarget::LineP1(_)) => { self.sketch.get_mut().coincident_lp1.pop(); }
            Some(GrabTarget::LineP2(_)) => { self.sketch.get_mut().coincident_lp2.pop(); }
            Some(GrabTarget::ArcCenter(_)) => { self.sketch.get_mut().coincident_arc_center.pop(); }
            Some(GrabTarget::ArcStart(_)) => { self.sketch.get_mut().coincident_arc_start.pop(); }
            Some(GrabTarget::ArcEnd(_)) => { self.sketch.get_mut().coincident_arc_end.pop(); }
            Some(GrabTarget::LineDrag(_)) => {
                self.sketch.get_mut().coincident_lp1.pop();
                self.sketch.get_mut().coincident_lp2.pop();
                if let Some(dp2) = self.drag_point2.take() {
                    self.sketch.get_mut().points.remove(dp2);
                }
            }
            Some(GrabTarget::ArcDrag(r)) => {
                self.sketch.get_mut().coincident_arc_center.pop();
                if let Some(saved) = self.drag_saved_arc_locks.take() {
                    let a = &mut self.sketch.get_mut().arcs[r];
                    a.constraints.has_target_radius = saved.had_radius;
                    a.constraints.target_radius = saved.old_radius;
                    a.constraints.has_target_radius_b = saved.had_radius_b;
                    a.constraints.target_radius_b = saved.old_radius_b;
                    a.rotation.optimize = saved.rotation_optimize;
                    a.constraints.has_target_sweep = saved.had_sweep;
                    a.constraints.target_sweep = saved.old_sweep;
                    a.constraints.sweep_sign = saved.old_sweep_sign;
                    a.start_angle.optimize = saved.start_optimize;
                    a.end_angle.optimize = saved.end_optimize;
                }
            }
            None => {}
        }
        self.sketch.get_mut().points.remove(drag_pt);
    }


    // End drag: remove temporary point and constraint, auto-snap, record action.
    // If the final cost is much worse than pre-drag, revert to pre-drag state.
    fn end_drag(&mut self, hit_threshold: f64) {
        self.begin_group();
        self.drag_snap_preview = None;
        self.drag_perp_already.clear();
        // Roll back the auto-anchor hack before remove_drag_apparatus
        // so the apparatus pop()s hit the right vec entries.
        if let Some(state) = self.drag_auto_anchors.take() {
            self.sketch.get_mut().remove_drag_auto_anchors(state);
        }
        if let Some(drag_pt) = self.drag_point.take() {
            let _drag_pos = self.sketch.points[drag_pt].pos.value;
            let grab = self.grab;

            // Remove drag apparatus and re-solve
            self.remove_drag_apparatus(drag_pt);
            let result = self.sketch.solve();
            self.last_cost = result.end_cost;

            // If cost is much worse than pre-drag, revert to pre-drag state
            if self.last_cost > self.drag_saved_cost + 1e-3
                && let Some(snap) = self.drag_saved_snapshot.take()
                    && let Ok(restored) = bincode::deserialize::<Sketch>(&snap) {
                        self.sketch = restored.into();
                        let result = self.sketch.solve();
                        self.last_cost = result.end_cost;
                    }
            self.drag_saved_snapshot = None;

            // Record drag as a non-deterministic action with full state snapshot
            let snapshot = bincode::serialize(&self.sketch).unwrap();
            let action = Action::Drag { snapshot };
            self.history.push(action, &self.sketch, arael_sketch_backend::history::CursorState { pos: self.command_cursor, tangent: self.command_cursor_tangent });

            // Auto-snap: if a point-like entity was dragged near another,
            // create a coincident constraint
            // Use the same in-loop "skip already-attached" filter as
            // update_drag so a twin sitting at the just-released position
            // cannot shadow the real new target.
            // Snapshot auto-perp hint before it's cleared -- it may still
            // apply even when no snap target wins, and it must be emitted
            // inside the same undo group as the drag itself.
            let perp_hint = self.drag_perp_snap.take();
            let hv_hint = self.drag_hv_hint.take();
            let collinear_hint = self.drag_collinear_hint.take();
            match grab {
                Some(g @ (GrabTarget::LineP1(line) | GrabTarget::LineP2(line))) => {
                    let is_p1 = matches!(g, GrabTarget::LineP1(_));
                    let ep_pos = if is_p1 {
                        self.sketch.lines[line].p1.value
                    } else {
                        self.sketch.lines[line].p2.value
                    };
                    let snap_kind = match self.find_snap_target_filter(
                        ep_pos, hit_threshold, Some(line), None,
                        |t| !self.has_existing_snap_attachment(g, *t),
                    ) {
                        Some((_, snap)) => {
                            self.apply_snap_coincident(snap, line, is_p1);
                            Some(snap)
                        }
                        None => None,
                    };
                    // Perpendicular fires when the perp hint was active
                    // AND the snap (if any) is a line body -- both
                    // constraints can geometrically coexist at the
                    // intersection. Other snap kinds pin the position
                    // and would conflict.
                    let perp_compatible = matches!(snap_kind, None | Some(SnapTarget::Line(_)));
                    if perp_compatible {
                        if let Some((host, _)) = perp_hint {
                            if !self.has_perp_conflict(line, host) {
                                self.exec(Action::ApplyPerpendicular { a: line, b: host });
                            }
                        }
                    }
                    // Auto-collinear: emit when the hint was active and
                    // no position-pinning snap fired.
                    if snap_kind.is_none()
                        && let Some((cl_line, host)) = collinear_hint
                        && cl_line == line
                    {
                        let action = Action::ApplyCollinear { a: line, b: host };
                        if arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action).is_none() {
                            self.exec(action);
                        }
                    }
                    // Auto-H/V: emit when the hint was active and no
                    // position-pinning snap fired. A line-body snap is
                    // incompatible (the snapped line fixes orientation
                    // already); other snaps already pinned the endpoint
                    // and we don't know it's still axis-aligned.
                    if snap_kind.is_none()
                        && let Some((hv_line, horizontal)) = hv_hint
                        && hv_line == line
                    {
                        let action = if horizontal {
                            Action::ApplyHorizontal { lines: vec![line] }
                        } else {
                            Action::ApplyVertical { lines: vec![line] }
                        };
                        if arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action).is_none() {
                            self.exec(action);
                        }
                    }
                }
                Some(g @ GrabTarget::ArcCenter(arc)) => {
                    let pos = self.sketch.arcs[arc].center.value;
                    if let Some((_, snap)) = self.find_snap_target_filter(
                        pos, hit_threshold, None, Some(arc),
                        |t| !self.has_existing_snap_attachment(g, *t),
                    ) {
                        self.apply_snap_coincident_arc(snap, arc, ArcPoint::Center, pos);
                    }
                }
                Some(g @ GrabTarget::ArcStart(arc)) => {
                    let pos = arc_start_pos(&self.sketch.arcs[arc]);
                    if let Some((_, snap)) = self.find_snap_target_filter(
                        pos, hit_threshold, None, Some(arc),
                        |t| !self.has_existing_snap_attachment(g, *t),
                    ) {
                        self.apply_snap_coincident_arc(snap, arc, ArcPoint::Start, pos);
                    }
                }
                Some(g @ GrabTarget::ArcEnd(arc)) => {
                    let pos = arc_end_pos(&self.sketch.arcs[arc]);
                    if let Some((_, snap)) = self.find_snap_target_filter(
                        pos, hit_threshold, None, Some(arc),
                        |t| !self.has_existing_snap_attachment(g, *t),
                    ) {
                        self.apply_snap_coincident_arc(snap, arc, ArcPoint::End, pos);
                    }
                }
                Some(g @ GrabTarget::Point(point)) => {
                    let pos = self.sketch.points[point].pos.value;
                    if let Some((_, snap)) = self.find_snap_target_filter(
                        pos, hit_threshold, None, None,
                        |t| !self.has_existing_snap_attachment(g, *t),
                    ) {
                        self.apply_snap_coincident_point(snap, point);
                    }
                }
                _ => {}
            }
            // Auto-snap errors (e.g. redundant coincident) should not
            // be shown -- clear both the error string and the
            // blocker-flash state so an internally-rejected
            // auto-perp / auto-snap doesn't flash constraints at the
            // user on every drag release.
            self.status_error = None;
            self.flash_names.clear();
            self.flash_start = None;
        }
        self.grab = None;
    }

    // Toggle selection
    /// Return a command-compatible name for a selection (for pasting into command input).
    fn selection_command_name(&self, sel: &Selection) -> Option<String> {
        match *sel {
            Selection::Point(r) => Some(self.sketch.points[r].name.clone()),
            Selection::Line(r) => Some(self.sketch.lines[r].name.clone()),
            Selection::LineP1(r) => Some(format!("{}.p1", self.sketch.lines[r].name)),
            Selection::LineP2(r) => Some(format!("{}.p2", self.sketch.lines[r].name)),
            Selection::Arc(r) => Some(self.sketch.arcs[r].name.clone()),
            Selection::ArcCenter(r) => Some(format!("{}.center", self.sketch.arcs[r].name)),
            Selection::ArcStart(r) => Some(format!("{}.start", self.sketch.arcs[r].name)),
            Selection::ArcEnd(r) => Some(format!("{}.end", self.sketch.arcs[r].name)),
            Selection::Dimension(i) => {
                if i < self.sketch.dimensions.len() {
                    Some(self.sketch.dimensions[i].name.clone())
                } else { None }
            }
            Selection::Constraint(id) => self.constraint_name(id),
        }
    }

    fn toggle_selection(&mut self, sel: Selection) {
        if let Some(idx) = self.selection.iter().position(|s| *s == sel) {
            self.selection.remove(idx);
        } else {
            // Constraints are exclusive - clear everything else when selecting one
            if matches!(sel, Selection::Constraint(_)) {
                self.selection.clear();
            } else {
                // Clear any constraint selections when selecting non-constraints
                self.selection.retain(|s| !matches!(s, Selection::Constraint(_)));
            }
            self.selection.push(sel);
        }
    }

    fn load_from_json(&mut self, json: &str) {
        match serde_json::from_str::<Sketch>(json) {
            Ok(mut sketch) => {
                sketch.dedup_constraints();
                sketch.consolidate_helper_constraints();
                sketch.fixup_tangent_signs();
                sketch.assign_constraint_names();
                let result = sketch.solve();
                self.last_cost = result.end_cost;
                self.sketch = sketch.into();
                self.selection.clear();
                self.history = History::new(&self.sketch);
                self.line_draw = None;
                self.circle_draw = None;
                self.arc_draw = None;
                self.pending_fit = true;
                self.status_error = None;
                self.compute_dof_async();
            }
            Err(e) => eprintln!("Failed to parse sketch: {}", e),
        }
    }


    /// Recompute cached cost from the current sketch state.
    pub fn update_cost(&mut self) {
        let mut params = Vec::new();
        self.sketch.get_mut().serialize(&mut params);
        self.last_cost = self.sketch.get_mut().calc_cost(&params);
    }

    /// Check if background DOF computation finished, update display.
    pub fn poll_dof(&mut self) {
        if let Some(dof) = self.dof_output.lock().unwrap().take() {
            // Worker results reflect whatever sketch state was in
            // dof_input when the worker picked it up. If the main
            // thread has since computed a definitive DOF inline (e.g.
            // validate_and_apply_constraint caches it after accepting
            // a constraint), that cached value is authoritative --
            // blindly adopting the older worker result would let a
            // post-AddLine DOF=16 clobber the post-Horizontal DOF=4
            // produced by the rect tool a moment earlier.
            if self.sketch.cached_dof.is_none() {
                self.dof_display = Some(dof);
                self.sketch.get_mut().cached_dof = Some(dof);
            }
        }
    }

    /// Queue DOF computation on the background worker thread.
    /// Only the latest sketch state is kept -- intermediate requests are discarded.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_dof_async(&mut self) {
        // Invalidate any result still sitting in dof_output from an
        // earlier (now-stale) sketch state -- e.g. --empty then
        // --script robot.cmd: the empty sketch's worker result (0)
        // would otherwise be picked up by the first poll_dof frame
        // and overwrite the correct cached value for the loaded
        // script.
        *self.dof_output.lock().unwrap() = None;
        if let Some(d) = self.sketch.cached_dof {
            self.dof_display = Some(d);
            return;
        }
        self.dof_display = None;
        if let Ok(data) = bincode::serialize(&self.sketch) {
            *self.dof_input.lock().unwrap() = Some(data);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn compute_dof_async(&mut self) {
        // Reading the DOF fills a cache and changes no structure, so it must
        // not retire the derived state or the warm session.
        self.dof_display = self.sketch.mutate_values(|s| s.dof().ok());
    }

    /// Create a CommandContext view of this app's state, run commands, sync back.
    pub fn run_commands(&mut self, input: &str) -> Vec<arael_sketch_backend::commands::CommandResult> {
        let empty_sketch = Sketch::new();
        let empty_history = arael_sketch_backend::history::History::new(&empty_sketch);
        let mut ctx = arael_sketch_backend::commands::CommandContext {
            sketch: std::mem::replace(&mut self.sketch, empty_sketch.into()),
            history: std::mem::replace(&mut self.history, empty_history),
            selection: std::mem::take(&mut self.selection),
            session_vars: std::mem::take(&mut self.session_vars),
            session_vecs: std::mem::take(&mut self.session_vecs),
            session_names: std::mem::take(&mut self.session_names),
            cursor: self.command_cursor,
            cursor_tangent: self.command_cursor_tangent,
            saved_cursor: arael_sketch_backend::history::CursorState::default(),
            status_error: self.status_error.take(),
            status_blocker_names: None,
            last_cost: self.last_cost,
            dof: self.dof_display,
            scale: self.scale,
            offset_x: self.offset.x,
            offset_y: self.offset.y,
            pending_fit: self.pending_fit,
            blocked_commands: Vec::new(),
            skip_dof_check: false,
            exit_requested: false,
            drag_raw: self.drag_raw,
            echo_stdout: self.echo_stdout,
        };
        let results = arael_sketch_backend::commands::execute(&mut ctx, input);
        // Sync back
        self.sketch = ctx.sketch;
        self.history = ctx.history;
        self.selection = ctx.selection;
        self.session_vars = ctx.session_vars;
        self.session_vecs = ctx.session_vecs;
        self.session_names = ctx.session_names;
        self.command_cursor = ctx.cursor;
        self.command_cursor_tangent = ctx.cursor_tangent;
        self.status_error = ctx.status_error;
        if let Some(names) = ctx.status_blocker_names.take() {
            self.start_constraint_flash(names);
        }
        self.last_cost = ctx.last_cost;
        self.dof_display = ctx.dof;
        self.scale = ctx.scale;
        self.offset.x = ctx.offset_x;
        self.offset.y = ctx.offset_y;
        self.pending_fit = ctx.pending_fit;
        if ctx.exit_requested { self.exit_requested = true; }
        self.show_hints = false;
        self.compute_dof_async();
        results
    }

    /// Begin a 3-flash cycle at 3 Hz on the named constraints. Called
    /// when a constraint is rejected and the blocker analysis
    /// identified conflicting existing constraints -- the user sees
    /// them briefly in the selected colour (even if they would be
    /// invisible in the normal render, e.g. coincident bridges).
    pub fn start_constraint_flash(&mut self, names: Vec<String>) {
        self.flash_names = names;
        self.flash_start = Some(web_time::Instant::now());
    }

    /// Return true if the named constraint is currently in an "on"
    /// phase of the flash cycle. Cycle: 3 flashes at 3 Hz => total
    /// duration 1 second, on during the first half of each 333ms
    /// period. Used by render code to pick the highlight colour.
    pub fn flash_on_now(&self, name: &str) -> bool {
        if !self.flash_pulse_on() { return false; }
        self.flash_names.iter().any(|n| n == name)
    }

    /// Pulse state: true during any "on" half-period of the flash.
    /// Shared by `flash_on_now` and by `flash_show_hidden`.
    pub fn flash_pulse_on(&self) -> bool {
        let Some(start) = self.flash_start else { return false };
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 1.0 { return false; }
        let period = 1.0 / 3.0;
        let phase = (elapsed % period) / period;
        phase < 0.5
    }

    /// True any time within the 1 s flash window (used by marker
    /// builders that must still decide visibility when we're in the
    /// "off" half of the current pulse -- we still want the marker to
    /// be registered so it can be highlighted when the pulse flips
    /// back on, without extra round-trips to rebuild the marker set).
    pub fn flash_window_active(&self) -> bool {
        let Some(start) = self.flash_start else { return false };
        start.elapsed().as_secs_f64() <= 1.0 && !self.flash_names.is_empty()
    }

    /// For a `ConstraintId`, return whether that constraint is a
    /// flash target (name appears in `flash_names`).
    pub fn is_flash_target(&self, id: ConstraintId) -> bool {
        if !self.flash_window_active() { return false; }
        match self.constraint_name(id) {
            Some(n) => self.flash_names.iter().any(|x| *x == n),
            None => false,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_commands_with_blocked(&mut self, input: &str, blocked: Vec<&'static str>) -> Vec<arael_sketch_backend::commands::CommandResult> {
        let empty_sketch = Sketch::new();
        let empty_history = arael_sketch_backend::history::History::new(&empty_sketch);
        let mut ctx = arael_sketch_backend::commands::CommandContext {
            sketch: std::mem::replace(&mut self.sketch, empty_sketch.into()),
            history: std::mem::replace(&mut self.history, empty_history),
            selection: std::mem::take(&mut self.selection),
            session_vars: std::mem::take(&mut self.session_vars),
            session_vecs: std::mem::take(&mut self.session_vecs),
            session_names: std::mem::take(&mut self.session_names),
            cursor: self.command_cursor,
            cursor_tangent: self.command_cursor_tangent,
            saved_cursor: arael_sketch_backend::history::CursorState::default(),
            status_error: self.status_error.take(),
            status_blocker_names: None,
            last_cost: self.last_cost,
            dof: self.dof_display,
            scale: self.scale,
            offset_x: self.offset.x,
            offset_y: self.offset.y,
            pending_fit: self.pending_fit,
            blocked_commands: blocked,
            skip_dof_check: false,
            exit_requested: false,
            drag_raw: self.drag_raw,
            echo_stdout: self.echo_stdout,
        };
        let results = arael_sketch_backend::commands::execute(&mut ctx, input);
        self.sketch = ctx.sketch;
        self.history = ctx.history;
        self.selection = ctx.selection;
        self.session_vars = ctx.session_vars;
        self.session_vecs = ctx.session_vecs;
        self.session_names = ctx.session_names;
        self.command_cursor = ctx.cursor;
        self.command_cursor_tangent = ctx.cursor_tangent;
        self.status_error = ctx.status_error;
        if let Some(names) = ctx.status_blocker_names.take() {
            self.start_constraint_flash(names);
        }
        self.last_cost = ctx.last_cost;
        self.dof_display = ctx.dof;
        self.scale = ctx.scale;
        self.offset.x = ctx.offset_x;
        self.offset.y = ctx.offset_y;
        self.pending_fit = ctx.pending_fit;
        self.show_hints = false;
        self.compute_dof_async();
        results
    }

    // Execute an action: apply to sketch and record in history.
    // For constraint actions: validates by solving and checking cost.
    // Rejects constraints that make a healthy sketch unsolvable.
    /// Apply an action. Returns what it added, so a caller acting on the new
    /// entity uses the ref the arena actually issued.
    pub fn exec(&mut self, action: Action) -> arael_sketch_backend::actions::Created {
        use arael_sketch_backend::actions::Created;
        let mut created = Created::Nothing;
        self.status_error = None;
        self.show_hints = false;

        if action.is_constraint_action() {
            match arael_sketch_backend::commands::validate_and_apply_constraint(
                self.sketch.get_mut(), &action, false)
            {
                Ok(new_cost) => {
                    self.last_cost = new_cost;
                    self.history.push(action, &self.sketch, arael_sketch_backend::history::CursorState { pos: self.command_cursor, tangent: self.command_cursor_tangent });
                }
                Err(rejection) => {
                    self.status_error = Some(rejection.message);
                    if !rejection.blocker_names.is_empty() {
                        self.start_constraint_flash(rejection.blocker_names);
                    }
                }
            }
        } else {
            created = action.apply(self.sketch.get_mut());
            self.sketch.get_mut().dedup_constraints();
            self.history.push(action, &self.sketch, arael_sketch_backend::history::CursorState { pos: self.command_cursor, tangent: self.command_cursor_tangent });
        }
        self.compute_dof_async();
        created
    }

    /// Apply a user parameter change with solve and cost check.
    /// Rolls back if the change causes constraints to break.
    pub fn apply_param_change(&mut self, action: Action) {
        use arael::simple_lm::LmProblem;
        let snapshot = bincode::serialize(&self.sketch).ok();
        let old_cost = {
            let mut params = Vec::new();
            self.sketch.get_mut().serialize(&mut params);
            self.sketch.get_mut().calc_cost(&params)
        };
        self.begin_group();
        self.exec(action);
        self.sketch.get_mut().update_expr_dim_values();
        let new_cost = self.sketch.solve().end_cost;
        self.last_cost = new_cost;
        if new_cost > old_cost + 1e-3
            && let Some(ref snap) = snapshot
                && let Ok(restored) = bincode::deserialize::<Sketch>(snap) {
                    self.sketch = restored.into();
                    self.status_error = Some("Parameter change rejected: could not satisfy all constraints".into());
                }
    }

    // Apply constraint to current selection
    // Check if the current selection exactly satisfies a constraint for immediate application
    fn can_apply_constraint(&self, ct: ConstraintType) -> bool {
        let sel = &self.selection;
        match ct {
            ConstraintType::Horizontal | ConstraintType::Vertical => {
                !sel.is_empty() && sel.iter().all(|s| matches!(s, Selection::Line(_)))
            }
            ConstraintType::Parallel => {
                // Line+Line, Line+Arc (either order), or Arc+Arc. Rejecting
                // circular arcs vs ellipses happens in apply_parallel -- the
                // selection gate only checks the coarse entity mix so the
                // user gets a clear "you need two orientable entities"
                // prompt rather than a silent no-op.
                sel.len() == 2 && {
                    let lines = sel.iter().filter(|s| matches!(s, Selection::Line(_))).count();
                    let arcs = sel.iter().filter(|s| matches!(s, Selection::Arc(_))).count();
                    lines + arcs == 2
                }
            }
            ConstraintType::Perpendicular => {
                sel.len() == 2 && sel.iter().all(|s| matches!(s, Selection::Line(_)))
            }
            ConstraintType::EqualLength => {
                sel.len() == 2 && (
                    sel.iter().all(|s| matches!(s, Selection::Line(_)))
                    || sel.iter().all(|s| matches!(s, Selection::Arc(_)))
                )
            }
            ConstraintType::Tangent => {
                sel.len() == 2 && {
                    let lines = sel.iter().filter(|s| matches!(s, Selection::Line(_))).count();
                    let arcs = sel.iter().filter(|s| matches!(s, Selection::Arc(_))).count();
                    (lines == 1 && arcs == 1) || arcs == 2
                }
            }
            ConstraintType::Coincident => {
                sel.len() == 2 && {
                    let point_like = |s: &Selection| matches!(s,
                        Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                        | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_));
                    let body = |s: &Selection| matches!(s, Selection::Line(_) | Selection::Arc(_));
                    // Two point-like, or one point-like + one body, or two bodies
                    (sel.iter().all(&point_like))
                    || (sel.iter().filter(|s| point_like(s)).count() == 1 && sel.iter().filter(|s| body(s)).count() == 1)
                    || (sel.iter().all(|s| matches!(s, Selection::Line(_))))
                }
            }
            ConstraintType::Collinear => {
                sel.len() == 2 && sel.iter().all(|s| matches!(s, Selection::Line(_)))
            }
            ConstraintType::Midpoint => {
                sel.len() == 2 && {
                    let point_like = |s: &Selection| matches!(s,
                        Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                        | Selection::ArcStart(_) | Selection::ArcEnd(_));
                    let lines = sel.iter().filter(|s| matches!(s, Selection::Line(_))).count();
                    let arcs = sel.iter().filter(|s| matches!(s, Selection::Arc(_))).count();
                    let pts = sel.iter().filter(|s| point_like(s)).count();
                    (pts == 1 && lines == 1) || (pts == 1 && arcs == 1)
                }
            }
            ConstraintType::Symmetry => {
                if sel.len() != 3 { return false; }
                // 3 lines (LL symmetry) or 2 point-likes + 1 line (PP symmetry)
                let lines = sel.iter().filter(|s| matches!(s, Selection::Line(_))).count();
                let arcs = sel.iter().filter(|s| matches!(s, Selection::Arc(_))).count();
                let point_likes = sel.iter().filter(|s| matches!(s,
                    Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_))).count();
                (lines == 3) || (lines == 1 && point_likes == 2) || (lines == 1 && arcs == 2)
            }
            ConstraintType::Lock => {
                !sel.is_empty() && sel.iter().all(|s| matches!(s,
                    Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_)))
            }
            ConstraintType::ToggleConstruction => {
                !sel.is_empty() && sel.iter().all(|s| matches!(s, Selection::Line(_) | Selection::Arc(_)))
            }
        }
    }

    // Check if entering constraint mode makes sense given current selection.
    // True if the selection could still lead to a valid constraint with more clicks.
    fn could_enter_constraint_mode(&self, ct: ConstraintType) -> bool {
        let sel = &self.selection;
        if sel.is_empty() { return true; }

        // Filter to only valid entities for this constraint
        let valid: Vec<&Selection> = sel.iter().filter(|s| Self::is_valid_for_constraint(ct, s)).collect();
        if valid.is_empty() { return false; }

        let needed = match ct {
            ConstraintType::Horizontal | ConstraintType::Vertical
            | ConstraintType::Lock | ConstraintType::ToggleConstruction => 1,
            _ => 2,
        };

        // If we already have enough, it must be directly applicable (handled by can_apply)
        if valid.len() >= needed { return false; }

        // For tangent with 1 selection: check it's a valid partial (line or arc)
        if ct == ConstraintType::Tangent && valid.len() == 1 {
            return matches!(valid[0], Selection::Line(_) | Selection::Arc(_));
        }

        true
    }

    // Check if a selection entity is valid to add in constraint mode
    fn is_valid_for_constraint(ct: ConstraintType, sel: &Selection) -> bool {
        match ct {
            ConstraintType::Horizontal | ConstraintType::Vertical
            | ConstraintType::Perpendicular => {
                matches!(sel, Selection::Line(_))
            }
            ConstraintType::Parallel => {
                // Lines always valid; arcs valid too since ellipses can pair
                // with lines or other ellipses through ArcLineParallel /
                // ArcArcParallel. Circular arcs still get through the gate
                // but are rejected in apply_parallel with a clear status
                // message.
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::EqualLength => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Tangent => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Coincident => {
                matches!(sel, Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_)
                    | Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Collinear => {
                matches!(sel, Selection::Line(_))
            }
            ConstraintType::Midpoint => {
                matches!(sel, Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcStart(_) | Selection::ArcEnd(_) | Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Symmetry => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_) | Selection::Point(_)
                    | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_))
            }
            ConstraintType::Lock => {
                matches!(sel, Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_))
            }
            ConstraintType::ToggleConstruction => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
        }
    }

    // Try to apply constraint if selection is valid, otherwise enter constraint mode
    fn try_apply_or_enter_mode(&mut self, ct: ConstraintType) {
        self.status_error = None;
        if self.can_apply_constraint(ct) {
            match ct {
                ConstraintType::Horizontal => self.apply_horizontal(),
                ConstraintType::Vertical => self.apply_vertical(),
                ConstraintType::Coincident => self.apply_coincident(),
                ConstraintType::Parallel => self.apply_parallel(),
                ConstraintType::Perpendicular => self.apply_perpendicular(),
                ConstraintType::EqualLength => self.apply_equal_length(),
                ConstraintType::Tangent => self.apply_tangent(),
                ConstraintType::Collinear => self.apply_collinear(),
                ConstraintType::Midpoint => self.apply_midpoint(),
                ConstraintType::Symmetry => self.apply_symmetry(),
                ConstraintType::Lock => self.apply_lock(),
                ConstraintType::ToggleConstruction => self.apply_toggle_construction(),
            }
        } else {
            // Enter constraint mode, keep current selection (filter invalid ones)
            self.selection.retain(|s| Self::is_valid_for_constraint(ct, s));
            self.tool = Tool::ConstraintMode(ct);
        }
    }

    // Check if two "vertices" (point-like entities) are already transitively coincident.
    // Uses the same union-find as compute_locked_sets.
    fn are_transitively_coincident(&self, sel_a: Selection, sel_b: Selection) -> bool {
        let np = self.sketch.points.slot_count();
        let nl = self.sketch.lines.slot_count();
        let na = self.sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;
        let mut parent: Vec<usize> = (0..total).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; } x
        };
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb { parent[ra] = rb; }
        };
        let pt_id = |r: Ref<Point>| r.index() as usize;
        let lp1_id = |r: Ref<Line>| np + r.index() as usize;
        let lp2_id = |r: Ref<Line>| np + nl + r.index() as usize;
        let ac_id = |r: Ref<Arc>| np + 2 * nl + r.index() as usize;
        let as_id = |r: Ref<Arc>| np + 2 * nl + na + r.index() as usize;
        let ae_id = |r: Ref<Arc>| np + 2 * nl + 2 * na + r.index() as usize;

        // Build unions (same as compute_locked_sets)
        for c in &self.sketch.coincident_pp { union(&mut parent, pt_id(c.a), pt_id(c.b)); }
        for c in &self.sketch.coincident_lp1 { union(&mut parent, lp1_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_lp2 { union(&mut parent, lp2_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_ll11 { union(&mut parent, lp1_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll12 { union(&mut parent, lp1_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_ll21 { union(&mut parent, lp2_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll22 { union(&mut parent, lp2_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_arc_center { union(&mut parent, pt_id(c.point), ac_id(c.arc)); }
        for c in &self.sketch.coincident_arc_start { union(&mut parent, pt_id(c.point), as_id(c.arc)); }
        for c in &self.sketch.coincident_arc_end { union(&mut parent, pt_id(c.point), ae_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_center { union(&mut parent, lp1_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_center { union(&mut parent, lp2_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_start { union(&mut parent, lp1_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_start { union(&mut parent, lp2_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_end { union(&mut parent, lp1_id(c.line), ae_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_end { union(&mut parent, lp2_id(c.line), ae_id(c.arc)); }
        for c in &self.sketch.concentric { union(&mut parent, ac_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_start { union(&mut parent, ac_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_end { union(&mut parent, ac_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_center { union(&mut parent, as_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_center { union(&mut parent, ae_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_start { union(&mut parent, as_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_end { union(&mut parent, as_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_start { union(&mut parent, ae_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_end { union(&mut parent, ae_id(c.a), ae_id(c.b)); }

        let sel_to_id = |s: Selection| -> Option<usize> {
            match s {
                Selection::Point(r) => Some(pt_id(r)),
                Selection::LineP1(r) => Some(lp1_id(r)),
                Selection::LineP2(r) => Some(lp2_id(r)),
                Selection::ArcCenter(r) => Some(ac_id(r)),
                Selection::ArcStart(r) => Some(as_id(r)),
                Selection::ArcEnd(r) => Some(ae_id(r)),
                _ => None,
            }
        };

        if let (Some(id_a), Some(id_b)) = (sel_to_id(sel_a), sel_to_id(sel_b)) {
            find(&mut parent, id_a) == find(&mut parent, id_b)
        } else {
            false
        }
    }

    // Convert a line endpoint + snap target to Selection pairs for transitive check
    fn snap_to_selection(snap: SnapTarget) -> Option<Selection> {
        match snap {
            SnapTarget::Point(r) => Some(Selection::Point(r)),
            SnapTarget::LineP1(r) => Some(Selection::LineP1(r)),
            SnapTarget::LineP2(r) => Some(Selection::LineP2(r)),
            SnapTarget::ArcCenter(r) => Some(Selection::ArcCenter(r)),
            SnapTarget::ArcStart(r) => Some(Selection::ArcStart(r)),
            SnapTarget::ArcEnd(r) => Some(Selection::ArcEnd(r)),
            SnapTarget::Line(_)
            | SnapTarget::ArcBody(_)
            | SnapTarget::LineMidpoint(_)
            | SnapTarget::ArcMidpoint(_) => None,
        }
    }

    // Map a point-like GrabTarget to the Selection identifying that
    // same entity, for use with `are_transitively_coincident`. Body
    // drags have no single-point Selection equivalent -> None.
    fn grab_to_selection(grab: GrabTarget) -> Option<Selection> {
        match grab {
            GrabTarget::Point(r) => Some(Selection::Point(r)),
            GrabTarget::LineP1(r) => Some(Selection::LineP1(r)),
            GrabTarget::LineP2(r) => Some(Selection::LineP2(r)),
            GrabTarget::ArcCenter(r) => Some(Selection::ArcCenter(r)),
            GrabTarget::ArcStart(r) => Some(Selection::ArcStart(r)),
            GrabTarget::ArcEnd(r) => Some(Selection::ArcEnd(r)),
            GrabTarget::LineDrag(_) | GrabTarget::ArcDrag(_) => None,
        }
    }

    // Selections currently anchored at `l`'s midpoint via any of the
    // midpoint-* constraint tables. Used to recognize transitive
    // attachment: if the dragged entity is coincident with any of
    // these, it's also at `l`'s midpoint (and on `l`'s body).
    fn selections_at_line_midpoint(&self, l: Ref<Line>) -> Vec<Selection> {
        let mut v = Vec::new();
        for c in &self.sketch.midpoint { if c.line == l { v.push(Selection::Point(c.point)); } }
        for c in &self.sketch.midpoint_lp1 { if c.target == l { v.push(Selection::LineP1(c.line)); } }
        for c in &self.sketch.midpoint_lp2 { if c.target == l { v.push(Selection::LineP2(c.line)); } }
        for c in &self.sketch.midpoint_arc_start { if c.line == l { v.push(Selection::ArcStart(c.arc)); } }
        for c in &self.sketch.midpoint_arc_end { if c.line == l { v.push(Selection::ArcEnd(c.arc)); } }
        v
    }

    fn selections_at_arc_midpoint(&self, a: Ref<Arc>) -> Vec<Selection> {
        let mut v = Vec::new();
        for c in &self.sketch.midpoint_arc_point { if c.arc == a { v.push(Selection::Point(c.point)); } }
        for c in &self.sketch.midpoint_lp1_arc { if c.arc == a { v.push(Selection::LineP1(c.line)); } }
        for c in &self.sketch.midpoint_lp2_arc { if c.arc == a { v.push(Selection::LineP2(c.line)); } }
        for c in &self.sketch.midpoint_arc_start_arc { if c.b == a { v.push(Selection::ArcStart(c.a)); } }
        for c in &self.sketch.midpoint_arc_end_arc { if c.b == a { v.push(Selection::ArcEnd(c.a)); } }
        v
    }

    // Whether the dragged entity is already attached to `snap`, so the
    // snap hint and pull should be suppressed during drag.
    //
    // Beyond direct constraint-table checks, a body/midpoint target is
    // also considered "attached" when the dragged entity is transitively
    // coincident with any endpoint of that body. Without this, dragging
    // a line endpoint that is coincident with another line's endpoint
    // would forever snap to that other line's body (cursor sits on it
    // by construction), shadowing real snap candidates.
    fn has_existing_snap_attachment(&self, grab: GrabTarget, snap: SnapTarget) -> bool {
        let Some(grab_sel) = Self::grab_to_selection(grab) else { return false; };

        if let Some(snap_sel) = Self::snap_to_selection(snap) {
            return self.are_transitively_coincident(grab_sel, snap_sel);
        }

        // Body / midpoint snap targets.
        match snap {
            SnapTarget::Line(l) => {
                // Attached to `l`'s body iff: transitively coincident
                // with either endpoint of `l`, OR transitively coincident
                // with ANY entity currently anchored at `l`'s midpoint
                // (midpoints sit on the body), OR a direct *_on_line
                // entry between the dragged entity and `l` exists.
                // Arc-point variants route through a helper point +
                // CoincidentArc* constraint + PointOnLine(helper, l);
                // detect that compound attachment here so the release
                // path doesn't re-add a duplicate PointOnLine.
                let arc_point_on_line = |arc: Ref<Arc>, which: ArcPoint| -> bool {
                    let helper = match which {
                        ArcPoint::Center => self.sketch.coincident_arc_center.iter().find(|c| c.arc == arc).map(|c| c.point),
                        ArcPoint::Start => self.sketch.coincident_arc_start.iter().find(|c| c.arc == arc).map(|c| c.point),
                        ArcPoint::End => self.sketch.coincident_arc_end.iter().find(|c| c.arc == arc).map(|c| c.point),
                    };
                    match helper {
                        Some(hp) => self.sketch.point_on_line.iter().any(|c| c.point == hp && c.line == l),
                        None => false,
                    }
                };
                let direct = match grab {
                    GrabTarget::Point(r) => self.sketch.point_on_line.iter().any(|c| c.point == r && c.line == l),
                    GrabTarget::LineP1(r) => self.sketch.line_p1_on_line.iter().any(|c| c.a == r && c.b == l),
                    GrabTarget::LineP2(r) => self.sketch.line_p2_on_line.iter().any(|c| c.a == r && c.b == l),
                    GrabTarget::ArcCenter(a) => arc_point_on_line(a, ArcPoint::Center),
                    GrabTarget::ArcStart(a) => arc_point_on_line(a, ArcPoint::Start),
                    GrabTarget::ArcEnd(a) => arc_point_on_line(a, ArcPoint::End),
                    _ => false,
                };
                direct
                    || self.are_transitively_coincident(grab_sel, Selection::LineP1(l))
                    || self.are_transitively_coincident(grab_sel, Selection::LineP2(l))
                    || self.selections_at_line_midpoint(l).iter()
                        .any(|s| self.are_transitively_coincident(grab_sel, *s))
            }
            SnapTarget::ArcBody(a) => {
                // Same arc-point-helper plumbing for point_on_arc.
                let arc_point_on_arc = |arc_src: Ref<Arc>, which: ArcPoint| -> bool {
                    let helper = match which {
                        ArcPoint::Center => self.sketch.coincident_arc_center.iter().find(|c| c.arc == arc_src).map(|c| c.point),
                        ArcPoint::Start => self.sketch.coincident_arc_start.iter().find(|c| c.arc == arc_src).map(|c| c.point),
                        ArcPoint::End => self.sketch.coincident_arc_end.iter().find(|c| c.arc == arc_src).map(|c| c.point),
                    };
                    match helper {
                        Some(hp) => self.sketch.point_on_arc.iter().any(|c| c.point == hp && c.arc == a),
                        None => false,
                    }
                };
                let direct = match grab {
                    GrabTarget::Point(r) => self.sketch.point_on_arc.iter().any(|c| c.point == r && c.arc == a),
                    GrabTarget::LineP1(r) => self.sketch.line_p1_on_arc.iter().any(|c| c.line == r && c.arc == a),
                    GrabTarget::LineP2(r) => self.sketch.line_p2_on_arc.iter().any(|c| c.line == r && c.arc == a),
                    GrabTarget::ArcCenter(src) => arc_point_on_arc(src, ArcPoint::Center),
                    GrabTarget::ArcStart(src) => arc_point_on_arc(src, ArcPoint::Start),
                    GrabTarget::ArcEnd(src) => arc_point_on_arc(src, ArcPoint::End),
                    _ => false,
                };
                direct
                    || self.are_transitively_coincident(grab_sel, Selection::ArcStart(a))
                    || self.are_transitively_coincident(grab_sel, Selection::ArcEnd(a))
                    || self.selections_at_arc_midpoint(a).iter()
                        .any(|s| self.are_transitively_coincident(grab_sel, *s))
            }
            SnapTarget::LineMidpoint(l) => {
                // Anchored to `l`'s midpoint iff transitively coincident
                // with any entity directly constrained there.
                self.selections_at_line_midpoint(l).iter()
                    .any(|s| self.are_transitively_coincident(grab_sel, *s))
            }
            SnapTarget::ArcMidpoint(a) => {
                self.selections_at_arc_midpoint(a).iter()
                    .any(|s| self.are_transitively_coincident(grab_sel, *s))
            }
            _ => false,
        }
    }

    // Check if a coincident constraint already exists (direct or transitive)
    /// True when the dragged endpoint of `line` (the P1 end if
    /// `is_p1`, otherwise the P2 end) is already coincident with,
    /// lying on, or tied to the midpoint of some other entity.
    /// Used to suppress the auto-H/V drag hint when dragging an
    /// interior vertex of a segmented structure -- the endpoint
    /// can't actually move in isolation there, so offering an
    /// axis-snap would just be noise.
    fn line_endpoint_has_connection(&self, line: Ref<Line>, is_p1: bool) -> bool {
        let s = &self.sketch;
        if is_p1 {
            if s.coincident_lp1.iter().any(|c| c.line == line) { return true; }
            if s.coincident_ll11.iter().any(|c| c.a == line || c.b == line) { return true; }
            if s.coincident_ll12.iter().any(|c| c.a == line) { return true; }
            if s.coincident_ll21.iter().any(|c| c.b == line) { return true; }
            if s.line_p1_on_line.iter().any(|c| c.a == line) { return true; }
            if s.coincident_lp1_arc_center.iter().any(|c| c.line == line) { return true; }
            if s.coincident_lp1_arc_start.iter().any(|c| c.line == line) { return true; }
            if s.coincident_lp1_arc_end.iter().any(|c| c.line == line) { return true; }
            if s.line_p1_on_arc.iter().any(|c| c.line == line) { return true; }
            if s.midpoint_lp1.iter().any(|c| c.line == line) { return true; }
            if s.midpoint_lp1_arc.iter().any(|c| c.line == line) { return true; }
        } else {
            if s.coincident_lp2.iter().any(|c| c.line == line) { return true; }
            if s.coincident_ll22.iter().any(|c| c.a == line || c.b == line) { return true; }
            if s.coincident_ll12.iter().any(|c| c.b == line) { return true; }
            if s.coincident_ll21.iter().any(|c| c.a == line) { return true; }
            if s.line_p2_on_line.iter().any(|c| c.a == line) { return true; }
            if s.coincident_lp2_arc_center.iter().any(|c| c.line == line) { return true; }
            if s.coincident_lp2_arc_start.iter().any(|c| c.line == line) { return true; }
            if s.coincident_lp2_arc_end.iter().any(|c| c.line == line) { return true; }
            if s.line_p2_on_arc.iter().any(|c| c.line == line) { return true; }
            if s.midpoint_lp2.iter().any(|c| c.line == line) { return true; }
            if s.midpoint_lp2_arc.iter().any(|c| c.line == line) { return true; }
        }
        false
    }

    fn has_existing_coincident_line(&self, line: Ref<Line>, is_p1: bool, snap: SnapTarget) -> bool {
        if let Some(snap_sel) = Self::snap_to_selection(snap) {
            let line_sel = if is_p1 { Selection::LineP1(line) } else { Selection::LineP2(line) };
            self.are_transitively_coincident(line_sel, snap_sel)
        } else {
            // Check direct constraints for Line/ArcBody/midpoint snap targets
            match (snap, is_p1) {
                (SnapTarget::ArcBody(arc), true) => self.sketch.line_p1_on_arc.iter().any(|c| c.line == line && c.arc == arc),
                (SnapTarget::ArcBody(arc), false) => self.sketch.line_p2_on_arc.iter().any(|c| c.line == line && c.arc == arc),
                (SnapTarget::Line(other), true) => self.sketch.line_p1_on_line.iter().any(|c| c.a == line && c.b == other),
                (SnapTarget::Line(other), false) => self.sketch.line_p2_on_line.iter().any(|c| c.a == line && c.b == other),
                (SnapTarget::LineMidpoint(other), true) => self.sketch.midpoint_lp1.iter().any(|c| c.line == line && c.target == other),
                (SnapTarget::LineMidpoint(other), false) => self.sketch.midpoint_lp2.iter().any(|c| c.line == line && c.target == other),
                (SnapTarget::ArcMidpoint(arc), true) => self.sketch.midpoint_lp1_arc.iter().any(|c| c.line == line && c.arc == arc),
                (SnapTarget::ArcMidpoint(arc), false) => self.sketch.midpoint_lp2_arc.iter().any(|c| c.line == line && c.arc == arc),
                _ => false,
            }
        }
    }

    // Get position of a DimensionEndpoint
    pub fn dim_endpoint_pos(&self, ep: &DimensionEndpoint) -> vect2d {
        match *ep {
            DimensionEndpoint::Point(r) => self.sketch.points[r].pos.value,
            DimensionEndpoint::LineP1(r) => self.sketch.lines[r].p1.value,
            DimensionEndpoint::LineP2(r) => self.sketch.lines[r].p2.value,
            DimensionEndpoint::ArcCenter(r) => self.sketch.arcs[r].center.value,
            DimensionEndpoint::ArcStart(r) => arc_start_pos(&self.sketch.arcs[r]),
            DimensionEndpoint::ArcEnd(r) => arc_end_pos(&self.sketch.arcs[r]),
        }
    }

    // Measure the current value for a dimension kind
    fn measure_dimension(&self, kind: &DimensionKind) -> f64 {
        match kind {
            DimensionKind::LineLength(r) => {
                let l = &self.sketch.lines[*r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                (dx * dx + dy * dy).sqrt()
            }
            DimensionKind::PointPointDistance(a, b) => {
                let pa = self.dim_endpoint_pos(a);
                let pb = self.dim_endpoint_pos(b);
                let dx = pb.x - pa.x;
                let dy = pb.y - pa.y;
                (dx * dx + dy * dy).sqrt()
            }
            DimensionKind::PointLineDistance(pt, line) => {
                let p = self.dim_endpoint_pos(pt);
                let l = &self.sketch.lines[*line];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-12 { return 0.0; }
                (((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx) / len).abs()
            }
            DimensionKind::ArcRadius(r) => {
                self.sketch.arcs[*r].radius.value
            }
            DimensionKind::ArcRadiusB(r) => {
                self.sketch.arcs[*r].radius_b.value
            }
            DimensionKind::ArcSweep(r) => {
                let a = &self.sketch.arcs[*r];
                rad2deg((a.end_angle.value - a.start_angle.value).abs())
            }
            DimensionKind::ArcRotation(r) => {
                rad2deg(self.sketch.arcs[*r].rotation.value)
            }
            DimensionKind::Angle(a, b, supplement) => {
                let la = &self.sketch.lines[*a];
                let lb = &self.sketch.lines[*b];
                let dx1 = la.p2.value.x - la.p1.value.x;
                let dy1 = la.p2.value.y - la.p1.value.y;
                let dx2 = lb.p2.value.x - lb.p1.value.x;
                let dy2 = lb.p2.value.y - lb.p1.value.y;
                let cross = dx1 * dy2 - dy1 * dx2;
                let dot = dx1 * dx2 + dy1 * dy2;
                let angle_rad = cross.atan2(dot).abs();
                
                if *supplement { 180.0 - rad2deg(angle_rad) } else { rad2deg(angle_rad) }
            }
            DimensionKind::HDistance(a, b) => {
                let pa = self.dim_endpoint_pos(a);
                let pb = self.dim_endpoint_pos(b);
                (pa.x - pb.x).abs()
            }
            DimensionKind::VDistance(a, b) => {
                let pa = self.dim_endpoint_pos(a);
                let pb = self.dim_endpoint_pos(b);
                (pa.y - pb.y).abs()
            }
            DimensionKind::LineAngle(r) => {
                let l = &self.sketch.lines[*r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                rad2deg(dy.atan2(dx))
            }
            DimensionKind::ConcentricDistance(a, b) => {
                let ra = self.sketch.arcs[*a].radius.value;
                let rb = self.sketch.arcs[*b].radius.value;
                (rb - ra).abs()
            }
            DimensionKind::LineLineDistance(a, b) => {
                let la = &self.sketch.lines[*a];
                let lb = &self.sketch.lines[*b];
                let dx = la.p2.value.x - la.p1.value.x;
                let dy = la.p2.value.y - la.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-12 { return 0.0; }
                (((lb.p1.value.x - la.p1.value.x) * dy
                - (lb.p1.value.y - la.p1.value.y) * dx) / len).abs()
            }
        }
    }

    /// Build the input-box display string for editing a dimension.
    /// Range dims round-trip to their source syntax (`>= 2`, `2 to 6`,
    /// `low to high`); expression dims show their source; numeric
    /// dims show the current value.
    fn dim_edit_string(dim: &Dimension) -> String {
        if let Some(rb) = &dim.range {
            return match rb {
                RangeBound::Min(v) => format!(">= {}", v),
                RangeBound::Max(v) => format!("<= {}", v),
                RangeBound::Between(lo, hi) => format!("{} to {}", lo, hi),
            };
        }
        if let Some(expr) = &dim.expr_str {
            return expr.clone();
        }
        format!("{:.4}", dim.value)
    }

    // Convert a Selection to a DimensionEndpoint (for point-like selections)
    fn selection_to_dim_endpoint(sel: &Selection) -> Option<DimensionEndpoint> {
        match *sel {
            Selection::Point(r) => Some(DimensionEndpoint::Point(r)),
            Selection::LineP1(r) => Some(DimensionEndpoint::LineP1(r)),
            Selection::LineP2(r) => Some(DimensionEndpoint::LineP2(r)),
            Selection::ArcCenter(r) => Some(DimensionEndpoint::ArcCenter(r)),
            Selection::ArcStart(r) => Some(DimensionEndpoint::ArcStart(r)),
            Selection::ArcEnd(r) => Some(DimensionEndpoint::ArcEnd(r)),
            _ => None,
        }
    }

    /// For two point-like endpoints plus the current mouse position,
    /// pick PointPointDistance / HDistance / VDistance based on where
    /// the mouse sits relative to the two points' bounding box. Dead
    /// zone: an inflated bbox (+/- 15 screen pixels on each axis);
    /// inside this zone -> PointPointDistance. Outside, pick H or V
    /// by whichever axis the mouse is farther outside the bbox on:
    /// more up/down than left/right -> HDistance, more left/right
    /// than up/down -> VDistance. This covers the "diagonal" sectors
    /// too -- not only the strict horizontal/vertical strips over
    /// the bbox edges.
    /// Called on the frame of the second-entity click to seed the dim
    /// kind, and re-called each preview frame while placing so the
    /// dimension type tracks the mouse zone.
    fn pick_point_pair_dim_kind(
        &self,
        a_ep: DimensionEndpoint,
        b_ep: DimensionEndpoint,
        mouse: Option<vect2d>,
    ) -> DimensionKind {
        let Some(m) = mouse else {
            return DimensionKind::PointPointDistance(a_ep, b_ep);
        };
        let pa = self.dim_endpoint_pos(&a_ep);
        let pb = self.dim_endpoint_pos(&b_ep);
        let min_x = pa.x.min(pb.x);
        let max_x = pa.x.max(pb.x);
        let min_y = pa.y.min(pb.y);
        let max_y = pa.y.max(pb.y);
        // 15 screen pixels converted to sketch units. `self.scale` is
        // pixels per sketch unit (see to_screen); guard against zero.
        let t = if self.scale > 1e-6 { 15.0_f64 / self.scale as f64 } else { 0.0 };
        // Signed distance outside the bbox per axis; 0 if inside.
        let dx_out = (m.x - max_x).max(min_x - m.x).max(0.0);
        let dy_out = (m.y - max_y).max(min_y - m.y).max(0.0);
        if dx_out < t && dy_out < t {
            DimensionKind::PointPointDistance(a_ep, b_ep)
        } else if dy_out > dx_out {
            DimensionKind::HDistance(a_ep, b_ep)
        } else {
            DimensionKind::VDistance(a_ep, b_ep)
        }
    }

    /// Same zone-based dispatch as `pick_point_pair_dim_kind` but for
    /// a single line: inside the dead-zone the natural measurement
    /// is `LineLength` (oblique along the line), and outside the
    /// band H/V-distance between the two endpoints takes over.
    fn pick_line_dim_kind(&self, line: Ref<Line>, mouse: Option<vect2d>) -> DimensionKind {
        let a_ep = DimensionEndpoint::LineP1(line);
        let b_ep = DimensionEndpoint::LineP2(line);
        match self.pick_point_pair_dim_kind(a_ep, b_ep, mouse) {
            DimensionKind::PointPointDistance(_, _) => DimensionKind::LineLength(line),
            other => other,
        }
    }

    /// Extract the "base line" of a dim kind when it is either a
    /// `LineLength` or an H/V-distance between the two endpoints of
    /// the same line. Used by the Phase 2 preview to keep re-picking
    /// between LineLength / HDistance / VDistance as the mouse moves,
    /// without losing which line was originally selected.
    fn line_dim_base_line(kind: &DimensionKind) -> Option<Ref<Line>> {
        match *kind {
            DimensionKind::LineLength(r) => Some(r),
            DimensionKind::HDistance(
                DimensionEndpoint::LineP1(r1),
                DimensionEndpoint::LineP2(r2),
            ) if r1 == r2 => Some(r1),
            DimensionKind::HDistance(
                DimensionEndpoint::LineP2(r1),
                DimensionEndpoint::LineP1(r2),
            ) if r1 == r2 => Some(r1),
            DimensionKind::VDistance(
                DimensionEndpoint::LineP1(r1),
                DimensionEndpoint::LineP2(r2),
            ) if r1 == r2 => Some(r1),
            DimensionKind::VDistance(
                DimensionEndpoint::LineP2(r1),
                DimensionEndpoint::LineP1(r2),
            ) if r1 == r2 => Some(r1),
            _ => None,
        }
    }

    // Try to determine DimensionKind from current selection
    fn selection_to_dim_kind(&self, mouse: Option<vect2d>) -> Option<DimensionKind> {
        let sel = &self.selection;
        if sel.len() == 1 {
            match sel[0] {
                Selection::Line(r) => return Some(self.pick_line_dim_kind(r, mouse)),
                Selection::Arc(r) => {
                    let a = &self.sketch.arcs[r];
                    if a.is_ellipse
                        && let Some(m) = mouse {
                            // Project mouse onto major/minor axes to pick radius or radius_b
                            let dx = m.x - a.center.value.x;
                            let dy = m.y - a.center.value.y;
                            let rot = a.rotation.value;
                            // Component along major axis vs minor axis
                            let major = (dx * rot.cos() + dy * rot.sin()).abs();
                            let minor = (-dx * rot.sin() + dy * rot.cos()).abs();
                            if minor > major {
                                return Some(DimensionKind::ArcRadiusB(r));
                            }
                        }
                    return Some(DimensionKind::ArcRadius(r));
                }
                _ => {}
            }
        }
        if sel.len() == 2 {
            // Two lines: parallel-within-tolerance -> line-line distance,
            // otherwise angle dimension. User click precision is coarse,
            // so ~3 degrees is the cutoff; a false-positive parallel is
            // harmless because the caller also applies a Parallel
            // constraint that snaps the lines exactly parallel.
            if let (Selection::Line(a), Selection::Line(b)) = (sel[0], sel[1]) {
                let la = &self.sketch.lines[a];
                let lb = &self.sketch.lines[b];
                let dax = la.p2.value.x - la.p1.value.x;
                let day = la.p2.value.y - la.p1.value.y;
                let dbx = lb.p2.value.x - lb.p1.value.x;
                let dby = lb.p2.value.y - lb.p1.value.y;
                let alen = (dax * dax + day * day).sqrt();
                let blen = (dbx * dbx + dby * dby).sqrt();
                if alen > 1e-12 && blen > 1e-12 {
                    let sin_theta = (dax * dby - day * dbx).abs() / (alen * blen);
                    if sin_theta < PARALLEL_SELECTION_TOL {
                        return Some(DimensionKind::LineLineDistance(a, b));
                    }
                }
                return Some(DimensionKind::Angle(a, b, false));
            }
            // Two arcs -> radial distance, offered when both are
            // circular and their centers currently coincide (within
            // epsilon). A prior `Concentric` constraint is no longer
            // required -- `ConcentricDistance` is self-contained and
            // enforces its own center-coincidence; the dim-placement
            // path still emits `ApplyConcentric` for visibility.
            if let (Selection::Arc(a), Selection::Arc(b)) = (sel[0], sel[1])
                && a != b
                && !self.sketch.arcs[a].is_ellipse
                && !self.sketch.arcs[b].is_ellipse
            {
                let ca = self.sketch.arcs[a].center.value;
                let cb = self.sketch.arcs[b].center.value;
                let dx = ca.x - cb.x;
                let dy = ca.y - cb.y;
                if (dx * dx + dy * dy).sqrt() < 1e-3 {
                    return Some(DimensionKind::ConcentricDistance(a, b));
                }
            }
            // Point + Line -> point-line distance
            let point_ep = sel.iter().find_map(Self::selection_to_dim_endpoint);
            let line_ref = sel.iter().find_map(|s| if let Selection::Line(r) = s { Some(*r) } else { None });
            if let (Some(ep), Some(line)) = (point_ep, line_ref) {
                return Some(DimensionKind::PointLineDistance(ep, line));
            }
            // Two point-like -> point-point / H-distance / V-distance
            // depending on where the mouse sits relative to the two
            // points (see `pick_point_pair_dim_kind`).
            let ep_a = Self::selection_to_dim_endpoint(&sel[0]);
            let ep_b = Self::selection_to_dim_endpoint(&sel[1]);
            if let (Some(a), Some(b)) = (ep_a, ep_b) {
                return Some(self.pick_point_pair_dim_kind(a, b, mouse));
            }
        }
        if sel.len() == 3 {
            // ArcCenter + ArcStart + ArcEnd of same arc -> sweep dimension
            let mut arc_ref = None;
            let mut has_center = false;
            let mut has_start = false;
            let mut has_end = false;
            for s in sel {
                match s {
                    Selection::ArcCenter(r) => { arc_ref = Some(*r); has_center = true; }
                    Selection::ArcStart(r) => { arc_ref = Some(*r); has_start = true; }
                    Selection::ArcEnd(r) => { arc_ref = Some(*r); has_end = true; }
                    _ => {}
                }
            }
            if has_center && has_start && has_end {
                // Verify all from same arc
                let all_same = sel.iter().all(|s| match s {
                    Selection::ArcCenter(r) | Selection::ArcStart(r) | Selection::ArcEnd(r) => Some(*r) == arc_ref,
                    _ => false,
                });
                if all_same
                    && let Some(r) = arc_ref {
                        return Some(DimensionKind::ArcSweep(r));
                    }
            }
        }
        None
    }

    // Compute perpendicular offset from a sketch point to the measurement line of a dimension kind
    #[allow(dead_code)]
    fn compute_dim_offset(&self, kind: &DimensionKind, mouse_sketch: vect2d) -> vect2d {
        let (p1, p2) = self.dim_endpoints(kind);
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-12);
        let nx = -dy / len;
        let ny = dx / len;
        // Project mouse onto the normal direction
        let offset_perp = (mouse_sketch.x - (p1.x + p2.x) / 2.0) * nx
                        + (mouse_sketch.y - (p1.y + p2.y) / 2.0) * ny;
        vect2d::new(0.0, offset_perp)
    }

    // Get the two sketch-space endpoints for a dimension kind
    pub fn dim_endpoints(&self, kind: &DimensionKind) -> (vect2d, vect2d) {
        match kind {
            DimensionKind::LineLength(r) => {
                let l = &self.sketch.lines[*r];
                (l.p1.value, l.p2.value)
            }
            DimensionKind::PointPointDistance(a, b) => {
                (self.dim_endpoint_pos(a), self.dim_endpoint_pos(b))
            }
            DimensionKind::PointLineDistance(pt, line) => {
                let p = self.dim_endpoint_pos(pt);
                let l = &self.sketch.lines[*line];
                // Project onto infinite line (not clamped to segment)
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len2 = dx * dx + dy * dy;
                let foot = if len2 < 1e-12 { l.p1.value } else {
                    let t = ((p.x - l.p1.value.x) * dx + (p.y - l.p1.value.y) * dy) / len2;
                    vect2d::new(l.p1.value.x + t * dx, l.p1.value.y + t * dy)
                };
                (p, foot)
            }
            DimensionKind::ArcRadius(r) | DimensionKind::ArcRadiusB(r) => {
                let a = &self.sketch.arcs[*r];
                let is_b = matches!(kind, DimensionKind::ArcRadiusB(_));
                let rv = if is_b { a.radius_b.value } else { a.radius.value };
                let angle = if a.is_ellipse {
                    if is_b { a.rotation.value + std::f64::consts::FRAC_PI_2 }
                    else { a.rotation.value }
                } else { 0.0 };
                let edge = vect2d::new(
                    a.center.value.x + rv * angle.cos(),
                    a.center.value.y + rv * angle.sin(),
                );
                (a.center.value, edge)
            }
            DimensionKind::ArcSweep(r) => {
                let a = &self.sketch.arcs[*r];
                let sa = a.start_angle.value;
                let ea = a.end_angle.value;
                let rad = a.radius.value;
                let p1 = vect2d::new(a.center.value.x + rad * sa.cos(), a.center.value.y + rad * sa.sin());
                let p2 = vect2d::new(a.center.value.x + rad * ea.cos(), a.center.value.y + rad * ea.sin());
                (p1, p2)
            }
            DimensionKind::ArcRotation(r) => {
                // Two endpoints along the major axis at the current
                // rotation, from center out to the +major-axis edge.
                let a = &self.sketch.arcs[*r];
                let rot = a.rotation.value;
                let edge = vect2d::new(
                    a.center.value.x + a.radius.value * rot.cos(),
                    a.center.value.y + a.radius.value * rot.sin(),
                );
                (a.center.value, edge)
            }
            DimensionKind::Angle(a, b, _) => {
                // Return midpoints of both lines (for hit testing fallback)
                let la = &self.sketch.lines[*a];
                let lb = &self.sketch.lines[*b];
                let ma = vect2d::new((la.p1.value.x + la.p2.value.x) / 2.0,
                                     (la.p1.value.y + la.p2.value.y) / 2.0);
                let mb = vect2d::new((lb.p1.value.x + lb.p2.value.x) / 2.0,
                                     (lb.p1.value.y + lb.p2.value.y) / 2.0);
                (ma, mb)
            }
            DimensionKind::HDistance(a, b) | DimensionKind::VDistance(a, b) => {
                (self.dim_endpoint_pos(a), self.dim_endpoint_pos(b))
            }
            DimensionKind::LineAngle(r) => {
                let l = &self.sketch.lines[*r];
                (l.p1.value, l.p2.value)
            }
            DimensionKind::ConcentricDistance(a, b) => {
                // Two endpoints on the line from the shared center along
                // the leader direction (offset), one at each radius.
                let center = self.sketch.arcs[*a].center.value;
                let ra = self.sketch.arcs[*a].radius.value;
                let rb = self.sketch.arcs[*b].radius.value;
                // Direction comes from the offset; default to +x if zero.
                let p1 = vect2d::new(center.x + ra, center.y);
                let p2 = vect2d::new(center.x + rb, center.y);
                (p1, p2)
            }
            DimensionKind::LineLineDistance(a, b) => {
                // Anchor at line A's midpoint; project perpendicularly onto
                // line B's infinite line. Once the Parallel constraint is
                // active this gives an arrow pair that straddles the gap.
                let la = &self.sketch.lines[*a];
                let lb = &self.sketch.lines[*b];
                let ma = vect2d::new((la.p1.value.x + la.p2.value.x) / 2.0,
                                     (la.p1.value.y + la.p2.value.y) / 2.0);
                let dx = lb.p2.value.x - lb.p1.value.x;
                let dy = lb.p2.value.y - lb.p1.value.y;
                let len2 = dx * dx + dy * dy;
                let foot = if len2 < 1e-12 { lb.p1.value } else {
                    let t = ((ma.x - lb.p1.value.x) * dx + (ma.y - lb.p1.value.y) * dy) / len2;
                    vect2d::new(lb.p1.value.x + t * dx, lb.p1.value.y + t * dy)
                };
                (ma, foot)
            }
        }
    }

    // Start a new undo group. All exec() calls until the next begin_group share the same group.
    pub fn begin_group(&mut self) {
        self.history.begin_group();
    }

    /// Kick off a corner op (fillet or chamfer) from the GUI: snapshot
    /// the pre-op sketch, record the first corner, apply it with 10 %
    /// of the shortest involved line's length as the starting
    /// radius/distance, then drop the user straight into value
    /// editing. Additional corners can be toggled in/out while
    /// editing. `command` is the backend command name the reapply
    /// loop runs.
    fn try_start_gui_fillet(&mut self, arg: &str, shortest_len: f64) {
        self.try_start_gui_corner_op("fillet", arg, shortest_len);
    }

    fn try_start_gui_chamfer(&mut self, arg: &str, shortest_len: f64) {
        self.try_start_gui_corner_op("chamfer", arg, shortest_len);
    }

    fn try_start_gui_corner_op(&mut self, command: &'static str, arg: &str, shortest_len: f64) {
        if shortest_len < 1e-6 { return; }
        let pre_snapshot = match bincode::serialize(&self.sketch) {
            Ok(s) => s,
            Err(_) => return,
        };
        let history_cursor_before = self.history.cursor;
        let initial_r = format!("{:.4}", shortest_len * 0.1);
        self.fillet_pending = Some(FilletPending {
            command,
            pre_snapshot,
            history_cursor_before,
            corners: vec![arg.to_string()],
            last_valid_radius: initial_r.clone(),
            last_applied_sig: String::new(),
        });
        // Pre-populate the dim input so Enter on an empty edit still
        // commits the 10 % mock. Reapply runs below; it creates the
        // primary dim, then we hook the dim-edit overlay up to it.
        self.dim_input = initial_r;
        self.reapply_fillets();
        // If reapply couldn't produce a primary dim (corner invalid),
        // bail and surface the error.
        if self.primary_fillet_dim_index().is_none() {
            self.cancel_pending_fillet();
            return;
        }
        self.dim_editing = true;
        self.dim_edit_index = self.primary_fillet_dim_index();
        self.dim_kind = None;
        self.dim_placing = false;
        self.dim_select_all = true;
        self.dim_derived = false;
        self.dim_derived_prev = false;
        self.dim_input_backup.clear();
        self.selection.clear();
    }

    /// Index of the radius dimension created by the first (primary)
    /// fillet in `fillet_pending`. None if reapply has never run or
    /// the primary fillet failed.
    pub fn primary_fillet_dim_index(&self) -> Option<usize> {
        // The primary fillet is the first entry in `corners`. After
        // reapply, its AddDimension is the oldest fillet-created dim;
        // scan by name prefix isn't reliable here, so find the first
        // dim whose kind is ArcRadius(arc) where arc is filleted.
        // Simpler: pick the first ArcRadius dim whose index >= the
        // pre-snapshot dim count. Use last-applied signature to know
        // how many dims were pre-existing.
        let p = self.fillet_pending.as_ref()?;
        // Deserialize pre_snapshot once to learn how many dims existed
        // before the fillet session started. Cheap vs. every-frame.
        let pre = bincode::deserialize::<Sketch>(&p.pre_snapshot).ok()?;
        let n_pre = pre.dimensions.len();
        // First dim added by fillet session (the primary's radius).
        if self.sketch.dimensions.len() > n_pre {
            Some(n_pre)
        } else {
            None
        }
    }

    /// Current radius token for reapply: the user's dim_input if it
    /// parses to something usable, otherwise the pending's last
    /// valid radius so the canvas keeps showing the most recent
    /// feasible fillet while the user is mid-edit.
    fn fillet_effective_radius(&self) -> Option<String> {
        let p = self.fillet_pending.as_ref()?;
        let typed = self.dim_input.trim();
        // Empty, "0", or unparseable -> fall back.
        if typed.is_empty() {
            return Some(p.last_valid_radius.clone());
        }
        // Strip snapshot prefix for parsing, but keep it in the
        // resubmitted token so the dim stores the literal snapshot.
        let parse_src = typed.strip_prefix('=').unwrap_or(typed).trim();
        if let Ok(v) = parse_src.parse::<f64>() {
            if v <= 0.0 {
                return Some(p.last_valid_radius.clone());
            }
            return Some(typed.to_string());
        }
        if arael_sym::parse(parse_src).is_ok()
            && arael_sketch_backend::commands::eval_expr(&self.sketch, parse_src)
                .map(|v| v > 0.0)
                .unwrap_or(false)
        {
            return Some(typed.to_string());
        }
        Some(p.last_valid_radius.clone())
    }

    /// Restore the pre-fillet sketch and reapply every pending
    /// corner from scratch. Called whenever the radius or the
    /// corner list changes during an active fillet edit. Cheap
    /// enough for live typing (one bincode deserialize + a handful
    /// of `fillet` command runs).
    pub fn reapply_fillets(&mut self) {
        let Some(p) = self.fillet_pending.as_ref() else { return; };
        let radius = match self.fillet_effective_radius() {
            Some(r) => r,
            None => return,
        };
        let sig = format!("{}|{}", radius, p.corners.join(","));
        if sig == p.last_applied_sig { return; }

        let pre_snapshot = p.pre_snapshot.clone();
        let history_cursor_before = p.history_cursor_before;
        let corners = p.corners.clone();

        // Restore pre-fillet state so the reapply is deterministic.
        if let Ok(s) = bincode::deserialize::<Sketch>(&pre_snapshot) {
            self.sketch = s.into();
        }
        self.history.actions.truncate(history_cursor_before);
        self.history.snapshots.truncate(history_cursor_before);
        self.history.cursors.truncate(history_cursor_before);
        self.history.groups.truncate(history_cursor_before);
        self.history.cursor = history_cursor_before;
        self.status_error = None;

        // Run primary corner op. Its dim name drives the rest.
        let command = self.fillet_pending.as_ref().map(|p| p.command).unwrap_or("fillet");
        let mut primary_dim_name: Option<String> = None;
        let mut applied_any = false;
        let mut first_result_radius = radius.clone();
        if let Some(first) = corners.first() {
            let cmd = format!("{} {} {}", command, first, radius);
            let results = self.run_commands(&cmd);
            let ok = !results.iter().any(|r| r.is_error)
                && self.sketch.dimensions.last().is_some();
            if ok {
                applied_any = true;
                primary_dim_name = self.sketch.dimensions.last().map(|d| d.name.clone());
            } else {
                // Primary failed: surface the error and leave
                // last_applied_sig empty so the next typed change
                // retries.
                if let Some(r) = results.iter().find(|r| r.is_error) {
                    self.status_error = Some(r.output.clone());
                }
                first_result_radius.clear();
            }
        }

        // Secondary corner ops reference the primary dim by name.
        if let Some(pdn) = &primary_dim_name {
            for corner in corners.iter().skip(1) {
                let cmd = format!("{} {} {}", command, corner, pdn);
                let results = self.run_commands(&cmd);
                if results.iter().any(|r| r.is_error) {
                    // Leave this corner failed; the others still stick.
                }
            }
        }

        // Collapse every action pushed in this reapply into a
        // single undo group. `cmd_fillet` opens its own group per
        // call, so without this stitching a 3-corner fillet tool
        // session would need three Ctrl+Z presses to undo.
        if self.history.cursor > history_cursor_before
            && let Some(&first) = self.history.groups.get(history_cursor_before)
        {
            for g in &mut self.history.groups[history_cursor_before..self.history.cursor] {
                *g = first;
            }
        }

        if let Some(pending) = self.fillet_pending.as_mut() {
            if applied_any && !first_result_radius.is_empty() {
                pending.last_valid_radius = first_result_radius;
            }
            pending.last_applied_sig = sig;
        }
        self.compute_dof_async();
    }

    /// Add a corner to the active fillet session, or remove it if
    /// it's already there. Reapplies so the preview updates.
    pub fn toggle_fillet_corner(&mut self, arg: &str) {
        if let Some(p) = self.fillet_pending.as_mut() {
            if let Some(idx) = p.corners.iter().position(|c| c == arg) {
                p.corners.remove(idx);
            } else {
                p.corners.push(arg.to_string());
            }
        }
        // If we removed the last corner, drop the pending and bail.
        if self.fillet_pending.as_ref().is_some_and(|p| p.corners.is_empty()) {
            self.cancel_pending_fillet();
            return;
        }
        self.reapply_fillets();
        // dim_edit_index may have shifted if dims were re-numbered.
        self.dim_edit_index = self.primary_fillet_dim_index();
    }

    /// Restore the sketch and history to the state captured at
    /// fillet start. Called from the Escape handler when a
    /// FilletPending is live, or when all corners are removed mid
    /// edit.
    pub fn cancel_pending_fillet(&mut self) {
        let Some(p) = self.fillet_pending.take() else { return; };
        if let Ok(s) = bincode::deserialize::<Sketch>(&p.pre_snapshot) {
            self.sketch = s.into();
        }
        self.history.actions.truncate(p.history_cursor_before);
        self.history.snapshots.truncate(p.history_cursor_before);
        self.history.cursors.truncate(p.history_cursor_before);
        self.history.groups.truncate(p.history_cursor_before);
        self.history.cursor = p.history_cursor_before;
        self.dim_editing = false;
        self.dim_edit_index = None;
        self.dim_kind = None;
        self.dim_input.clear();
        self.status_error = None;
        self.compute_dof_async();
    }

    fn apply_horizontal(&mut self) {
        self.begin_group();
        let lines: Vec<Ref<Line>> = self.selection.iter().filter_map(|s| {
            if let Selection::Line(r) = s { Some(*r) } else { None }
        }).collect();
        if !lines.is_empty() {
            let action = Action::ApplyHorizontal { lines };
            if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                self.status_error = Some(err);
                return;
            }
            self.exec(action);
        }
    }

    fn apply_vertical(&mut self) {
        self.begin_group();
        let lines: Vec<Ref<Line>> = self.selection.iter().filter_map(|s| {
            if let Selection::Line(r) = s { Some(*r) } else { None }
        }).collect();
        if !lines.is_empty() {
            let action = Action::ApplyVertical { lines };
            if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                self.status_error = Some(err);
                return;
            }
            self.exec(action);
        }
    }

    // Get a human-readable name for a selection
    fn selection_name(&self, sel: Selection) -> String {
        match sel {
            Selection::Point(r) => self.sketch.points[r].name.clone(),
            Selection::Line(r) => self.sketch.lines[r].name.clone(),
            Selection::LineP1(r) => format!("{}.p1", self.sketch.lines[r].name),
            Selection::LineP2(r) => format!("{}.p2", self.sketch.lines[r].name),
            Selection::Arc(r) => self.sketch.arcs[r].name.clone(),
            Selection::ArcCenter(r) => format!("{}.c", self.sketch.arcs[r].name),
            Selection::ArcStart(r) => format!("{}.s", self.sketch.arcs[r].name),
            Selection::ArcEnd(r) => format!("{}.e", self.sketch.arcs[r].name),
            Selection::Constraint(_) => "constraint".to_string(),
            Selection::Dimension(i) => {
                if i < self.sketch.dimensions.len() {
                    self.sketch.dimensions[i].name.clone()
                } else {
                    "dim?".to_string()
                }
            }
        }
    }

    // Get position of a point-like selection
    fn selection_pos(&self, sel: Selection) -> Option<vect2d> {
        match sel {
            Selection::Point(r) => Some(self.sketch.points[r].pos.value),
            Selection::LineP1(r) => Some(self.sketch.lines[r].p1.value),
            Selection::LineP2(r) => Some(self.sketch.lines[r].p2.value),
            Selection::ArcCenter(r) => Some(self.sketch.arcs[r].center.value),
            Selection::ArcStart(r) => Some(arc_start_pos(&self.sketch.arcs[r])),
            Selection::ArcEnd(r) => Some(arc_end_pos(&self.sketch.arcs[r])),
            Selection::Line(_) | Selection::Arc(_) | Selection::Constraint(_) | Selection::Dimension(_) => None,
        }
    }

    // Create a coincident constraint between a point-like selection and a helper point.
    // Returns the Action, or None if the selection is not a point-like entity.
    fn coincident_action_to_point(sel: Selection, point: Ref<Point>) -> Option<Action> {
        match sel {
            Selection::Point(r) => Some(Action::ApplyCoincidentPP { a: point, b: r }),
            Selection::LineP1(r) => Some(Action::ApplyCoincidentLP1 { line: r, point }),
            Selection::LineP2(r) => Some(Action::ApplyCoincidentLP2 { line: r, point }),
            Selection::ArcCenter(r) => Some(Action::ApplyCoincidentArcCenter { point, arc: r }),
            Selection::ArcStart(r) => Some(Action::ApplyCoincidentArcStart { point, arc: r }),
            Selection::ArcEnd(r) => Some(Action::ApplyCoincidentArcEnd { point, arc: r }),
            Selection::Line(_) | Selection::Arc(_) | Selection::Constraint(_) | Selection::Dimension(_) => None,
        }
    }

    fn apply_coincident(&mut self) {
        self.begin_group();
        if self.selection.len() != 2 { return; }
        let (s0, s1) = (self.selection[0], self.selection[1]);

        // Check for transitive coincidence (already coincident through other constraints)
        if self.are_transitively_coincident(s0, s1) {
            let name_a = self.selection_name(s0);
            let name_b = self.selection_name(s1);
            self.status_error = Some(format!("{} and {} are already coincident", name_a, name_b));
            return;
        }

        // Point-on-line / endpoint-on-line: special cases (different constraint type)
        match (s0, s1) {
            (Selection::Point(point), Selection::Line(line))
            | (Selection::Line(line), Selection::Point(point)) => {
                self.exec(Action::ApplyPointOnLine { point, line });
                return;
            }
            (Selection::LineP1(a), Selection::Line(b))
            | (Selection::Line(b), Selection::LineP1(a)) => {
                self.exec(Action::ApplyLineP1OnLine { a, b });
                return;
            }
            (Selection::LineP2(a), Selection::Line(b))
            | (Selection::Line(b), Selection::LineP2(a)) => {
                self.exec(Action::ApplyLineP2OnLine { a, b });
                return;
            }
            // Arc endpoint on line
            (Selection::ArcCenter(src_arc), Selection::Line(line))
            | (Selection::Line(line), Selection::ArcCenter(src_arc)) => {
                self.exec(Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcCenter(src_arc), line });
                return;
            }
            (Selection::ArcStart(src_arc), Selection::Line(line))
            | (Selection::Line(line), Selection::ArcStart(src_arc)) => {
                self.exec(Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcStart(src_arc), line });
                return;
            }
            (Selection::ArcEnd(src_arc), Selection::Line(line))
            | (Selection::Line(line), Selection::ArcEnd(src_arc)) => {
                self.exec(Action::ApplyEndpointOnLine { endpoint: DimensionEndpoint::ArcEnd(src_arc), line });
                return;
            }
            // Point on arc/circle
            (Selection::Point(point), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyPointOnArc { point, arc });
                return;
            }
            // Line endpoint on arc/circle (direct constraint)
            (Selection::LineP1(line), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyLineP1OnArc { line, arc });
                return;
            }
            (Selection::LineP2(line), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyLineP2OnArc { line, arc });
                return;
            }
            // Arc endpoint on arc/circle
            (Selection::ArcCenter(src_arc), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::ArcCenter(src_arc)) => {
                self.exec(Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcCenter(src_arc), arc });
                return;
            }
            (Selection::ArcStart(src_arc), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::ArcStart(src_arc)) => {
                self.exec(Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcStart(src_arc), arc });
                return;
            }
            (Selection::ArcEnd(src_arc), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::ArcEnd(src_arc)) => {
                self.exec(Action::ApplyEndpointOnArc { endpoint: DimensionEndpoint::ArcEnd(src_arc), arc });
                return;
            }
            // Line-to-line (default: a.p2 == b.p1)
            (Selection::Line(a), Selection::Line(b)) => {
                let action = Action::ApplyCoincidentLL21 { a, b };
                if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                    self.status_error = Some(err);
                    return;
                }
                self.exec(action);
                return;
            }
            _ => {}
        }

        // For two point-like selections: try direct constraint first, fall back to helper point
        let pos = match self.selection_pos(s0) {
            Some(p) => p,
            None => return,
        };
        if self.selection_pos(s1).is_none() { return; }

        // Direct constraints for common cases (no helper point needed)
        match (s0, s1) {
            (Selection::Point(a), Selection::Point(b)) => {
                self.exec(Action::ApplyCoincidentPP { a, b });
                return;
            }
            (Selection::LineP1(a), Selection::LineP1(b)) => {
                self.exec(Action::ApplyCoincidentLL11 { a, b }); return;
            }
            (Selection::LineP1(a), Selection::LineP2(b)) => {
                self.exec(Action::ApplyCoincidentLL12 { a, b }); return;
            }
            (Selection::LineP2(a), Selection::LineP1(b)) => {
                self.exec(Action::ApplyCoincidentLL21 { a, b }); return;
            }
            (Selection::LineP2(a), Selection::LineP2(b)) => {
                self.exec(Action::ApplyCoincidentLL22 { a, b }); return;
            }
            (Selection::LineP1(line), Selection::Point(point))
            | (Selection::Point(point), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1 { line, point }); return;
            }
            (Selection::LineP2(line), Selection::Point(point))
            | (Selection::Point(point), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2 { line, point }); return;
            }
            (Selection::Point(point), Selection::ArcCenter(arc))
            | (Selection::ArcCenter(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyCoincidentArcCenter { point, arc }); return;
            }
            (Selection::Point(point), Selection::ArcStart(arc))
            | (Selection::ArcStart(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyCoincidentArcStart { point, arc }); return;
            }
            (Selection::Point(point), Selection::ArcEnd(arc))
            | (Selection::ArcEnd(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyCoincidentArcEnd { point, arc }); return;
            }
            (Selection::ArcCenter(a), Selection::ArcCenter(b)) => {
                let action = Action::ApplyConcentric { a, b };
                if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                    self.status_error = Some(err);
                    return;
                }
                self.exec(action); return;
            }
            // Line endpoint <-> Arc point (direct)
            (Selection::LineP1(line), Selection::ArcCenter(arc))
            | (Selection::ArcCenter(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1ArcCenter { line, arc }); return;
            }
            (Selection::LineP2(line), Selection::ArcCenter(arc))
            | (Selection::ArcCenter(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2ArcCenter { line, arc }); return;
            }
            (Selection::LineP1(line), Selection::ArcStart(arc))
            | (Selection::ArcStart(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1ArcStart { line, arc }); return;
            }
            (Selection::LineP2(line), Selection::ArcStart(arc))
            | (Selection::ArcStart(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2ArcStart { line, arc }); return;
            }
            (Selection::LineP1(line), Selection::ArcEnd(arc))
            | (Selection::ArcEnd(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1ArcEnd { line, arc }); return;
            }
            (Selection::LineP2(line), Selection::ArcEnd(arc))
            | (Selection::ArcEnd(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2ArcEnd { line, arc }); return;
            }
            // Arc-Arc endpoint (direct)
            (Selection::ArcCenter(a), Selection::ArcStart(b)) => {
                self.exec(Action::ApplyCoincidentArcCenterStart { a, b }); return;
            }
            (Selection::ArcStart(b), Selection::ArcCenter(a)) => {
                self.exec(Action::ApplyCoincidentArcCenterStart { a, b }); return;
            }
            (Selection::ArcCenter(a), Selection::ArcEnd(b)) => {
                self.exec(Action::ApplyCoincidentArcCenterEnd { a, b }); return;
            }
            (Selection::ArcEnd(b), Selection::ArcCenter(a)) => {
                self.exec(Action::ApplyCoincidentArcCenterEnd { a, b }); return;
            }
            (Selection::ArcStart(a), Selection::ArcStart(b)) => {
                self.exec(Action::ApplyCoincidentArcStartStart { a, b }); return;
            }
            (Selection::ArcStart(a), Selection::ArcEnd(b)) => {
                self.exec(Action::ApplyCoincidentArcStartEnd { a, b }); return;
            }
            (Selection::ArcEnd(a), Selection::ArcStart(b)) => {
                self.exec(Action::ApplyCoincidentArcEndStart { a, b }); return;
            }
            (Selection::ArcEnd(a), Selection::ArcEnd(b)) => {
                self.exec(Action::ApplyCoincidentArcEndEnd { a, b }); return;
            }
            _ => {}
        }

        // General case: create a helper point and constrain both selections to it
        let helper = self.sketch.get_mut().add_helper_point(pos);
        if let Some(action) = Self::coincident_action_to_point(s0, helper) {
            self.exec(action);
        }
        if let Some(action) = Self::coincident_action_to_point(s1, helper) {
            self.exec(action);
        }
    }

    fn apply_parallel(&mut self) {
        if self.selection.len() != 2 { return; }
        self.begin_group();
        // Line-Line, Arc-Line (either order), and Arc-Arc all map to
        // the Parallel tool. Circular arcs are rejected because their
        // rotation is Param::fixed(0.0) -- only ellipses have an
        // optimisable major-axis direction.
        let s0 = self.selection[0];
        let s1 = self.selection[1];
        let ellipse_ref = |r: Ref<Arc>| -> Option<Ref<Arc>> {
            self.sketch.arcs.get(r).and_then(|a| if a.is_ellipse { Some(r) } else { None })
        };
        let action = match (s0, s1) {
            (Selection::Line(a), Selection::Line(b)) => Some(Action::ApplyParallel { a, b }),
            (Selection::Arc(a), Selection::Line(l)) | (Selection::Line(l), Selection::Arc(a)) => {
                ellipse_ref(a).map(|arc| Action::ApplyArcLineParallel { arc, line: l })
            }
            (Selection::Arc(a), Selection::Arc(b)) => {
                match (ellipse_ref(a), ellipse_ref(b)) {
                    (Some(ea), Some(eb)) if ea != eb => Some(Action::ApplyArcArcParallel { a: ea, b: eb }),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(action) = action else {
            self.status_error = Some("Parallel: pick two lines, a line + an ellipse, or two ellipses (circular arcs have no orientation).".into());
            return;
        };
        if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
            self.status_error = Some(err);
            return;
        }
        self.exec(action);
    }

    fn apply_perpendicular(&mut self) {
        self.begin_group();
        if self.selection.len() == 2
            && let (Selection::Line(a), Selection::Line(b)) = (self.selection[0], self.selection[1]) {
                let action = Action::ApplyPerpendicular { a, b };
                if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                    self.status_error = Some(err);
                    return;
                }
                self.exec(action);
            }
    }

    fn apply_collinear(&mut self) {
        self.begin_group();
        if self.selection.len() == 2
            && let (Selection::Line(a), Selection::Line(b)) = (self.selection[0], self.selection[1]) {
                let action = Action::ApplyCollinear { a, b };
                if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                    self.status_error = Some(err);
                    return;
                }
                self.exec(action);
            }
    }

    fn apply_symmetry(&mut self) {
        self.begin_group();
        if self.selection.len() == 3 {
            // Line-Line-Line: first two = sides, third = mirror
            if let (Selection::Line(a), Selection::Line(c), Selection::Line(b)) =
                (self.selection[0], self.selection[1], self.selection[2])
            {
                let action = Action::ApplySymmetryLL { a, b, c };
                if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                    self.status_error = Some(err);
                    return;
                }
                self.exec(action);
                return;
            }
            // Arc-Arc symmetry: find the line (mirror axis) and two arcs in any order
            {
                let mut aa_arcs = Vec::new();
                let mut aa_line = None;
                for s in &self.selection {
                    match s {
                        Selection::Arc(r) => aa_arcs.push(*r),
                        Selection::Line(r) => aa_line = Some(*r),
                        _ => {}
                    }
                }
                if aa_arcs.len() == 2 && aa_line.is_some() {
                    self.exec(Action::ApplySymmetryAA { a: aa_arcs[0], line: aa_line.unwrap(), c: aa_arcs[1] });
                    return;
                }
            }
            // Point/endpoint - Line - Point/endpoint symmetry
            let sel = self.selection.clone();
            let to_point = |sketch: &mut Sketch, s: &Selection| -> Option<Ref<Point>> {
                match s {
                    Selection::Point(r) => Some(*r),
                    Selection::LineP1(r) => {
                        let pos = sketch.lines[*r].p1.value;
                        let hp = sketch.add_helper_point(pos);
                        sketch.coincident_lp1.push(CoincidentLP1 { line: *r, point: hp, nid: 0, cid: 0, hb: CrossBlock::new() });
                        Some(hp)
                    }
                    Selection::LineP2(r) => {
                        let pos = sketch.lines[*r].p2.value;
                        let hp = sketch.add_helper_point(pos);
                        sketch.coincident_lp2.push(CoincidentLP2 { line: *r, point: hp, nid: 0, cid: 0, hb: CrossBlock::new() });
                        Some(hp)
                    }
                    Selection::ArcCenter(r) => {
                        let pos = sketch.arcs[*r].center.value;
                        let hp = sketch.add_helper_point(pos);
                        sketch.coincident_arc_center.push(CoincidentArcCenter { point: hp, arc: *r, nid: 0, cid: 0, hb: CrossBlock::new() });
                        Some(hp)
                    }
                    Selection::ArcStart(r) => {
                        let pos = arael_sketch_backend::geometry::arc_start_pos(&sketch.arcs[*r]);
                        let hp = sketch.add_helper_point(pos);
                        sketch.coincident_arc_start.push(CoincidentArcStart { point: hp, arc: *r, nid: 0, cid: 0, hb: CrossBlock::new() });
                        Some(hp)
                    }
                    Selection::ArcEnd(r) => {
                        let pos = arael_sketch_backend::geometry::arc_end_pos(&sketch.arcs[*r]);
                        let hp = sketch.add_helper_point(pos);
                        sketch.coincident_arc_end.push(CoincidentArcEnd { point: hp, arc: *r, nid: 0, cid: 0, hb: CrossBlock::new() });
                        Some(hp)
                    }
                    _ => None,
                }
            };
            // Find the line (mirror axis) and two point-likes in any order
            let line_idx = sel.iter().position(|s| matches!(s, Selection::Line(_)));
            if let Some(li) = line_idx {
                let line = match sel[li] { Selection::Line(r) => r, _ => unreachable!() };
                let others: Vec<_> = sel.iter().enumerate()
                    .filter(|&(i, _)| i != li).map(|(_, s)| s).collect();
                if others.len() == 2 {
                    let a = to_point(self.sketch.get_mut(), others[0]);
                    let c = to_point(self.sketch.get_mut(), others[1]);
                    if let (Some(a), Some(c)) = (a, c) {
                        self.exec(Action::ApplySymmetryPP { a, line, c });
                    }
                }
            }
        }
    }

    fn apply_midpoint(&mut self) {
        self.begin_group();
        if self.selection.len() != 2 { return; }
        let (s0, s1) = (self.selection[0], self.selection[1]);
        // Find the line and the point-like entity
        let action = match (s0, s1) {
            (Selection::Point(p), Selection::Line(l)) | (Selection::Line(l), Selection::Point(p)) =>
                Some(Action::ApplyMidpoint { point: p, line: l }),
            (Selection::LineP1(src), Selection::Line(tgt)) | (Selection::Line(tgt), Selection::LineP1(src)) =>
                Some(Action::ApplyMidpointLP1 { line: src, target: tgt }),
            (Selection::LineP2(src), Selection::Line(tgt)) | (Selection::Line(tgt), Selection::LineP2(src)) =>
                Some(Action::ApplyMidpointLP2 { line: src, target: tgt }),
            (Selection::ArcStart(arc), Selection::Line(l)) | (Selection::Line(l), Selection::ArcStart(arc)) =>
                Some(Action::ApplyMidpointArcStart { arc, line: l }),
            (Selection::ArcEnd(arc), Selection::Line(l)) | (Selection::Line(l), Selection::ArcEnd(arc)) =>
                Some(Action::ApplyMidpointArcEnd { arc, line: l }),
            // Arc as target (angular midpoint)
            (Selection::Point(p), Selection::Arc(a)) | (Selection::Arc(a), Selection::Point(p)) =>
                Some(Action::ApplyMidpointArcPoint { point: p, arc: a }),
            (Selection::LineP1(l), Selection::Arc(a)) | (Selection::Arc(a), Selection::LineP1(l)) =>
                Some(Action::ApplyMidpointLP1Arc { line: l, arc: a }),
            (Selection::LineP2(l), Selection::Arc(a)) | (Selection::Arc(a), Selection::LineP2(l)) =>
                Some(Action::ApplyMidpointLP2Arc { line: l, arc: a }),
            (Selection::ArcStart(src), Selection::Arc(tgt)) | (Selection::Arc(tgt), Selection::ArcStart(src)) =>
                Some(Action::ApplyMidpointArcStartArc { a: src, b: tgt }),
            (Selection::ArcEnd(src), Selection::Arc(tgt)) | (Selection::Arc(tgt), Selection::ArcEnd(src)) =>
                Some(Action::ApplyMidpointArcEndArc { a: src, b: tgt }),
            _ => None,
        };
        if let Some(action) = action {
            if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                self.status_error = Some(err);
                return;
            }
            self.exec(action);
        }
    }

    fn apply_toggle_construction(&mut self) {
        self.begin_group();
        for sel in &self.selection.clone() {
            match *sel {
                Selection::Line(r) => { self.exec(Action::ToggleConstructionLine { line: r }); }
                Selection::Arc(r) => { self.exec(Action::ToggleConstructionArc { arc: r }); }
                _ => {}
            }
        }
    }

    // Find the directly locked vertex in the same transitive group as `sel`.
    // Returns unlock actions for all directly locked vertices in the group.
    fn find_direct_locks_in_group(&self, sel: Selection) -> Vec<Action> {
        let (_pt_locked, _l_p1_locked, _l_p2_locked, _arc_c_locked) = self.compute_locked_sets();

        // Build union-find (same as compute_locked_sets)
        let np = self.sketch.points.slot_count();
        let nl = self.sketch.lines.slot_count();
        let na = self.sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;
        let mut parent: Vec<usize> = (0..total).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; } x
        };
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb { parent[ra] = rb; }
        };
        let pt_id = |r: Ref<Point>| r.index() as usize;
        let lp1_id = |r: Ref<Line>| np + r.index() as usize;
        let lp2_id = |r: Ref<Line>| np + nl + r.index() as usize;
        let ac_id = |r: Ref<Arc>| np + 2 * nl + r.index() as usize;

        // Build unions (abbreviated - same constraint list as compute_locked_sets)
        for c in &self.sketch.coincident_pp { union(&mut parent, pt_id(c.a), pt_id(c.b)); }
        for c in &self.sketch.coincident_lp1 { union(&mut parent, lp1_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_lp2 { union(&mut parent, lp2_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_ll11 { union(&mut parent, lp1_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll12 { union(&mut parent, lp1_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_ll21 { union(&mut parent, lp2_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll22 { union(&mut parent, lp2_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_arc_center { union(&mut parent, pt_id(c.point), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_center { union(&mut parent, lp1_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_center { union(&mut parent, lp2_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.concentric { union(&mut parent, ac_id(c.a), ac_id(c.b)); }
        // (other arc constraint unions omitted for brevity - they follow the same pattern as compute_locked_sets)
        for c in &self.sketch.coincident_arc_start { union(&mut parent, pt_id(c.point), np + 2*nl + na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_arc_end { union(&mut parent, pt_id(c.point), np + 2*nl + 2*na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp1_arc_start { union(&mut parent, lp1_id(c.line), np + 2*nl + na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp2_arc_start { union(&mut parent, lp2_id(c.line), np + 2*nl + na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp1_arc_end { union(&mut parent, lp1_id(c.line), np + 2*nl + 2*na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp2_arc_end { union(&mut parent, lp2_id(c.line), np + 2*nl + 2*na + c.arc.index() as usize); }

        let sel_id = match sel {
            Selection::Point(r) => Some(pt_id(r)),
            Selection::LineP1(r) => Some(lp1_id(r)),
            Selection::LineP2(r) => Some(lp2_id(r)),
            Selection::ArcCenter(r) => Some(ac_id(r)),
            _ => None,
        };
        let sel_id = match sel_id { Some(id) => id, None => return Vec::new() };
        let sel_root = find(&mut parent, sel_id);

        let mut actions = Vec::new();
        // Find all directly locked vertices in the same group
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.constraints.has_fix_x && p.constraints.has_fix_y && find(&mut parent, pt_id(r)) == sel_root {
                actions.push(Action::UnlockPoint { point: r });
            }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if !l.p1.optimize && find(&mut parent, lp1_id(r)) == sel_root {
                actions.push(Action::UnlockLineP1 { line: r });
            }
            if !l.p2.optimize && find(&mut parent, lp2_id(r)) == sel_root {
                actions.push(Action::UnlockLineP2 { line: r });
            }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            if !a.center.optimize && find(&mut parent, ac_id(r)) == sel_root {
                actions.push(Action::UnlockArcCenter { arc: r });
            }
        }
        actions
    }

    fn apply_lock(&mut self) {
        self.begin_group();
        let (pt_locked, l_p1_locked, l_p2_locked, arc_c_locked) = self.compute_locked_sets();
        for sel in &self.selection.clone() {
            let is_locked = match *sel {
                Selection::Point(r) => pt_locked.contains(&r.index()),
                Selection::LineP1(r) => l_p1_locked.contains(&r.index()),
                Selection::LineP2(r) => l_p2_locked.contains(&r.index()),
                Selection::ArcCenter(r) => arc_c_locked.contains(&r.index()),
                _ => false,
            };
            if is_locked {
                // Unlock all directly locked vertices in the transitive group
                let unlock_actions = self.find_direct_locks_in_group(*sel);
                for action in unlock_actions {
                    self.exec(action);
                }
            } else {
                match *sel {
                    Selection::Point(r) => {
                        let pos = self.sketch.points[r].pos.value;
                        self.exec(Action::LockPoint { point: r, pos });
                    }
                    Selection::LineP1(r) => {
                        let pos = self.sketch.lines[r].p1.value;
                        self.exec(Action::LockLineP1 { line: r, pos });
                    }
                    Selection::LineP2(r) => {
                        let pos = self.sketch.lines[r].p2.value;
                        self.exec(Action::LockLineP2 { line: r, pos });
                    }
                    Selection::ArcCenter(r) => {
                        let pos = self.sketch.arcs[r].center.value;
                        self.exec(Action::LockArcCenter { arc: r, pos });
                    }
                    _ => {}
                }
            }
        }
    }

    fn apply_tangent(&mut self) {
        self.begin_group();
        if self.selection.len() != 2 { return; }
        let (s0, s1) = (self.selection[0], self.selection[1]);
        match (s0, s1) {
            (Selection::Line(line), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::Line(line)) => {
                self.exec(Action::ApplyTangentLA { line, arc });
            }
            (Selection::Arc(a), Selection::Arc(b)) => {
                let action = Action::ApplyTangentAA { a, b };
                if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                    self.status_error = Some(err);
                    return;
                }
                self.exec(action);
            }
            _ => {}
        }
    }

    fn apply_equal_length(&mut self) {
        self.begin_group();
        if self.selection.len() == 2 {
            match (self.selection[0], self.selection[1]) {
                (Selection::Line(a), Selection::Line(b)) => {
                    let action = Action::ApplyEqualLength { a, b };
                    if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                        self.status_error = Some(err);
                        return;
                    }
                    self.exec(action);
                }
                (Selection::Arc(a), Selection::Arc(b)) => {
                    let action = Action::ApplyEqualRadius { a, b };
                    if let Some(err) = arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action) {
                        self.status_error = Some(err);
                        return;
                    }
                    self.exec(action);
                }
                _ => {}
            }
        }
    }

    // Find a snap target near a position (for line drawing auto-coincident)
    fn find_snap_target(&self, sketch_pos: vect2d, threshold: f64) -> Option<(vect2d, SnapTarget)> {
        self.find_snap_target_ex(sketch_pos, threshold, None, None)
    }

    // When dragging line `line`'s endpoint (is_p1 = which end is being
    // dragged), find the host line the OPPOSITE endpoint is anchored to.
    // Used by the auto-perpendicular-snap feature.
    //
    // v1 scope: shared-endpoint coincidence and endpoint-on-line body.
    // Returns the first match found; if the opposite endpoint is
    // coincident with several lines, picks one deterministically.
    fn find_anchor_host_line_for_drag(&self, line: Ref<Line>, is_p1_dragged: bool) -> Option<Ref<Line>> {
        let opposite_is_p1 = !is_p1_dragged;
        for c in &self.sketch.coincident_ll11 {
            if opposite_is_p1 && c.a == line { return Some(c.b); }
            if opposite_is_p1 && c.b == line { return Some(c.a); }
        }
        for c in &self.sketch.coincident_ll12 {
            if opposite_is_p1 && c.a == line { return Some(c.b); }
            if !opposite_is_p1 && c.b == line { return Some(c.a); }
        }
        for c in &self.sketch.coincident_ll21 {
            if !opposite_is_p1 && c.a == line { return Some(c.b); }
            if opposite_is_p1 && c.b == line { return Some(c.a); }
        }
        for c in &self.sketch.coincident_ll22 {
            if !opposite_is_p1 && c.a == line { return Some(c.b); }
            if !opposite_is_p1 && c.b == line { return Some(c.a); }
        }
        for c in &self.sketch.line_p1_on_line {
            if opposite_is_p1 && c.a == line { return Some(c.b); }
        }
        for c in &self.sketch.line_p2_on_line {
            if !opposite_is_p1 && c.a == line { return Some(c.b); }
        }
        None
    }

    // If the cursor sits within `threshold_px` of the virtual line that
    // is perpendicular to `host` at `anchor`, returns the cursor projected
    // onto that virtual line. None otherwise. Threshold is a perpendicular
    // distance from the virtual axis, measured in screen pixels.
    fn try_perp_snap(
        &self,
        anchor: vect2d,
        host_p1: vect2d,
        host_p2: vect2d,
        cursor: vect2d,
        threshold_px: f32,
    ) -> Option<vect2d> {
        if self.snap_disabled { return None; }
        let hdx = host_p2.x - host_p1.x;
        let hdy = host_p2.y - host_p1.y;
        let hlen = (hdx * hdx + hdy * hdy).sqrt();
        if hlen < 1e-9 { return None; }
        let hd_x = hdx / hlen;
        let hd_y = hdy / hlen;
        let hp_x = -hd_y;
        let hp_y = hd_x;
        let cx = cursor.x - anchor.x;
        let cy = cursor.y - anchor.y;
        // Deviation from the perpendicular axis = signed component along host.
        let along = cx * hd_x + cy * hd_y;
        let along_px = (along.abs() as f32) * self.scale;
        if along_px >= threshold_px { return None; }
        let t = cx * hp_x + cy * hp_y;
        Some(vect2d::new(anchor.x + t * hp_x, anchor.y + t * hp_y))
    }

    // Host line referenced by a `SnapTarget` for purposes of
    // auto-perpendicular-snap -- the line whose orientation defines the
    // virtual 90-degree axis. Only makes sense for line-endpoint and
    // line-body snaps; other targets return None.
    pub fn perp_host_from_snap(snap: SnapTarget) -> Option<Ref<Line>> {
        match snap {
            SnapTarget::LineP1(h) | SnapTarget::LineP2(h) | SnapTarget::Line(h) => Some(h),
            _ => None,
        }
    }

    // Find the host line that best fits an auto-perpendicular snap
    // between `anchor` and the cursor. A host is any line whose
    // endpoint or body passes through `anchor`: the "last point may
    // lie on several lines at once" case. Returns the match with the
    // smallest along-host deviation (closest to a perfect 90).
    // Excludes `exclude` (typically the line being dragged).
    pub fn find_best_perp_host_at(
        &self,
        anchor: vect2d,
        cursor: vect2d,
        threshold_px: f32,
        exclude: Option<Ref<Line>>,
    ) -> Option<(Ref<Line>, vect2d)> {
        if self.snap_disabled { return None; }
        const HOST_EPS: f64 = 1e-4;
        let mut best: Option<(f32, Ref<Line>, vect2d)> = None;
        for r in self.sketch.lines.refs() {
            if Some(r) == exclude { continue; }
            let l = &self.sketch.lines[r];
            let d1 = ((l.p1.value.x - anchor.x).powi(2) + (l.p1.value.y - anchor.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - anchor.x).powi(2) + (l.p2.value.y - anchor.y).powi(2)).sqrt();
            let db = arael_sketch_backend::geometry::point_to_segment_dist(anchor, l.p1.value, l.p2.value);
            if !(d1 < HOST_EPS || d2 < HOST_EPS || db < HOST_EPS) { continue; }
            if let Some(p) = self.try_perp_snap(anchor, l.p1.value, l.p2.value, cursor, threshold_px) {
                let hdx = l.p2.value.x - l.p1.value.x;
                let hdy = l.p2.value.y - l.p1.value.y;
                let hlen = (hdx * hdx + hdy * hdy).sqrt().max(1e-12);
                let hd_x = hdx / hlen;
                let hd_y = hdy / hlen;
                let along = ((cursor.x - anchor.x) * hd_x + (cursor.y - anchor.y) * hd_y).abs() as f32
                    * self.scale;
                if best.as_ref().map_or(true, |b| along < b.0) {
                    best = Some((along, r, p));
                }
            }
        }
        best.map(|(_, r, p)| (r, p))
    }

    /// Auto-collinear host search. Mirrors `find_best_perp_host_at`
    /// but looks for a host whose *infinite* line passes through
    /// `anchor` AND is close to `cursor` in the perpendicular sense
    /// (rather than the along sense that defines perp). Returns the
    /// cursor's projection onto the host's infinite line -- the end
    /// position the drawn/dragged line would commit to if the user
    /// releases. Used for auto-collinear on line creation and drag.
    pub fn find_best_collinear_host_at(
        &self,
        anchor: vect2d,
        cursor: vect2d,
        threshold_px: f32,
        exclude: Option<Ref<Line>>,
    ) -> Option<(Ref<Line>, vect2d)> {
        if self.snap_disabled { return None; }
        const HOST_EPS: f64 = 1e-4;
        let mut best: Option<(f32, Ref<Line>, vect2d)> = None;
        for r in self.sketch.lines.refs() {
            if Some(r) == exclude { continue; }
            let l = &self.sketch.lines[r];
            let d1 = ((l.p1.value.x - anchor.x).powi(2) + (l.p1.value.y - anchor.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - anchor.x).powi(2) + (l.p2.value.y - anchor.y).powi(2)).sqrt();
            let db = arael_sketch_backend::geometry::point_to_segment_dist(anchor, l.p1.value, l.p2.value);
            if !(d1 < HOST_EPS || d2 < HOST_EPS || db < HOST_EPS) { continue; }
            let hdx = l.p2.value.x - l.p1.value.x;
            let hdy = l.p2.value.y - l.p1.value.y;
            let hlen = (hdx * hdx + hdy * hdy).sqrt();
            if hlen < 1e-12 { continue; }
            let hd_x = hdx / hlen;
            let hd_y = hdy / hlen;
            // Perpendicular distance from cursor to host's infinite line.
            let cx = cursor.x - anchor.x;
            let cy = cursor.y - anchor.y;
            let perp_dist = (cx * (-hd_y) + cy * hd_x).abs();
            let perp_px = (perp_dist as f32) * self.scale;
            if perp_px >= threshold_px { continue; }
            // Require the drawn segment to run *along* the host, not
            // just through its origin: the along projection must be
            // non-trivial compared to the perp deviation. Otherwise a
            // zero-length probe looks collinear with every host.
            let along = cx * hd_x + cy * hd_y;
            let along_px = (along.abs() as f32) * self.scale;
            if along_px < threshold_px * 3.0 { continue; }
            // Project cursor onto host's infinite line at anchor.
            let foot = vect2d::new(anchor.x + along * hd_x, anchor.y + along * hd_y);
            if best.as_ref().map_or(true, |b| perp_px < b.0) {
                best = Some((perp_px, r, foot));
            }
        }
        best.map(|(_, r, p)| (r, p))
    }

    /// True if applying Collinear between `a` and `b` is a direct
    /// conflict (already exists, or parallel/perpendicular is set).
    fn has_collinear_conflict(&self, a: Ref<Line>, b: Ref<Line>) -> bool {
        self.sketch.collinear.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a))
            || self.sketch.perpendicular.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a))
    }

    // End-side perpendicular snap. The drawn line goes from `start` to
    // a point on the snapped target line; if that target's direction is
    // near perpendicular to the drawn direction, snap the end to the
    // foot-of-perpendicular from `start` onto the target. Threshold is
    // measured in screen pixels along the target.
    pub fn try_perp_end_snap(
        &self,
        start: vect2d,
        target_p1: vect2d,
        target_p2: vect2d,
        cursor_on_target: vect2d,
        threshold_px: f32,
    ) -> Option<vect2d> {
        if self.snap_disabled { return None; }
        let tdx = target_p2.x - target_p1.x;
        let tdy = target_p2.y - target_p1.y;
        let tlen2 = tdx * tdx + tdy * tdy;
        if tlen2 < 1e-18 { return None; }
        let s = ((start.x - target_p1.x) * tdx + (start.y - target_p1.y) * tdy) / tlen2;
        let foot = vect2d::new(target_p1.x + s * tdx, target_p1.y + s * tdy);
        let dx = cursor_on_target.x - foot.x;
        let dy = cursor_on_target.y - foot.y;
        let d_px = ((dx * dx + dy * dy).sqrt() as f32) * self.scale;
        if d_px >= threshold_px { return None; }
        Some(foot)
    }

    // True if a Perpendicular constraint between `a` and `b` already
    // exists (either order), or a Parallel constraint between them would
    // make perpendicular a conflict. Used to skip auto-perp.
    fn has_perp_conflict(&self, a: Ref<Line>, b: Ref<Line>) -> bool {
        self.sketch.perpendicular.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a))
            || self.sketch.parallel.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a))
    }

    fn find_snap_target_ex(&self, sketch_pos: vect2d, threshold: f64, exclude_line: Option<Ref<Line>>, exclude_arc: Option<Ref<Arc>>) -> Option<(vect2d, SnapTarget)> {
        self.find_snap_target_filter(sketch_pos, threshold, exclude_line, exclude_arc, |_| true)
    }

    // Like find_snap_target_ex but skips any candidate for which `keep`
    // returns false. Used by update_drag to skip targets already attached
    // to the dragged entity, so a second-closest unattached candidate can
    // still win (e.g. dragging a coincident-paired endpoint: the paired
    // twin sits at the cursor with distance ~= 0 and must not block
    // farther candidates within threshold).
    fn find_snap_target_filter(
        &self,
        sketch_pos: vect2d,
        threshold: f64,
        exclude_line: Option<Ref<Line>>,
        exclude_arc: Option<Ref<Arc>>,
        keep: impl Fn(&SnapTarget) -> bool,
    ) -> Option<(vect2d, SnapTarget)> {
        // Hold-to-disable modifier: Shift. See EditorApp::snap_disabled.
        if self.snap_disabled { return None; }

        // First pass: check points and line endpoints (high priority)
        let mut best: Option<(f64, vect2d, SnapTarget)> = None;

        // Snap zone is tighter than the hit-test threshold. Snap pulls the
        // drawn/dragged geometry around; aggressive snap zones feel sticky.
        // Hit-test (clicking) keeps the full threshold elsewhere.
        let threshold = threshold * 0.5;
        let mut check = |dist: f64, pos: vect2d, target: SnapTarget| {
            if dist < threshold
                && (best.is_none() || dist < best.unwrap().0)
                && keep(&target) {
                    best = Some((dist, pos, target));
                }
        };

        // Standalone points (skip helpers)
        for r in self.sketch.points.refs() {
            if self.drag_point == Some(r) { continue; }
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d, p.pos.value, SnapTarget::Point(r));
        }

        // Line endpoints + midpoint
        for r in self.sketch.lines.refs() {
            if exclude_line == Some(r) { continue; }
            let l = &self.sketch.lines[r];

            let d1 = ((l.p1.value.x - sketch_pos.x).powi(2)
                    + (l.p1.value.y - sketch_pos.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - sketch_pos.x).powi(2)
                    + (l.p2.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d1, l.p1.value, SnapTarget::LineP1(r));
            check(d2, l.p2.value, SnapTarget::LineP2(r));

            let mid = vect2d::new((l.p1.value.x + l.p2.value.x) * 0.5,
                                  (l.p1.value.y + l.p2.value.y) * 0.5);
            let dm = ((mid.x - sketch_pos.x).powi(2)
                    + (mid.y - sketch_pos.y).powi(2)).sqrt();
            check(dm, mid, SnapTarget::LineMidpoint(r));
        }

        // Arc centers and endpoints (same priority as points/line endpoints)
        for r in self.sketch.arcs.refs() {
            if exclude_arc == Some(r) { continue; }
            let a = &self.sketch.arcs[r];
            let dc = ((a.center.value.x - sketch_pos.x).powi(2)
                    + (a.center.value.y - sketch_pos.y).powi(2)).sqrt();
            check(dc, a.center.value, SnapTarget::ArcCenter(r));
            if !a.closed {
                let sp = arc_start_pos(a);
                let ep = arc_end_pos(a);
                let ds = ((sp.x - sketch_pos.x).powi(2) + (sp.y - sketch_pos.y).powi(2)).sqrt();
                let de = ((ep.x - sketch_pos.x).powi(2) + (ep.y - sketch_pos.y).powi(2)).sqrt();
                check(ds, sp, SnapTarget::ArcStart(r));
                check(de, ep, SnapTarget::ArcEnd(r));
                // Arc midpoint (parametric): half-way between start and end
                // angles along the sweep direction.
                let mid_t = (a.start_angle.value + a.end_angle.value) * 0.5;
                let mp = arc_point_at(a, mid_t);
                let dmp = ((mp.x - sketch_pos.x).powi(2) + (mp.y - sketch_pos.y).powi(2)).sqrt();
                check(dmp, mp, SnapTarget::ArcMidpoint(r));
            }
        }

        // If we found a point/endpoint, prefer it over line body
        if best.is_some() {
            return best.map(|(_, pos, target)| (pos, target));
        }

        // Second pass: check line bodies and arc/circle curves (lower priority)
        for r in self.sketch.lines.refs() {
            if exclude_line == Some(r) { continue; }
            let t = SnapTarget::Line(r);
            if !keep(&t) { continue; }
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            if d < threshold
                && (best.is_none() || d < best.unwrap().0) {
                    let proj = project_onto_segment(sketch_pos, l.p1.value, l.p2.value);
                    best = Some((d, proj, t));
                }
        }

        for r in self.sketch.arcs.refs() {
            if exclude_arc == Some(r) { continue; }
            let t = SnapTarget::ArcBody(r);
            if !keep(&t) { continue; }
            let a = &self.sketch.arcs[r];
            let (d, proj) = point_to_arc_dist(sketch_pos, a);
            if d < threshold
                && (best.is_none() || d < best.unwrap().0) {
                    best = Some((d, proj, t));
                }
        }

        best.map(|(_, pos, target)| (pos, target))
    }

    // Apply a coincident/on-line constraint between a snap target and a line endpoint.
    // For arc snap targets that lack a direct line-arc constraint, uses a helper point.
    fn apply_snap_coincident(&mut self, snap: SnapTarget, line: Ref<Line>, is_p1: bool) {
        if self.has_existing_coincident_line(line, is_p1, snap) { return; }
        match (snap, is_p1) {
            (SnapTarget::Point(p), true) => { self.exec(Action::ApplyCoincidentLP1 { line, point: p }); }
            (SnapTarget::Point(p), false) => { self.exec(Action::ApplyCoincidentLP2 { line, point: p }); }
            (SnapTarget::LineP1(other), true) => { self.exec(Action::ApplyCoincidentLL11 { a: line, b: other }); }
            (SnapTarget::LineP1(other), false) => { self.exec(Action::ApplyCoincidentLL21 { a: line, b: other }); }
            (SnapTarget::LineP2(other), true) => { self.exec(Action::ApplyCoincidentLL12 { a: line, b: other }); }
            (SnapTarget::LineP2(other), false) => { self.exec(Action::ApplyCoincidentLL22 { a: line, b: other }); }
            (SnapTarget::Line(other), true) => { self.exec(Action::ApplyLineP1OnLine { a: line, b: other }); }
            (SnapTarget::Line(other), false) => { self.exec(Action::ApplyLineP2OnLine { a: line, b: other }); }
            // Direct line-arc constraints
            (SnapTarget::ArcCenter(arc), true) => { self.exec(Action::ApplyCoincidentLP1ArcCenter { line, arc }); }
            (SnapTarget::ArcCenter(arc), false) => { self.exec(Action::ApplyCoincidentLP2ArcCenter { line, arc }); }
            (SnapTarget::ArcStart(arc), true) => { self.exec(Action::ApplyCoincidentLP1ArcStart { line, arc }); }
            (SnapTarget::ArcStart(arc), false) => { self.exec(Action::ApplyCoincidentLP2ArcStart { line, arc }); }
            (SnapTarget::ArcEnd(arc), true) => { self.exec(Action::ApplyCoincidentLP1ArcEnd { line, arc }); }
            (SnapTarget::ArcEnd(arc), false) => { self.exec(Action::ApplyCoincidentLP2ArcEnd { line, arc }); }
            // Line endpoint on arc body (direct constraint)
            (SnapTarget::ArcBody(arc), true) => { self.exec(Action::ApplyLineP1OnArc { line, arc }); }
            (SnapTarget::ArcBody(arc), false) => { self.exec(Action::ApplyLineP2OnArc { line, arc }); }
            // Line endpoint at another line's midpoint
            (SnapTarget::LineMidpoint(target), true) => { self.exec(Action::ApplyMidpointLP1 { line, target }); }
            (SnapTarget::LineMidpoint(target), false) => { self.exec(Action::ApplyMidpointLP2 { line, target }); }
            // Line endpoint at an arc's midpoint
            (SnapTarget::ArcMidpoint(arc), true) => { self.exec(Action::ApplyMidpointLP1Arc { line, arc }); }
            (SnapTarget::ArcMidpoint(arc), false) => { self.exec(Action::ApplyMidpointLP2Arc { line, arc }); }
        }
    }

    // Apply a coincident/on-line constraint between a snap target and a standalone point
    fn apply_snap_coincident_point(&mut self, snap: SnapTarget, point: Ref<Point>) {
        if let Some(snap_sel) = Self::snap_to_selection(snap)
            && self.are_transitively_coincident(Selection::Point(point), snap_sel) { return; }
        let action = match snap {
            SnapTarget::Point(other) => Action::ApplyCoincidentPP { a: point, b: other },
            SnapTarget::LineP1(line) => Action::ApplyCoincidentLP1 { line, point },
            SnapTarget::LineP2(line) => Action::ApplyCoincidentLP2 { line, point },
            SnapTarget::Line(line) => Action::ApplyPointOnLine { point, line },
            SnapTarget::LineMidpoint(line) => Action::ApplyMidpoint { point, line },
            SnapTarget::ArcCenter(arc) => Action::ApplyCoincidentArcCenter { point, arc },
            SnapTarget::ArcStart(arc) => Action::ApplyCoincidentArcStart { point, arc },
            SnapTarget::ArcEnd(arc) => Action::ApplyCoincidentArcEnd { point, arc },
            SnapTarget::ArcMidpoint(arc) => Action::ApplyMidpointArcPoint { point, arc },
            SnapTarget::ArcBody(arc) => Action::ApplyPointOnArc { point, arc },
        };
        self.exec(action);
    }


    // Apply a snap constraint between a snap target and an arc point.
    // Uses direct constraints where possible; helper point for Line/ArcBody.
    fn apply_snap_coincident_arc(&mut self, snap: SnapTarget, arc: Ref<Arc>, which: ArcPoint, pos: vect2d) {
        // Check transitive coincidence
        if let Some(snap_sel) = Self::snap_to_selection(snap) {
            let arc_sel = match which {
                ArcPoint::Center => Selection::ArcCenter(arc),
                ArcPoint::Start => Selection::ArcStart(arc),
                ArcPoint::End => Selection::ArcEnd(arc),
            };
            if self.are_transitively_coincident(arc_sel, snap_sel) { return; }
        }
        // Direct Line endpoint <-> Arc point constraints
        match (&which, snap) {
            (ArcPoint::Center, SnapTarget::LineP1(line)) | (ArcPoint::Center, SnapTarget::LineP2(line)) => {
                let is_p1 = matches!(snap, SnapTarget::LineP1(_));
                if is_p1 { self.exec(Action::ApplyCoincidentLP1ArcCenter { line, arc }); }
                else { self.exec(Action::ApplyCoincidentLP2ArcCenter { line, arc }); }
                return;
            }
            (ArcPoint::Start, SnapTarget::LineP1(line)) | (ArcPoint::Start, SnapTarget::LineP2(line)) => {
                let is_p1 = matches!(snap, SnapTarget::LineP1(_));
                if is_p1 { self.exec(Action::ApplyCoincidentLP1ArcStart { line, arc }); }
                else { self.exec(Action::ApplyCoincidentLP2ArcStart { line, arc }); }
                return;
            }
            (ArcPoint::End, SnapTarget::LineP1(line)) | (ArcPoint::End, SnapTarget::LineP2(line)) => {
                let is_p1 = matches!(snap, SnapTarget::LineP1(_));
                if is_p1 { self.exec(Action::ApplyCoincidentLP1ArcEnd { line, arc }); }
                else { self.exec(Action::ApplyCoincidentLP2ArcEnd { line, arc }); }
                return;
            }
            // Direct Arc <-> Arc point constraints
            (ArcPoint::Center, SnapTarget::ArcCenter(other)) => { self.exec(Action::ApplyConcentric { a: arc, b: other }); return; }
            (ArcPoint::Center, SnapTarget::ArcStart(other)) => { self.exec(Action::ApplyCoincidentArcCenterStart { a: arc, b: other }); return; }
            (ArcPoint::Center, SnapTarget::ArcEnd(other)) => { self.exec(Action::ApplyCoincidentArcCenterEnd { a: arc, b: other }); return; }
            (ArcPoint::Start, SnapTarget::ArcCenter(other)) => { self.exec(Action::ApplyCoincidentArcStartCenter { a: arc, b: other }); return; }
            (ArcPoint::Start, SnapTarget::ArcStart(other)) => { self.exec(Action::ApplyCoincidentArcStartStart { a: arc, b: other }); return; }
            (ArcPoint::Start, SnapTarget::ArcEnd(other)) => { self.exec(Action::ApplyCoincidentArcStartEnd { a: arc, b: other }); return; }
            (ArcPoint::End, SnapTarget::ArcCenter(other)) => { self.exec(Action::ApplyCoincidentArcEndCenter { a: arc, b: other }); return; }
            (ArcPoint::End, SnapTarget::ArcStart(other)) => { self.exec(Action::ApplyCoincidentArcEndStart { a: arc, b: other }); return; }
            (ArcPoint::End, SnapTarget::ArcEnd(other)) => { self.exec(Action::ApplyCoincidentArcEndEnd { a: arc, b: other }); return; }
            _ => {}
        }
        // Convert ArcPoint to DimensionEndpoint
        let endpoint = match which {
            ArcPoint::Center => DimensionEndpoint::ArcCenter(arc),
            ArcPoint::Start => DimensionEndpoint::ArcStart(arc),
            ArcPoint::End => DimensionEndpoint::ArcEnd(arc),
        };
        match snap {
            SnapTarget::Line(line) => {
                self.exec(Action::ApplyEndpointOnLine { endpoint, line });
            }
            SnapTarget::ArcBody(target_arc) => {
                self.exec(Action::ApplyEndpointOnArc { endpoint, arc: target_arc });
            }
            _ => {
                // Point, LineP1/P2, ArcCenter/Start/End: create helper + bridge + coincident
                let hp_pos = pos;
                let hp = self.sketch.get_mut().add_helper_point(hp_pos);
                match which {
                    ArcPoint::Center => self.exec(Action::ApplyCoincidentArcCenter { point: hp, arc }),
                    ArcPoint::Start => self.exec(Action::ApplyCoincidentArcStart { point: hp, arc }),
                    ArcPoint::End => self.exec(Action::ApplyCoincidentArcEnd { point: hp, arc }),
                };
                self.apply_snap_coincident_point(snap, hp);
            }
        }
    }

    /// Resolve a `ConstraintId` to its user-visible name (`C<nid>` for
    /// numbered constraints, `CL0H` / `CL0V` for line flag constraints).
    /// `HelperBridge` has no user-visible constraint name since it's a
    /// synthetic grouping of coincident constraints.
    pub fn constraint_name(&self, id: ConstraintId) -> Option<String> {
        use arael_sketch_solver::format_flag_name;
        match id {
            ConstraintId::Horizontal(r) => Some(format_flag_name(&self.sketch.lines[r].name, 'H')),
            ConstraintId::Vertical(r) => Some(format_flag_name(&self.sketch.lines[r].name, 'V')),
            ConstraintId::Parallel(i) => Some(format!("C{}", self.sketch.parallel[i].nid)),
            ConstraintId::ArcLineParallel(i) => Some(format!("C{}", self.sketch.arc_line_parallel[i].nid)),
            ConstraintId::ArcArcParallel(i) => Some(format!("C{}", self.sketch.arc_arc_parallel[i].nid)),
            ConstraintId::Perpendicular(i) => Some(format!("C{}", self.sketch.perpendicular[i].nid)),
            ConstraintId::EqualLength(i) => Some(format!("C{}", self.sketch.equal_length[i].nid)),
            ConstraintId::EqualRadius(i) => Some(format!("C{}", self.sketch.equal_radius[i].nid)),
            ConstraintId::Concentric(i) => Some(format!("C{}", self.sketch.concentric[i].nid)),
            ConstraintId::TangentLA(i) => Some(format!("C{}", self.sketch.tangent_la[i].nid)),
            ConstraintId::TangentAA(i) => Some(format!("C{}", self.sketch.tangent_aa[i].nid)),
            ConstraintId::Collinear(i) => Some(format!("C{}", self.sketch.collinear[i].nid)),
            ConstraintId::Symmetry(i) => Some(format!("C{}", self.sketch.symmetry_ll[i].nid)),
            ConstraintId::SymmetryPP(i) => Some(format!("C{}", self.sketch.symmetry_pp[i].nid)),
            ConstraintId::SymmetryAA(i) => Some(format!("C{}", self.sketch.symmetry_aa[i].nid)),
            ConstraintId::Midpoint(kind, i) => {
                let nid = match kind {
                    MidpointKind::Point => self.sketch.midpoint[i].nid,
                    MidpointKind::LP1 => self.sketch.midpoint_lp1[i].nid,
                    MidpointKind::LP2 => self.sketch.midpoint_lp2[i].nid,
                    MidpointKind::ArcStart => self.sketch.midpoint_arc_start[i].nid,
                    MidpointKind::ArcEnd => self.sketch.midpoint_arc_end[i].nid,
                    MidpointKind::ArcPoint => self.sketch.midpoint_arc_point[i].nid,
                    MidpointKind::LP1Arc => self.sketch.midpoint_lp1_arc[i].nid,
                    MidpointKind::LP2Arc => self.sketch.midpoint_lp2_arc[i].nid,
                    MidpointKind::ArcStartArc => self.sketch.midpoint_arc_start_arc[i].nid,
                    MidpointKind::ArcEndArc => self.sketch.midpoint_arc_end_arc[i].nid,
                };
                Some(format!("C{}", nid))
            }
            ConstraintId::Coincident(kind, i) => {
                let nid = match kind {
                    CoincidentKind::PP => self.sketch.coincident_pp[i].nid,
                    CoincidentKind::LP1 => self.sketch.coincident_lp1[i].nid,
                    CoincidentKind::LP2 => self.sketch.coincident_lp2[i].nid,
                    CoincidentKind::LL11 => self.sketch.coincident_ll11[i].nid,
                    CoincidentKind::LL12 => self.sketch.coincident_ll12[i].nid,
                    CoincidentKind::LL21 => self.sketch.coincident_ll21[i].nid,
                    CoincidentKind::LL22 => self.sketch.coincident_ll22[i].nid,
                    CoincidentKind::PointOnLine => self.sketch.point_on_line[i].nid,
                    CoincidentKind::PointOnArc => self.sketch.point_on_arc[i].nid,
                    CoincidentKind::LP1OnLine => self.sketch.line_p1_on_line[i].nid,
                    CoincidentKind::LP2OnLine => self.sketch.line_p2_on_line[i].nid,
                    CoincidentKind::LP1OnArc => self.sketch.line_p1_on_arc[i].nid,
                    CoincidentKind::LP2OnArc => self.sketch.line_p2_on_arc[i].nid,
                    CoincidentKind::ArcCenter => self.sketch.coincident_arc_center[i].nid,
                    CoincidentKind::ArcStart => self.sketch.coincident_arc_start[i].nid,
                    CoincidentKind::ArcEnd => self.sketch.coincident_arc_end[i].nid,
                    CoincidentKind::LP1ArcCenter => self.sketch.coincident_lp1_arc_center[i].nid,
                    CoincidentKind::LP2ArcCenter => self.sketch.coincident_lp2_arc_center[i].nid,
                    CoincidentKind::LP1ArcStart => self.sketch.coincident_lp1_arc_start[i].nid,
                    CoincidentKind::LP2ArcStart => self.sketch.coincident_lp2_arc_start[i].nid,
                    CoincidentKind::LP1ArcEnd => self.sketch.coincident_lp1_arc_end[i].nid,
                    CoincidentKind::LP2ArcEnd => self.sketch.coincident_lp2_arc_end[i].nid,
                    CoincidentKind::ArcCenterStart => self.sketch.coincident_arc_center_start[i].nid,
                    CoincidentKind::ArcCenterEnd => self.sketch.coincident_arc_center_end[i].nid,
                    CoincidentKind::ArcStartCenter => self.sketch.coincident_arc_start_center[i].nid,
                    CoincidentKind::ArcEndCenter => self.sketch.coincident_arc_end_center[i].nid,
                    CoincidentKind::ArcStartStart => self.sketch.coincident_arc_start_start[i].nid,
                    CoincidentKind::ArcStartEnd => self.sketch.coincident_arc_start_end[i].nid,
                    CoincidentKind::ArcEndStart => self.sketch.coincident_arc_end_start[i].nid,
                    CoincidentKind::ArcEndEnd => self.sketch.coincident_arc_end_end[i].nid,
                };
                Some(format!("C{}", nid))
            }
            // A helper-point bridge is plumbing: a helper Pc<n>
            // anchored to one real entity via CoincidentArc* / LP1 /
            // LP2 / ... and attached to another via point_on_line /
            // point_on_arc. The user-facing constraint is the
            // "attach" relation, not the bridge itself -- so resolve
            // to that C<nid> when available. Helper names (Pc<n>)
            // must never leak out.
            ConstraintId::HelperBridge(pt) => {
                if let Some(c) = self.sketch.point_on_line.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                if let Some(c) = self.sketch.point_on_arc.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                // Fall back to the first anchoring coincidence.
                if let Some(c) = self.sketch.coincident_pp.iter().find(|c| c.a == pt || c.b == pt) {
                    return Some(format!("C{}", c.nid));
                }
                if let Some(c) = self.sketch.coincident_lp1.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                if let Some(c) = self.sketch.coincident_lp2.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                if let Some(c) = self.sketch.coincident_arc_center.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                if let Some(c) = self.sketch.coincident_arc_start.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                if let Some(c) = self.sketch.coincident_arc_end.iter().find(|c| c.point == pt) {
                    return Some(format!("C{}", c.nid));
                }
                None
            }
        }
    }

    pub fn describe_constraint(&self, id: ConstraintId) -> String {
        // For most constraints, format as "C<nid>: <listing>" / "CL0H: <listing>"
        // so the UI status bar matches `list` / `info <name>` output.
        if let Some(name) = self.constraint_name(id)
            && let Some(desc) = self.sketch.find_constraint_description(&name) {
            return format!("{}: {}", name, desc);
        }
        let ln = |r: Ref<Line>| self.sketch.lines[r].name.clone();
        let an = |r: Ref<Arc>| self.sketch.arcs[r].name.clone();
        let pn = |r: Ref<Point>| self.sketch.points[r].name.clone();
        match id {
            ConstraintId::Horizontal(r) => format!("H({})", ln(r)),
            ConstraintId::Vertical(r) => format!("V({})", ln(r)),
            ConstraintId::Parallel(i) => { let c = &self.sketch.parallel[i]; format!("Parallel({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::ArcLineParallel(i) => { let c = &self.sketch.arc_line_parallel[i]; format!("Parallel({}, {})", an(c.arc), ln(c.line)) }
            ConstraintId::ArcArcParallel(i) => { let c = &self.sketch.arc_arc_parallel[i]; format!("Parallel({}, {})", an(c.a), an(c.b)) }
            ConstraintId::Perpendicular(i) => { let c = &self.sketch.perpendicular[i]; format!("Perp({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::EqualLength(i) => { let c = &self.sketch.equal_length[i]; format!("Equal({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::EqualRadius(i) => { let c = &self.sketch.equal_radius[i]; format!("EqualR({}, {})", an(c.a), an(c.b)) }
            ConstraintId::Concentric(i) => { let c = &self.sketch.concentric[i]; format!("Concentric({}, {})", an(c.a), an(c.b)) }
            ConstraintId::TangentLA(i) => { let c = &self.sketch.tangent_la[i]; format!("Tangent({}, {})", ln(c.line), an(c.arc)) }
            ConstraintId::TangentAA(i) => { let c = &self.sketch.tangent_aa[i]; format!("Tangent({}, {})", an(c.a), an(c.b)) }
            ConstraintId::Collinear(i) => { let c = &self.sketch.collinear[i]; format!("Collinear({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::Symmetry(i) => { let c = &self.sketch.symmetry_ll[i]; format!("Symmetry({}, {}, {})", ln(c.a), ln(c.b), ln(c.c)) }
            ConstraintId::SymmetryPP(i) => { let c = &self.sketch.symmetry_pp[i]; format!("Symmetry({}, {}, {})", self.sketch.point_display_name(c.a), ln(c.line), self.sketch.point_display_name(c.c)) }
            ConstraintId::SymmetryAA(i) => { let c = &self.sketch.symmetry_aa[i]; format!("Symmetry({}, {}, {})", an(c.a), ln(c.line), an(c.c)) }
            ConstraintId::Midpoint(kind, i) => {
                let desc = match kind {
                    MidpointKind::Point => { let c = &self.sketch.midpoint[i]; format!("{} @ mid({})", pn(c.point), ln(c.line)) }
                    MidpointKind::LP1 => { let c = &self.sketch.midpoint_lp1[i]; format!("{}.p1 @ mid({})", ln(c.line), ln(c.target)) }
                    MidpointKind::LP2 => { let c = &self.sketch.midpoint_lp2[i]; format!("{}.p2 @ mid({})", ln(c.line), ln(c.target)) }
                    MidpointKind::ArcStart => { let c = &self.sketch.midpoint_arc_start[i]; format!("{}.s @ mid({})", an(c.arc), ln(c.line)) }
                    MidpointKind::ArcEnd => { let c = &self.sketch.midpoint_arc_end[i]; format!("{}.e @ mid({})", an(c.arc), ln(c.line)) }
                    MidpointKind::ArcPoint => { let c = &self.sketch.midpoint_arc_point[i]; format!("{} @ mid({})", pn(c.point), an(c.arc)) }
                    MidpointKind::LP1Arc => { let c = &self.sketch.midpoint_lp1_arc[i]; format!("{}.p1 @ mid({})", ln(c.line), an(c.arc)) }
                    MidpointKind::LP2Arc => { let c = &self.sketch.midpoint_lp2_arc[i]; format!("{}.p2 @ mid({})", ln(c.line), an(c.arc)) }
                    MidpointKind::ArcStartArc => { let c = &self.sketch.midpoint_arc_start_arc[i]; format!("{}.s @ mid({})", an(c.a), an(c.b)) }
                    MidpointKind::ArcEndArc => { let c = &self.sketch.midpoint_arc_end_arc[i]; format!("{}.e @ mid({})", an(c.a), an(c.b)) }
                };
                format!("Midpoint({})", desc)
            }
            ConstraintId::Coincident(kind, i) => {
                let desc = match kind {
                    CoincidentKind::PP => { let c = &self.sketch.coincident_pp[i]; format!("{} = {}", pn(c.a), pn(c.b)) }
                    CoincidentKind::LP1 => { let c = &self.sketch.coincident_lp1[i]; format!("{}.p1 = {}", ln(c.line), pn(c.point)) }
                    CoincidentKind::LP2 => { let c = &self.sketch.coincident_lp2[i]; format!("{}.p2 = {}", ln(c.line), pn(c.point)) }
                    CoincidentKind::LL11 => { let c = &self.sketch.coincident_ll11[i]; format!("{}.p1 = {}.p1", ln(c.a), ln(c.b)) }
                    CoincidentKind::LL12 => { let c = &self.sketch.coincident_ll12[i]; format!("{}.p1 = {}.p2", ln(c.a), ln(c.b)) }
                    CoincidentKind::LL21 => { let c = &self.sketch.coincident_ll21[i]; format!("{}.p2 = {}.p1", ln(c.a), ln(c.b)) }
                    CoincidentKind::LL22 => { let c = &self.sketch.coincident_ll22[i]; format!("{}.p2 = {}.p2", ln(c.a), ln(c.b)) }
                    CoincidentKind::PointOnLine => { let c = &self.sketch.point_on_line[i]; format!("{} on {}", pn(c.point), ln(c.line)) }
                    CoincidentKind::PointOnArc => { let c = &self.sketch.point_on_arc[i]; format!("{} on {}", pn(c.point), an(c.arc)) }
                    CoincidentKind::LP1OnLine => { let c = &self.sketch.line_p1_on_line[i]; format!("{}.p1 on {}", ln(c.a), ln(c.b)) }
                    CoincidentKind::LP2OnLine => { let c = &self.sketch.line_p2_on_line[i]; format!("{}.p2 on {}", ln(c.a), ln(c.b)) }
                    CoincidentKind::LP1OnArc => { let c = &self.sketch.line_p1_on_arc[i]; format!("{}.p1 on {}", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2OnArc => { let c = &self.sketch.line_p2_on_arc[i]; format!("{}.p2 on {}", ln(c.line), an(c.arc)) }
                    CoincidentKind::ArcCenter => { let c = &self.sketch.coincident_arc_center[i]; format!("{} = {}.c", pn(c.point), an(c.arc)) }
                    CoincidentKind::ArcStart => { let c = &self.sketch.coincident_arc_start[i]; format!("{} = {}.s", pn(c.point), an(c.arc)) }
                    CoincidentKind::ArcEnd => { let c = &self.sketch.coincident_arc_end[i]; format!("{} = {}.e", pn(c.point), an(c.arc)) }
                    CoincidentKind::LP1ArcCenter => { let c = &self.sketch.coincident_lp1_arc_center[i]; format!("{}.p1 = {}.c", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2ArcCenter => { let c = &self.sketch.coincident_lp2_arc_center[i]; format!("{}.p2 = {}.c", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP1ArcStart => { let c = &self.sketch.coincident_lp1_arc_start[i]; format!("{}.p1 = {}.s", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2ArcStart => { let c = &self.sketch.coincident_lp2_arc_start[i]; format!("{}.p2 = {}.s", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP1ArcEnd => { let c = &self.sketch.coincident_lp1_arc_end[i]; format!("{}.p1 = {}.e", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2ArcEnd => { let c = &self.sketch.coincident_lp2_arc_end[i]; format!("{}.p2 = {}.e", ln(c.line), an(c.arc)) }
                    CoincidentKind::ArcCenterStart => { let c = &self.sketch.coincident_arc_center_start[i]; format!("{}.c = {}.s", an(c.a), an(c.b)) }
                    CoincidentKind::ArcCenterEnd => { let c = &self.sketch.coincident_arc_center_end[i]; format!("{}.c = {}.e", an(c.a), an(c.b)) }
                    CoincidentKind::ArcStartCenter => { let c = &self.sketch.coincident_arc_start_center[i]; format!("{}.s = {}.c", an(c.a), an(c.b)) }
                    CoincidentKind::ArcEndCenter => { let c = &self.sketch.coincident_arc_end_center[i]; format!("{}.e = {}.c", an(c.a), an(c.b)) }
                    CoincidentKind::ArcStartStart => { let c = &self.sketch.coincident_arc_start_start[i]; format!("{}.s = {}.s", an(c.a), an(c.b)) }
                    CoincidentKind::ArcStartEnd => { let c = &self.sketch.coincident_arc_start_end[i]; format!("{}.s = {}.e", an(c.a), an(c.b)) }
                    CoincidentKind::ArcEndStart => { let c = &self.sketch.coincident_arc_end_start[i]; format!("{}.e = {}.s", an(c.a), an(c.b)) }
                    CoincidentKind::ArcEndEnd => { let c = &self.sketch.coincident_arc_end_end[i]; format!("{}.e = {}.e", an(c.a), an(c.b)) }
                };
                format!("Coinc({})", desc)
            }
            // HelperBridge is resolved to the underlying point_on /
            // coincidence constraint via `constraint_name` above, so
            // this arm is only reached when the bridge has no
            // resolvable sub-constraint (shouldn't happen in a
            // well-formed sketch). Fall back to the generic label.
            ConstraintId::HelperBridge(_) => "bridge".to_string(),
        }
    }

    // Get the line/arc/point refs involved in a constraint (for highlighting).
    // Endpoint-specific fields (line_p1s, arc_starts, etc.) highlight just
    // the endpoint, not the whole entity.
    pub fn constraint_entities(&self, id: ConstraintId) -> ConstraintEntities {
        let mut e = ConstraintEntities::default();
        let lines = &mut e.lines;
        let arcs = &mut e.arcs;
        let points = &mut e.points;
        match id {
            ConstraintId::Horizontal(r) | ConstraintId::Vertical(r) => { lines.push(r); }
            ConstraintId::Parallel(i) => {
                let c = &self.sketch.parallel[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::ArcLineParallel(i) => {
                let c = &self.sketch.arc_line_parallel[i];
                arcs.push(c.arc); lines.push(c.line);
            }
            ConstraintId::ArcArcParallel(i) => {
                let c = &self.sketch.arc_arc_parallel[i];
                arcs.push(c.a); arcs.push(c.b);
            }
            ConstraintId::Perpendicular(i) => {
                let c = &self.sketch.perpendicular[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::EqualLength(i) => {
                let c = &self.sketch.equal_length[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::EqualRadius(i) => {
                let c = &self.sketch.equal_radius[i];
                arcs.push(c.a); arcs.push(c.b);
            }
            ConstraintId::Concentric(i) => {
                let c = &self.sketch.concentric[i];
                arcs.push(c.a); arcs.push(c.b);
            }
            ConstraintId::TangentLA(i) => {
                let c = &self.sketch.tangent_la[i];
                lines.push(c.line); arcs.push(c.arc);
            }
            ConstraintId::TangentAA(i) => {
                let c = &self.sketch.tangent_aa[i];
                arcs.push(c.a); arcs.push(c.b);
            }
            ConstraintId::Collinear(i) => {
                let c = &self.sketch.collinear[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::Symmetry(i) => {
                let c = &self.sketch.symmetry_ll[i];
                lines.push(c.a); lines.push(c.b); lines.push(c.c);
            }
            ConstraintId::SymmetryPP(i) => {
                let c = &self.sketch.symmetry_pp[i];
                lines.push(c.line);
                // Resolve helper points to the specific endpoints they bridge to
                for pt in [c.a, c.c] {
                    if self.sketch.points.get(pt).is_some_and(|p| p.helper) {
                        let mut found = false;
                        for cc in &self.sketch.coincident_lp1 { if cc.point == pt { e.line_p1s.push(cc.line); found = true; break; } }
                        if !found { for cc in &self.sketch.coincident_lp2 { if cc.point == pt { e.line_p2s.push(cc.line); found = true; break; } } }
                        if !found { for cc in &self.sketch.coincident_arc_center { if cc.point == pt { e.arc_centers.push(cc.arc); found = true; break; } } }
                        if !found { for cc in &self.sketch.coincident_arc_start { if cc.point == pt { e.arc_starts.push(cc.arc); found = true; break; } } }
                        if !found { for cc in &self.sketch.coincident_arc_end { if cc.point == pt { e.arc_ends.push(cc.arc); found = true; break; } } }
                        if !found { points.push(pt); }
                    } else {
                        points.push(pt);
                    }
                }
            }
            ConstraintId::SymmetryAA(i) => {
                let c = &self.sketch.symmetry_aa[i];
                lines.push(c.line);
                arcs.push(c.a);
                arcs.push(c.c);
            }
            ConstraintId::Midpoint(kind, i) => {
                match kind {
                    MidpointKind::Point => { let c = &self.sketch.midpoint[i]; lines.push(c.line); }
                    MidpointKind::LP1 => { let c = &self.sketch.midpoint_lp1[i]; lines.push(c.line); lines.push(c.target); }
                    MidpointKind::LP2 => { let c = &self.sketch.midpoint_lp2[i]; lines.push(c.line); lines.push(c.target); }
                    MidpointKind::ArcStart => { let c = &self.sketch.midpoint_arc_start[i]; arcs.push(c.arc); lines.push(c.line); }
                    MidpointKind::ArcEnd => { let c = &self.sketch.midpoint_arc_end[i]; arcs.push(c.arc); lines.push(c.line); }
                    MidpointKind::ArcPoint => { let c = &self.sketch.midpoint_arc_point[i]; arcs.push(c.arc); }
                    MidpointKind::LP1Arc => { let c = &self.sketch.midpoint_lp1_arc[i]; lines.push(c.line); arcs.push(c.arc); }
                    MidpointKind::LP2Arc => { let c = &self.sketch.midpoint_lp2_arc[i]; lines.push(c.line); arcs.push(c.arc); }
                    MidpointKind::ArcStartArc => { let c = &self.sketch.midpoint_arc_start_arc[i]; arcs.push(c.a); arcs.push(c.b); }
                    MidpointKind::ArcEndArc => { let c = &self.sketch.midpoint_arc_end_arc[i]; arcs.push(c.a); arcs.push(c.b); }
                }
            }
            ConstraintId::Coincident(kind, i) => {
                match kind {
                    CoincidentKind::LP1 => { let c = &self.sketch.coincident_lp1[i]; lines.push(c.line); }
                    CoincidentKind::LP2 => { let c = &self.sketch.coincident_lp2[i]; lines.push(c.line); }
                    CoincidentKind::LL11 | CoincidentKind::LL12 | CoincidentKind::LL21 | CoincidentKind::LL22 => {
                        let (a, b) = match kind {
                            CoincidentKind::LL11 => { let c = &self.sketch.coincident_ll11[i]; (c.a, c.b) }
                            CoincidentKind::LL12 => { let c = &self.sketch.coincident_ll12[i]; (c.a, c.b) }
                            CoincidentKind::LL21 => { let c = &self.sketch.coincident_ll21[i]; (c.a, c.b) }
                            CoincidentKind::LL22 => { let c = &self.sketch.coincident_ll22[i]; (c.a, c.b) }
                            _ => unreachable!(),
                        };
                        lines.push(a); lines.push(b);
                    }
                    CoincidentKind::PointOnLine => { let c = &self.sketch.point_on_line[i]; lines.push(c.line); }
                    CoincidentKind::PointOnArc => { let c = &self.sketch.point_on_arc[i]; arcs.push(c.arc); }
                    CoincidentKind::LP1OnLine => { let c = &self.sketch.line_p1_on_line[i]; lines.push(c.a); lines.push(c.b); }
                    CoincidentKind::LP2OnLine => { let c = &self.sketch.line_p2_on_line[i]; lines.push(c.a); lines.push(c.b); }
                    CoincidentKind::LP1OnArc => { let c = &self.sketch.line_p1_on_arc[i]; lines.push(c.line); arcs.push(c.arc); }
                    CoincidentKind::LP2OnArc => { let c = &self.sketch.line_p2_on_arc[i]; lines.push(c.line); arcs.push(c.arc); }
                    CoincidentKind::ArcCenter => { let c = &self.sketch.coincident_arc_center[i]; arcs.push(c.arc); }
                    CoincidentKind::ArcStart => { let c = &self.sketch.coincident_arc_start[i]; arcs.push(c.arc); }
                    CoincidentKind::ArcEnd => { let c = &self.sketch.coincident_arc_end[i]; arcs.push(c.arc); }
                    CoincidentKind::LP1ArcCenter | CoincidentKind::LP1ArcStart | CoincidentKind::LP1ArcEnd => {
                        match kind {
                            CoincidentKind::LP1ArcCenter => { let c = &self.sketch.coincident_lp1_arc_center[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP1ArcStart => { let c = &self.sketch.coincident_lp1_arc_start[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP1ArcEnd => { let c = &self.sketch.coincident_lp1_arc_end[i]; lines.push(c.line); arcs.push(c.arc); }
                            _ => unreachable!(),
                        }
                    }
                    CoincidentKind::LP2ArcCenter | CoincidentKind::LP2ArcStart | CoincidentKind::LP2ArcEnd => {
                        match kind {
                            CoincidentKind::LP2ArcCenter => { let c = &self.sketch.coincident_lp2_arc_center[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP2ArcStart => { let c = &self.sketch.coincident_lp2_arc_start[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP2ArcEnd => { let c = &self.sketch.coincident_lp2_arc_end[i]; lines.push(c.line); arcs.push(c.arc); }
                            _ => unreachable!(),
                        }
                    }
                    CoincidentKind::ArcCenterStart | CoincidentKind::ArcCenterEnd
                    | CoincidentKind::ArcStartCenter | CoincidentKind::ArcEndCenter
                    | CoincidentKind::ArcStartStart | CoincidentKind::ArcStartEnd
                    | CoincidentKind::ArcEndStart | CoincidentKind::ArcEndEnd => {
                        // These are all Arc-Arc; get both arcs
                        let (a, b) = match kind {
                            CoincidentKind::ArcCenterStart => { let c = &self.sketch.coincident_arc_center_start[i]; (c.a, c.b) }
                            CoincidentKind::ArcCenterEnd => { let c = &self.sketch.coincident_arc_center_end[i]; (c.a, c.b) }
                            CoincidentKind::ArcStartCenter => { let c = &self.sketch.coincident_arc_start_center[i]; (c.a, c.b) }
                            CoincidentKind::ArcEndCenter => { let c = &self.sketch.coincident_arc_end_center[i]; (c.a, c.b) }
                            CoincidentKind::ArcStartStart => { let c = &self.sketch.coincident_arc_start_start[i]; (c.a, c.b) }
                            CoincidentKind::ArcStartEnd => { let c = &self.sketch.coincident_arc_start_end[i]; (c.a, c.b) }
                            CoincidentKind::ArcEndStart => { let c = &self.sketch.coincident_arc_end_start[i]; (c.a, c.b) }
                            CoincidentKind::ArcEndEnd => { let c = &self.sketch.coincident_arc_end_end[i]; (c.a, c.b) }
                            _ => unreachable!(),
                        };
                        arcs.push(a); arcs.push(b);
                    }
                    CoincidentKind::PP => {} // point-point: no lines/arcs to highlight
                }
            }
            ConstraintId::HelperBridge(pt) => {
                // Find all lines/arcs connected through this helper point
                for c in &self.sketch.coincident_lp1 { if c.point == pt { lines.push(c.line); } }
                for c in &self.sketch.coincident_lp2 { if c.point == pt { lines.push(c.line); } }
                for c in &self.sketch.point_on_line { if c.point == pt { lines.push(c.line); } }
                for c in &self.sketch.point_on_arc { if c.point == pt { arcs.push(c.arc); } }
                for c in &self.sketch.coincident_arc_center { if c.point == pt { arcs.push(c.arc); } }
                for c in &self.sketch.coincident_arc_start { if c.point == pt { arcs.push(c.arc); } }
                for c in &self.sketch.coincident_arc_end { if c.point == pt { arcs.push(c.arc); } }
            }
        }
        e
    }

    // Delete a constraint by id
    fn delete_constraint(&mut self, id: ConstraintId) {
        self.begin_group();
        self.exec(Action::DeleteConstraint { id });
        self.selection.clear();
    }

    // Hit test for delete: returns target only if exactly one entity is in range.
    // Only standalone points and lines (body, not endpoints) are delete targets.
    #[allow(dead_code)]
    fn hit_test_delete(&self, sketch_pos: vect2d, threshold: f64) -> Option<DeleteTarget> {
        let mut targets: Vec<DeleteTarget> = Vec::new();

        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            if d < threshold { targets.push(DeleteTarget::Point(r)); }
        }

        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            if d < threshold { targets.push(DeleteTarget::Line(r)); }
        }

        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let (d, _) = point_to_arc_dist(sketch_pos, a);
            if d < threshold { targets.push(DeleteTarget::Arc(r)); }
        }

        if targets.len() == 1 { Some(targets[0]) } else { None }
    }

    // Fit all entities into view with margin
    pub fn fit_all(&mut self, rect: egui::Rect) {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        let mut has_any = false;
        let mut extend = |x: f64, y: f64| {
            has_any = true;
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        };

        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            extend(p.pos.value.x, p.pos.value.y);
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            extend(l.p1.value.x, l.p1.value.y);
            extend(l.p2.value.x, l.p2.value.y);
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            // Include start and end points
            let sp = arael_sketch_backend::geometry::arc_start_pos(a);
            let ep = arael_sketch_backend::geometry::arc_end_pos(a);
            extend(sp.x, sp.y);
            extend(ep.x, ep.y);
            // Include axis points (0, 90, 180, 270 deg) if within sweep range
            let sa = a.start_angle.value;
            let ea = a.end_angle.value;
            let sweep = ea - sa;
            let norm_in_sweep = |angle: f64| -> bool {
                if sweep.abs() >= std::f64::consts::TAU - 1e-9 { return true; }
                let d = (angle - sa) % std::f64::consts::TAU;
                let d = if sweep > 0.0 {
                    if d < 0.0 { d + std::f64::consts::TAU } else { d }
                } else {
                    if d > 0.0 { d - std::f64::consts::TAU } else { d }
                };
                if sweep > 0.0 { d >= 0.0 && d <= sweep }
                else { d <= 0.0 && d >= sweep }
            };
            for k in 0..4 {
                let axis_angle = k as f64 * std::f64::consts::FRAC_PI_2;
                if norm_in_sweep(axis_angle) {
                    let pt = arael_sketch_backend::geometry::arc_point_at(a, axis_angle);
                    extend(pt.x, pt.y);
                }
            }
            // Sample every 16 degrees along the arc
            let steps = ((sweep.abs().to_degrees() / 16.0).ceil() as usize).max(1);
            for i in 1..steps {
                let t = sa + sweep * (i as f64 / steps as f64);
                let pt = arael_sketch_backend::geometry::arc_point_at(a, t);
                extend(pt.x, pt.y);
            }
        }

        if !has_any { return; }

        // Add small padding if all points are coincident
        if max_x - min_x < 1e-6 { min_x -= 1.0; max_x += 1.0; }
        if max_y - min_y < 1e-6 { min_y -= 1.0; max_y += 1.0; }

        let margin = 0.08; // 8% margin on each side
        let w = rect.width();
        let h = rect.height();
        let span_x = (max_x - min_x) as f32;
        let span_y = (max_y - min_y) as f32;

        // Scale to fit with margin
        let usable_w = w * (1.0 - 2.0 * margin);
        let usable_h = h * (1.0 - 2.0 * margin);
        self.scale = (usable_w / span_x).min(usable_h / span_y).clamp(1e-4, 1e7);

        // Center: sketch center -> screen center
        let cx = (min_x + max_x) as f32 / 2.0;
        let cy = (min_y + max_y) as f32 / 2.0;
        self.offset.x = rect.center().x - cx * self.scale;
        self.offset.y = rect.center().y + cy * self.scale; // y flipped
    }

    // Compute which points/endpoints are transitively locked via coincident chains.
    // Returns (point_locked, line_p1_locked, line_p2_locked, arc_c_locked) as HashSets of Refs.
    pub fn compute_locked_sets(&self) -> (
        std::collections::HashSet<u32>,  // locked point indices
        std::collections::HashSet<u32>,  // locked line indices (p1)
        std::collections::HashSet<u32>,  // locked line indices (p2)
        std::collections::HashSet<u32>,  // locked arc indices (center)
    ) {
        // Flat IDs: points, line p1s, line p2s, arc centers, arc starts, arc ends
        let np = self.sketch.points.slot_count();
        let nl = self.sketch.lines.slot_count();
        let na = self.sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;

        let mut parent: Vec<usize> = (0..total).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
            x
        };
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb { parent[ra] = rb; }
        };

        let pt_id = |r: Ref<Point>| r.index() as usize;
        let lp1_id = |r: Ref<Line>| np + r.index() as usize;
        let lp2_id = |r: Ref<Line>| np + nl + r.index() as usize;
        let ac_id = |r: Ref<Arc>| np + 2 * nl + r.index() as usize;
        let as_id = |r: Ref<Arc>| np + 2 * nl + na + r.index() as usize;
        let ae_id = |r: Ref<Arc>| np + 2 * nl + 2 * na + r.index() as usize;

        // Point-Point, Line-Point, Line-Line
        for c in &self.sketch.coincident_pp { union(&mut parent, pt_id(c.a), pt_id(c.b)); }
        for c in &self.sketch.coincident_lp1 { union(&mut parent, lp1_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_lp2 { union(&mut parent, lp2_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_ll11 { union(&mut parent, lp1_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll12 { union(&mut parent, lp1_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_ll21 { union(&mut parent, lp2_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll22 { union(&mut parent, lp2_id(c.a), lp2_id(c.b)); }
        // Point-Arc
        for c in &self.sketch.coincident_arc_center { union(&mut parent, pt_id(c.point), ac_id(c.arc)); }
        for c in &self.sketch.coincident_arc_start { union(&mut parent, pt_id(c.point), as_id(c.arc)); }
        for c in &self.sketch.coincident_arc_end { union(&mut parent, pt_id(c.point), ae_id(c.arc)); }
        // Line-Arc
        for c in &self.sketch.coincident_lp1_arc_center { union(&mut parent, lp1_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_center { union(&mut parent, lp2_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_start { union(&mut parent, lp1_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_start { union(&mut parent, lp2_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_end { union(&mut parent, lp1_id(c.line), ae_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_end { union(&mut parent, lp2_id(c.line), ae_id(c.arc)); }
        // Arc-Arc
        for c in &self.sketch.concentric { union(&mut parent, ac_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_start { union(&mut parent, ac_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_end { union(&mut parent, ac_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_center { union(&mut parent, as_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_center { union(&mut parent, ae_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_start { union(&mut parent, as_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_end { union(&mut parent, as_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_start { union(&mut parent, ae_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_end { union(&mut parent, ae_id(c.a), ae_id(c.b)); }

        // Find locked roots
        let mut locked_roots: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.constraints.has_fix_x && p.constraints.has_fix_y {
                locked_roots.insert(find(&mut parent, pt_id(r)));
            }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if !l.p1.optimize { locked_roots.insert(find(&mut parent, lp1_id(r))); }
            if !l.p2.optimize { locked_roots.insert(find(&mut parent, lp2_id(r))); }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            if !a.center.optimize { locked_roots.insert(find(&mut parent, ac_id(r))); }
        }

        // Collect locked vertices
        let mut pt_locked = std::collections::HashSet::new();
        let mut l_p1_locked = std::collections::HashSet::new();
        let mut l_p2_locked = std::collections::HashSet::new();
        let mut arc_c_locked = std::collections::HashSet::new();

        for r in self.sketch.points.refs() {
            if locked_roots.contains(&find(&mut parent, pt_id(r))) {
                pt_locked.insert(r.index());
            }
        }
        for r in self.sketch.lines.refs() {
            if locked_roots.contains(&find(&mut parent, lp1_id(r))) {
                l_p1_locked.insert(r.index());
            }
            if locked_roots.contains(&find(&mut parent, lp2_id(r))) {
                l_p2_locked.insert(r.index());
            }
        }
        for r in self.sketch.arcs.refs() {
            if locked_roots.contains(&find(&mut parent, ac_id(r))) {
                arc_c_locked.insert(r.index());
            }
        }

        (pt_locked, l_p1_locked, l_p2_locked, arc_c_locked)
    }

    // Check if a specific line endpoint is selected
    pub fn is_endpoint_selected(&self, line_ref: Ref<Line>, is_p1: bool) -> bool {
        self.selection.iter().any(|s| {
            if is_p1 {
                *s == Selection::LineP1(line_ref)
            } else {
                *s == Selection::LineP2(line_ref)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// eframe App impl -- placed in a separate file would be too large, include inline
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut file_path: Option<String> = None;
    let mut verbose = false;
    let mut empty = false;
    let mut dark = false;
    let mut mcp_addr: Option<std::net::SocketAddr> = None;
    let mut mcp_verbose = false;
    let mut mcp_allow_all = false;
    let mut echo_stdout = false;
    let mut no_gui = false;
    let mut script_path: Option<String> = None;
    let mut drag_raw = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                eprintln!("Usage: arael-sketch [OPTIONS] [FILE.json]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --verbose, -v   Print solver iterations");
                eprintln!("  --empty         Start with empty sketch");
                eprintln!("  --dark          Start in dark mode");
                eprintln!("  --script FILE   Execute commands from file at startup");
                eprintln!("  --stdout        Echo command output to stdout");
                eprintln!("  --nogui         Run without GUI (use with --script)");
                eprintln!("  --drag-raw      Hard-pin drag: deforms sketch when cursor target is infeasible");
                eprintln!("                  (default is soft: sketch stays at cost ~ 0, point lags if pinned)");
                eprintln!("  --mcp [addr]    Start MCP server (default 127.0.0.1:8585)");
                eprintln!("  --mcp-verbose   Log all MCP traffic to stdout");
                eprintln!("  --mcp-allow-all Auto-approve MCP OAuth connections");
                eprintln!("  --help, -h      Show this help");
                std::process::exit(0);
            }
            "--verbose" | "-v" => verbose = true,
            "--empty" => empty = true,
            "--dark" => dark = true,
            "--mcp-verbose" => mcp_verbose = true,
            "--mcp-allow-all" => mcp_allow_all = true,
            "--stdout" => echo_stdout = true,
            "--nogui" => no_gui = true,
            "--drag-raw" => drag_raw = true,
            "--script" => {
                if i + 1 < args.len() {
                    i += 1;
                    script_path = Some(args[i].clone());
                } else {
                    eprintln!("--script requires a file path");
                    std::process::exit(1);
                }
            }
            "--mcp" => {
                // Check if next arg is an address
                let addr_str = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    args[i].as_str()
                } else {
                    "127.0.0.1:8585"
                };
                // Parse: could be "host:port" or just "port"
                mcp_addr = Some(if let Ok(port) = addr_str.parse::<u16>() {
                    std::net::SocketAddr::from(([127, 0, 0, 1], port))
                } else {
                    addr_str.parse().unwrap_or_else(|e| {
                        eprintln!("Invalid MCP address '{}': {}", addr_str, e);
                        std::process::exit(1);
                    })
                });
            }
            arg if !arg.starts_with('-') => file_path = Some(arg.to_string()),
            other => {
                eprintln!("Unknown option: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut app = if let Some(ref path) = file_path {
        let json = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}", path, e);
            std::process::exit(1);
        });
        let mut app = EditorApp::default();
        app.load_from_json(&json);
        app
    } else if empty {
        let mut app = EditorApp::default();
        let empty_sketch = serde_json::to_string(&Sketch::new()).unwrap();
        app.load_from_json(&empty_sketch);
        app
    } else {
        EditorApp::default()
    };

    if verbose {
        arael_sketch_solver::set_verbose(true);
    }
    if echo_stdout {
        app.echo_stdout = true;
    }
    if drag_raw {
        app.drag_raw = true;
    }
    if dark {
        app.dark_mode = true;
        app.colors = ColorScheme::dark();
    }
    if let Some(ref script) = script_path {
        let content = std::fs::read_to_string(script).unwrap_or_else(|e| {
            eprintln!("Failed to read script {}: {}", script, e);
            std::process::exit(1);
        });
        // Join non-comment lines with ; for single-batch execution
        let commands: Vec<&str> = content.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        let line_count = commands.len();
        let batch = commands.join(";");
        let start = web_time::Instant::now();
        app.run_commands(&batch);
        let elapsed = start.elapsed();
        eprintln!("Script {} executed {} commands in {:.3}s", script, line_count, elapsed.as_secs_f64());
    }
    if no_gui {
        return Ok(());
    }
    if let Some(addr) = mcp_addr {
        let egui_ctx = std::sync::Arc::clone(&app.egui_ctx);
        let wake: arael_sketch_backend::mcp_server::WakeFn = std::sync::Arc::new(move || {
            if let Some(ctx) = egui_ctx.lock().unwrap().as_ref() {
                ctx.request_repaint();
            }
        });
        app.mcp_rx = Some(arael_sketch_backend::mcp_server::start(addr, mcp_verbose, mcp_allow_all, wake));
    }
    app.compute_dof_async();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_title("Arael Sketch Editor")
            .with_app_id("arael-sketch-editor"),
        ..Default::default()
    };
    eframe::run_native(
        "Arael Sketch Editor",
        options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;
    let web_options = eframe::WebOptions::default();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document.get_element_by_id("arael_canvas").unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(EditorApp::default()))),
            )
            .await
            .expect("failed to start eframe");
    });
}
