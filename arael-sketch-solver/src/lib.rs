//! 2D parametric constraint-based sketch solver.
//!
//! Work in progress -- a parametric CAD sketching tool built on the arael
//! optimization framework. Draw geometry, apply constraints, and the
//! solver keeps everything consistent in real time.
//!
//! # Entities
//!
//! Three entity types, each owning its own parameters:
//!
//! - [`Point`] -- 2D position (x, y)
//! - [`Line`] -- two endpoints p1, p2 (4 params), plus optional
//!   horizontal/vertical/length constraints
//! - [`Arc`] -- center, radius, start/end angle (5 params for arcs,
//!   3 for circles where angles are fixed)
//!
//! Shared geometry (e.g. two lines meeting at a point) is enforced via
//! coincident constraints, not shared references.
//!
//! # Constraints
//!
//! Over 40 constraint types including: coincident (point-point, line-line,
//! point-on-line, point-on-arc), parallel, perpendicular, tangent,
//! equal length/radius, distance, horizontal/vertical distance, and more.
//! All constraints are symbolically differentiated at compile time.
//!
//! # Solving
//!
//! The [`Sketch::solve()`] method runs Levenberg-Marquardt optimization with
//! drift regularization. Locking: use `Param::fixed(value)` to pin
//! individual parameters. The solver skips fixed params entirely.

pub mod dimensions;
pub use dimensions::*;
pub mod metas;
pub use metas::*;
pub mod symbol_bag;
pub use symbol_bag::SymbolBag;
pub mod expr_constraint;
pub use expr_constraint::{ExpressionConstraint, ExprRewrite, rewrite_expr_symbols};
pub mod blocker;
pub use blocker::{BlockerReport, analyze as analyze_blockers};
pub mod drag;
pub use drag::{DragApparatus, DragTarget};
pub mod probe;
pub mod registry;
pub use arael::rank::RankResult;
pub use registry::{CollectionMeta, ConstraintArenas, ConstraintCollection, EndpointRole,
    RefRole, SketchConstraint, decode_endpoint};

use arael::simple_lm::RootProblem;
use arael::model::{CrossBlock, JacobianModel, Param, SelfBlock, TripletBlock};

/// Wall-clock lap timer for verbose tracing. The clock is polled only
/// when verbose() was on at creation: otherwise every method is a no-op
/// returning `Duration::ZERO` and `on()` is false. Uses web_time, so
/// polling is safe under wasm as well.
pub struct Timer {
    start: Option<web_time::Instant>,
    checkpoint: Option<web_time::Instant>,
}

impl Timer {
    pub fn new() -> Self {
        let now = verbose().then(web_time::Instant::now);
        Self { start: now, checkpoint: now }
    }

    /// True when the timer is live; gate trace prints on this.
    pub fn on(&self) -> bool {
        self.start.is_some()
    }

    /// Time since the previous lap (or creation).
    pub fn lap(&mut self) -> std::time::Duration {
        match self.checkpoint.as_mut() {
            Some(cp) => {
                let now = web_time::Instant::now();
                let dur = now.duration_since(*cp);
                *cp = now;
                dur
            }
            None => std::time::Duration::ZERO,
        }
    }

    /// Time since creation.
    pub fn total(&self) -> std::time::Duration {
        self.start.map(|s| s.elapsed()).unwrap_or(std::time::Duration::ZERO)
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Solve tracing, on for the life of the process.
///
/// Global because it belongs to the session and not to the sketch: clear,
/// undo, redo and load all replace the sketch whole, so a flag living on it
/// is lost by whichever of them runs next. Threading it through every call
/// that might print was the alternative, and it swamped the signatures.
static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn solve tracing on or off.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether solve tracing is on.
pub fn verbose() -> bool {
    VERBOSE.load(std::sync::atomic::Ordering::Relaxed)
}
use arael::vect::vect2d;
use arael::refs::{Ref, Arena};

// Entity and constraint types must share one module scope because the
// #[arael::model] macro emits private `_PARAM_COUNT` constants that
// CrossBlock<A, B> expansions need to reference.
include!("entities.rs");
// arc_math.rs must expand before constraints.rs: its #[arael::function]
// registrations are used by the constraint bodies there.
include!("arc_math.rs");
include!("constraints.rs");

// ---------------------------------------------------------------------------
// DOF analysis
// ---------------------------------------------------------------------------

/// Snapshot returned by `Sketch::add_drag_auto_anchors` and consumed
/// by `Sketch::remove_drag_auto_anchors` to roll auto-anchors back at
/// drag-end. Cloneable so the GUI can also roll anchors back from a
/// serialized clone of the live sketch.
#[derive(Clone)]
pub struct DragAutoAnchorState {
    helper_points: std::vec::Vec<arael::refs::Ref<Point>>,
}

/// Result of DOF (degrees of freedom) analysis.
pub struct DofResult {
    /// Number of unconstrained degrees of freedom.
    pub dof: usize,
    /// Parameter names indexed by param index (only filled when analyze=true).
    pub param_names: Vec<String>,
    /// Eigenvalues from Hessian decomposition (only filled when analyze=true).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors, one per eigenvalue (only filled when analyze=true).
    pub eigenvectors: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(root, extended, jacobian)]
pub struct Sketch {
    pub points: Arena<Point>,
    pub lines: Arena<Line>,
    pub arcs: Arena<Arc>,
    // Solver parameters
    pub drift_isigma: f64,
    pub constraint_isigma: f64,
    /// Minimum length threshold for soft Heaviside penalties on line length,
    /// arc radius, and tangent projection.
    #[serde(default = "default_min_length")]
    pub min_length: f64,
    // Auto-naming counters
    pub next_point_id: u32,
    pub next_line_id: u32,
    pub next_arc_id: u32,
    // Cross-constraint collections
    pub coincident_pp: std::vec::Vec<CoincidentPP>,
    pub coincident_lp1: std::vec::Vec<CoincidentLP1>,
    pub coincident_lp2: std::vec::Vec<CoincidentLP2>,
    pub coincident_ll11: std::vec::Vec<CoincidentLL11>,
    pub coincident_ll12: std::vec::Vec<CoincidentLL12>,
    pub coincident_ll21: std::vec::Vec<CoincidentLL21>,
    pub coincident_ll22: std::vec::Vec<CoincidentLL22>,
    pub distance_pp: std::vec::Vec<DistancePP>,
    pub hdistance_pp: std::vec::Vec<HorizontalDistancePP>,
    pub vdistance_pp: std::vec::Vec<VerticalDistancePP>,
    pub point_on_line: std::vec::Vec<PointOnLine>,
    pub midpoint: std::vec::Vec<MidpointConstraint>,
    pub midpoint_lp1: std::vec::Vec<MidpointLP1>,
    pub midpoint_lp2: std::vec::Vec<MidpointLP2>,
    pub midpoint_arc_start: std::vec::Vec<MidpointArcStart>,
    pub midpoint_arc_end: std::vec::Vec<MidpointArcEnd>,
    #[serde(default)]
    pub midpoint_arc_point: std::vec::Vec<MidpointArcPoint>,
    #[serde(default)]
    pub midpoint_lp1_arc: std::vec::Vec<MidpointLP1Arc>,
    #[serde(default)]
    pub midpoint_lp2_arc: std::vec::Vec<MidpointLP2Arc>,
    #[serde(default)]
    pub midpoint_arc_start_arc: std::vec::Vec<MidpointArcStartArc>,
    #[serde(default)]
    pub midpoint_arc_end_arc: std::vec::Vec<MidpointArcEndArc>,
    pub point_on_arc: std::vec::Vec<PointOnArc>,
    pub parallel: std::vec::Vec<Parallel>,
    pub perpendicular: std::vec::Vec<Perpendicular>,
    #[serde(default)]
    pub arc_line_parallel: std::vec::Vec<ArcLineParallel>,
    #[serde(default)]
    pub arc_arc_parallel: std::vec::Vec<ArcArcParallel>,
    pub collinear: std::vec::Vec<Collinear>,
    pub equal_length: std::vec::Vec<EqualLength>,
    pub angle: std::vec::Vec<AngleConstraint>,
    pub tangent_la: std::vec::Vec<TangentLA>,
    pub concentric: std::vec::Vec<Concentric>,
    pub equal_radius: std::vec::Vec<EqualRadius>,
    pub tangent_aa: std::vec::Vec<TangentAA>,
    pub symmetry_ll: std::vec::Vec<SymmetryLL>,
    #[serde(default)]
    pub symmetry_pp: std::vec::Vec<SymmetryPP>,
    #[serde(default)]
    pub symmetry_aa: std::vec::Vec<SymmetryAA>,
    pub distance_pl: std::vec::Vec<DistancePL>,
    pub distance_lp1l: std::vec::Vec<DistanceLP1L>,
    pub distance_lp2l: std::vec::Vec<DistanceLP2L>,
    pub distance_arc_center_l: std::vec::Vec<DistanceArcCenterL>,
    pub distance_arc_start_l: std::vec::Vec<DistanceArcStartL>,
    pub distance_arc_end_l: std::vec::Vec<DistanceArcEndL>,
    pub line_p1_on_line: std::vec::Vec<LineP1OnLine>,
    pub line_p2_on_line: std::vec::Vec<LineP2OnLine>,
    pub coincident_arc_center: std::vec::Vec<CoincidentArcCenter>,
    pub coincident_arc_start: std::vec::Vec<CoincidentArcStart>,
    pub coincident_arc_end: std::vec::Vec<CoincidentArcEnd>,
    // Line endpoint <-> Arc point
    pub coincident_lp1_arc_center: std::vec::Vec<CoincidentLP1ArcCenter>,
    pub coincident_lp2_arc_center: std::vec::Vec<CoincidentLP2ArcCenter>,
    pub coincident_lp1_arc_start: std::vec::Vec<CoincidentLP1ArcStart>,
    pub coincident_lp2_arc_start: std::vec::Vec<CoincidentLP2ArcStart>,
    pub coincident_lp1_arc_end: std::vec::Vec<CoincidentLP1ArcEnd>,
    pub coincident_lp2_arc_end: std::vec::Vec<CoincidentLP2ArcEnd>,
    // Arc-Arc endpoint
    pub coincident_arc_center_start: std::vec::Vec<CoincidentArcCenterStart>,
    pub coincident_arc_center_end: std::vec::Vec<CoincidentArcCenterEnd>,
    pub coincident_arc_start_center: std::vec::Vec<CoincidentArcStartCenter>,
    pub coincident_arc_end_center: std::vec::Vec<CoincidentArcEndCenter>,
    pub coincident_arc_start_start: std::vec::Vec<CoincidentArcStartStart>,
    pub coincident_arc_start_end: std::vec::Vec<CoincidentArcStartEnd>,
    pub coincident_arc_end_start: std::vec::Vec<CoincidentArcEndStart>,
    pub coincident_arc_end_end: std::vec::Vec<CoincidentArcEndEnd>,
    pub line_p1_on_arc: std::vec::Vec<LineP1OnArc>,
    pub line_p2_on_arc: std::vec::Vec<LineP2OnArc>,
    pub distance_ll11: std::vec::Vec<DistanceLL11>,
    pub distance_ll12: std::vec::Vec<DistanceLL12>,
    pub distance_ll21: std::vec::Vec<DistanceLL21>,
    pub distance_ll22: std::vec::Vec<DistanceLL22>,
    pub distance_lp1: std::vec::Vec<DistanceLP1>,
    pub distance_lp2: std::vec::Vec<DistanceLP2>,
    pub distance_arc_center_p: std::vec::Vec<DistanceArcCenterP>,
    pub distance_arc_start_p: std::vec::Vec<DistanceArcStartP>,
    pub distance_arc_end_p: std::vec::Vec<DistanceArcEndP>,
    pub distance_arc_center_l1: std::vec::Vec<DistanceArcCenterL1>,
    pub distance_arc_center_l2: std::vec::Vec<DistanceArcCenterL2>,
    pub distance_arc_start_l1: std::vec::Vec<DistanceArcStartL1>,
    pub distance_arc_start_l2: std::vec::Vec<DistanceArcStartL2>,
    pub distance_arc_end_l1: std::vec::Vec<DistanceArcEndL1>,
    pub distance_arc_end_l2: std::vec::Vec<DistanceArcEndL2>,
    pub distance_aa_ce_ce: std::vec::Vec<DistanceAACeCe>,
    pub distance_aa_ce_s: std::vec::Vec<DistanceAACeS>,
    pub distance_aa_ce_e: std::vec::Vec<DistanceAACeE>,
    pub distance_aa_s_ce: std::vec::Vec<DistanceAASCe>,
    pub distance_aa_s_s: std::vec::Vec<DistanceAASS>,
    pub distance_aa_s_e: std::vec::Vec<DistanceAASE>,
    pub distance_aa_e_ce: std::vec::Vec<DistanceAAECe>,
    pub distance_aa_e_s: std::vec::Vec<DistanceAAES>,
    pub distance_aa_e_e: std::vec::Vec<DistanceAAEE>,
    /// Radial distance between two concentric arcs/circles.
    #[serde(default)] pub distance_concentric: std::vec::Vec<DistanceConcentric>,
    // Axis distance (horizontal/vertical unified)
    #[serde(default)] pub axis_distance_ll11: std::vec::Vec<AxisDistanceLL11>,
    #[serde(default)] pub axis_distance_ll12: std::vec::Vec<AxisDistanceLL12>,
    #[serde(default)] pub axis_distance_ll21: std::vec::Vec<AxisDistanceLL21>,
    #[serde(default)] pub axis_distance_ll22: std::vec::Vec<AxisDistanceLL22>,
    #[serde(default)] pub axis_distance_lp1: std::vec::Vec<AxisDistanceLP1>,
    #[serde(default)] pub axis_distance_lp2: std::vec::Vec<AxisDistanceLP2>,
    #[serde(default)] pub axis_distance_arc_center_p: std::vec::Vec<AxisDistanceArcCenterP>,
    #[serde(default)] pub axis_distance_arc_start_p: std::vec::Vec<AxisDistanceArcStartP>,
    #[serde(default)] pub axis_distance_arc_end_p: std::vec::Vec<AxisDistanceArcEndP>,
    #[serde(default)] pub axis_distance_arc_center_l1: std::vec::Vec<AxisDistanceArcCenterL1>,
    #[serde(default)] pub axis_distance_arc_center_l2: std::vec::Vec<AxisDistanceArcCenterL2>,
    #[serde(default)] pub axis_distance_arc_start_l1: std::vec::Vec<AxisDistanceArcStartL1>,
    #[serde(default)] pub axis_distance_arc_start_l2: std::vec::Vec<AxisDistanceArcStartL2>,
    #[serde(default)] pub axis_distance_arc_end_l1: std::vec::Vec<AxisDistanceArcEndL1>,
    #[serde(default)] pub axis_distance_arc_end_l2: std::vec::Vec<AxisDistanceArcEndL2>,
    #[serde(default)] pub axis_distance_aa_ce_ce: std::vec::Vec<AxisDistanceAACeCe>,
    #[serde(default)] pub axis_distance_aa_ce_s: std::vec::Vec<AxisDistanceAACeS>,
    #[serde(default)] pub axis_distance_aa_ce_e: std::vec::Vec<AxisDistanceAACeE>,
    #[serde(default)] pub axis_distance_aa_s_ce: std::vec::Vec<AxisDistanceAASCe>,
    #[serde(default)] pub axis_distance_aa_s_s: std::vec::Vec<AxisDistanceAASS>,
    #[serde(default)] pub axis_distance_aa_s_e: std::vec::Vec<AxisDistanceAASE>,
    #[serde(default)] pub axis_distance_aa_e_ce: std::vec::Vec<AxisDistanceAAECe>,
    #[serde(default)] pub axis_distance_aa_e_s: std::vec::Vec<AxisDistanceAAES>,
    #[serde(default)] pub axis_distance_aa_e_e: std::vec::Vec<AxisDistanceAAEE>,
    /// Endpoint on the normal of a line / arc at its endpoint.
    #[serde(default)] pub on_normal_ll: std::vec::Vec<EndpointOnNormalLL>,
    #[serde(default)] pub on_normal_aa: std::vec::Vec<EndpointOnNormalAA>,
    /// Pattern images: the copy is the source moved by a translation
    /// (world axes / a line's frame) or a rotation about a point.
    #[serde(default)] pub image_line_t: std::vec::Vec<ImageLineT>,
    #[serde(default)] pub image_line_tf: std::vec::Vec<ImageLineTF>,
    #[serde(default)] pub image_line_r: std::vec::Vec<ImageLineR>,
    #[serde(default)] pub image_arc_t: std::vec::Vec<ImageArcT>,
    #[serde(default)] pub image_arc_tf: std::vec::Vec<ImageArcTF>,
    #[serde(default)] pub image_arc_r: std::vec::Vec<ImageArcR>,
    #[serde(default)] pub image_point_t: std::vec::Vec<ImagePointT>,
    #[serde(default)] pub image_point_tf: std::vec::Vec<ImagePointTF>,
    #[serde(default)] pub image_point_r: std::vec::Vec<ImagePointR>,
    // Dimension annotations
    #[arael(skip)]
    pub dimensions: std::vec::Vec<Dimension>,
    pub next_dimension_id: u32,
    /// Meta-constraints (see [`metas`]): recorded operations that own
    /// what they created. Kept consistent by the backend's engines.
    #[arael(skip)]
    #[serde(default)]
    pub metas: std::vec::Vec<Meta>,
    #[serde(default)]
    pub next_meta_id: u32,
    /// Messages for the user raised while a mutation was applied (an
    /// offset dropped because its result was edited, say). Whoever ran
    /// the mutation drains them with [`take_notices`](Self::take_notices).
    #[arael(skip)]
    #[serde(skip)]
    pub notices: std::vec::Vec<String>,
    // Next auto-assigned numeric constraint id (C<nid>). 0 is reserved
    // as the "unassigned" sentinel picked up by assign_constraint_names().
    #[serde(default = "default_next_constraint_id")]
    pub next_constraint_id: u32,
    // User-defined parameters
    #[arael(skip)]
    #[serde(default)]
    pub user_params: std::vec::Vec<UserParam>,
    // Expression constraints (parametric dimensions)
    #[arael(skip)]
    #[serde(skip)]
    pub expr_constraints: std::vec::Vec<ExpressionConstraint>,
    #[arael(skip)]
    #[serde(skip)]
    symbol_bag: Option<SymbolBag>,
    // Shared TripletBlock for all expression constraints
    #[serde(skip)]
    pub expr_hb: TripletBlock<f64>,
    /// DOF cache, keyed to the structure generation it was computed
    /// at: a stale entry reads as absent through cached_dof(), so no
    /// consumer has to remember to clear it. Value-only mutation paths
    /// that change the instantaneous rank still clear explicitly
    /// (clear_cached_dof) where they know better.
    #[arael(skip)]
    #[serde(skip)]
    cached_dof: Option<(u64, usize)>,
    /// Rank-analysis cache (DOF plus the null-space basis the probes
    /// test against), keyed like cached_dof. Owned here so every
    /// consumer -- GUI display, drag-start probes, commands, MCP --
    /// reads one authority instead of keeping private copies.
    #[arael(skip)]
    #[serde(skip)]
    cached_rank: Option<(u64, arael::rank::RankResult)>,
    /// Bumped by every structural mutation (see `SketchCell::get_mut`).
    /// A cache records the generation it was built at and rebuilds when the
    /// two differ, so one signal serves every cache independently.
    #[arael(skip)]
    #[serde(skip)]
    structure_gen: u64,
}

fn default_min_length() -> f64 { 0.0001 }

/// Owns a [`Sketch`] and gates mutable access to it.
///
/// Reading is unrestricted -- the cell derefs to the sketch, so `cell.points`
/// and `cell.lines[r]` read as before. Mutating goes one of two ways:
///
/// - [`get_mut`](Self::get_mut) hands out `&mut Sketch` and bumps the
///   structural generation, so every cache rebuilds. This is the default and
///   it is always correct.
/// - [`mutate_values`](Self::mutate_values) does not bump it, for a mutation
///   that changes parameter VALUES only. Dragging is the case: a point moves
///   every frame while the entities, constraints and dimensions stand still,
///   and rebuilding the symbol bag and expression constraints per frame is
///   most of what a drag costs.
///
/// The closure form is deliberate. A promise made after the fact drifts from
/// the code it describes as soon as someone adds a line to the block; here the
/// promise and the body it covers cannot be separated.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct SketchCell {
    sketch: Sketch,
    /// Generation the derived state was last rebuilt at. Not serialized: a
    /// sketch that arrives by load or undo has no derived state at all.
    #[serde(skip)]
    prepared_gen: Option<u64>,
    /// A warm solving context and the generation it was built for. Holding it
    /// across solves reuses the ordering and symbolic factorization; the
    /// generation is what says the structure it learned still stands.
    #[serde(skip)]
    session: Option<(u64, arael::simple_lm::LmSession<f64, arael::simple_lm::SparseFaer<f64>>)>,
}

impl SketchCell {
    pub fn new(sketch: Sketch) -> Self {
        SketchCell { sketch, prepared_gen: None, session: None }
    }

    /// Read access. Prefer the deref -- this is for where inference needs it.
    pub fn get(&self) -> &Sketch {
        &self.sketch
    }

    /// Mutable access for a change that may alter the structure: entities,
    /// constraints, dimensions, expressions, or any `optimize` flag. Bumps the
    /// structural generation.
    pub fn get_mut(&mut self) -> &mut Sketch {
        self.sketch.structure_gen = self.sketch.structure_gen.wrapping_add(1);
        &mut self.sketch
    }

    /// Mutable access for a change that alters parameter VALUES only, leaving
    /// every cache valid. Anything structural inside the closure is a bug --
    /// the caches will not notice it.
    pub fn mutate_values<R>(&mut self, f: impl FnOnce(&mut Sketch) -> R) -> R {
        f(&mut self.sketch)
    }

    /// Solve. Structure-preserving by nature -- it moves parameters and the
    /// dimension values that read off them, and adds nothing -- so it does not
    /// bump the generation, and the derived state is rebuilt only when the
    /// structure has actually moved. Solving repeatedly is what a drag does.
    /// Compute (or serve the cached) DOF. A query: runs through the
    /// value-only door, so the derived state and warm session survive.
    pub fn dof(&mut self) -> Result<usize, String> {
        self.mutate_values(|s| s.dof())
    }

    /// Compute (or refresh) the rank analysis for the current
    /// structure generation, through the value-only door. Read the
    /// result back through the deref (`cached_rank`, `cached_dof`).
    pub fn ensure_rank(&mut self) -> Result<(), String> {
        self.mutate_values(|s| s.ensure_rank().map(|_| ()))
    }

    /// Validate an expression against the current sketch. A query in
    /// the gate's sense: it rebuilds only derived state.
    pub fn validate_expr(&mut self, expr_str: &str) -> Result<(), String> {
        self.mutate_values(|s| s.validate_expr(expr_str))
    }

    /// Serialize the parameters and evaluate the current cost --
    /// read-only in the gate's sense.
    pub fn current_cost(&mut self) -> f64 {
        self.mutate_values(|s| s.current_cost())
    }

    pub fn solve(&mut self) -> arael::simple_lm::LmResult<f64> {
        let cur = self.sketch.structure_gen;
        if self.prepared_gen != Some(cur) {
            let __t = web_time::Instant::now();
            self.sketch.prepare_derived();
            if verbose() { eprintln!("[PREPARE] {:.1} ms", __t.elapsed().as_secs_f64() * 1e3); }
            self.prepared_gen = Some(cur);
        }
        // A session that learned a different structure is not reusable.
        if self.session.as_ref().map(|(g, _)| *g) != Some(cur) {
            self.session = Some((
                cur,
                arael::simple_lm::LmSession::new(arael::simple_lm::SparseFaer::new()),
            ));
        }
        let SketchCell { sketch, session, .. } = self;
        let (_, sess) = session.as_mut().expect("just ensured above");
        sketch.solve_prepared_with(sess)
    }

    /// Take the sketch out, discarding the cell.
    pub fn into_inner(self) -> Sketch {
        self.sketch
    }
}

impl std::ops::Deref for SketchCell {
    type Target = Sketch;
    fn deref(&self) -> &Sketch {
        &self.sketch
    }
}

impl From<Sketch> for SketchCell {
    fn from(sketch: Sketch) -> Self {
        SketchCell { sketch, prepared_gen: None, session: None }
    }
}

fn default_next_constraint_id() -> u32 { 1 }

/// Format a synthetic constraint name for a flag-style constraint on a
/// named entity. Pattern: `C<entity><flag>`, e.g. "CL0H", "CL3V".
pub fn format_flag_name(entity_name: &str, flag: char) -> String {
    format!("C{}{}", entity_name, flag)
}

/// Parse a synthetic constraint name into its entity-name + flag-char
/// components. Returns None for names that don't match `C<entity>F`
/// where the first char is 'C' and the final char is an uppercase
/// ASCII flag tag ('H' or 'V' today).
pub fn parse_flag_name(token: &str) -> Option<(String, char)> {
    let bytes = token.as_bytes();
    if bytes.len() < 3 || bytes[0] != b'C' { return None; }
    let flag = bytes[bytes.len() - 1] as char;
    if !matches!(flag, 'H' | 'V') { return None; }
    let entity = &token[1..token.len() - 1];
    if entity.is_empty() { return None; }
    Some((entity.to_string(), flag))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Locate the cid of the axis-distance constraint (horizontal=true
/// for HDistance, false for VDistance) that matches the endpoint
/// pair. Mirrors the add path in
/// `arael-sketch/src/actions.rs::push_axis_distance`.
fn find_axis_cid(
    s: &Sketch,
    a: &crate::dimensions::DimensionEndpoint,
    b: &crate::dimensions::DimensionEndpoint,
    horizontal: bool,
) -> Option<u32> {
    use crate::dimensions::DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) => {
            if horizontal {
                s.hdistance_pp.iter().find(|c| c.a == *pa && c.b == *pb).map(|c| c.cid)
            } else {
                s.vdistance_pp.iter().find(|c| c.a == *pa && c.b == *pb).map(|c| c.cid)
            }
        }
        (LineP1(la), LineP1(lb)) => s.axis_distance_ll11.iter().find(|c| c.a == *la && c.b == *lb && c.horizontal == horizontal).map(|c| c.cid),
        (LineP1(la), LineP2(lb)) => s.axis_distance_ll12.iter().find(|c| c.a == *la && c.b == *lb && c.horizontal == horizontal).map(|c| c.cid),
        (LineP2(la), LineP1(lb)) => s.axis_distance_ll21.iter().find(|c| c.a == *la && c.b == *lb && c.horizontal == horizontal).map(|c| c.cid),
        (LineP2(la), LineP2(lb)) => s.axis_distance_ll22.iter().find(|c| c.a == *la && c.b == *lb && c.horizontal == horizontal).map(|c| c.cid),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) =>
            s.axis_distance_lp1.iter().find(|c| c.line == *l && c.point == *p && c.horizontal == horizontal).map(|c| c.cid),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) =>
            s.axis_distance_lp2.iter().find(|c| c.line == *l && c.point == *p && c.horizontal == horizontal).map(|c| c.cid),
        (ArcCenter(ar), Point(p)) | (Point(p), ArcCenter(ar)) =>
            s.axis_distance_arc_center_p.iter().find(|c| c.arc == *ar && c.point == *p && c.horizontal == horizontal).map(|c| c.cid),
        (ArcStart(ar), Point(p)) | (Point(p), ArcStart(ar)) =>
            s.axis_distance_arc_start_p.iter().find(|c| c.arc == *ar && c.point == *p && c.horizontal == horizontal).map(|c| c.cid),
        (ArcEnd(ar), Point(p)) | (Point(p), ArcEnd(ar)) =>
            s.axis_distance_arc_end_p.iter().find(|c| c.arc == *ar && c.point == *p && c.horizontal == horizontal).map(|c| c.cid),
        (ArcCenter(ar), LineP1(l)) | (LineP1(l), ArcCenter(ar)) =>
            s.axis_distance_arc_center_l1.iter().find(|c| c.arc == *ar && c.line == *l && c.horizontal == horizontal).map(|c| c.cid),
        (ArcCenter(ar), LineP2(l)) | (LineP2(l), ArcCenter(ar)) =>
            s.axis_distance_arc_center_l2.iter().find(|c| c.arc == *ar && c.line == *l && c.horizontal == horizontal).map(|c| c.cid),
        (ArcStart(ar), LineP1(l)) | (LineP1(l), ArcStart(ar)) =>
            s.axis_distance_arc_start_l1.iter().find(|c| c.arc == *ar && c.line == *l && c.horizontal == horizontal).map(|c| c.cid),
        (ArcStart(ar), LineP2(l)) | (LineP2(l), ArcStart(ar)) =>
            s.axis_distance_arc_start_l2.iter().find(|c| c.arc == *ar && c.line == *l && c.horizontal == horizontal).map(|c| c.cid),
        (ArcEnd(ar), LineP1(l)) | (LineP1(l), ArcEnd(ar)) =>
            s.axis_distance_arc_end_l1.iter().find(|c| c.arc == *ar && c.line == *l && c.horizontal == horizontal).map(|c| c.cid),
        (ArcEnd(ar), LineP2(l)) | (LineP2(l), ArcEnd(ar)) =>
            s.axis_distance_arc_end_l2.iter().find(|c| c.arc == *ar && c.line == *l && c.horizontal == horizontal).map(|c| c.cid),
        (ArcCenter(a), ArcCenter(b)) => s.axis_distance_aa_ce_ce.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcCenter(a), ArcStart(b))  => s.axis_distance_aa_ce_s.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcCenter(a), ArcEnd(b))    => s.axis_distance_aa_ce_e.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcStart(a), ArcCenter(b))  => s.axis_distance_aa_s_ce.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcStart(a), ArcStart(b))   => s.axis_distance_aa_s_s.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcStart(a), ArcEnd(b))     => s.axis_distance_aa_s_e.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcEnd(a), ArcCenter(b))    => s.axis_distance_aa_e_ce.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcEnd(a), ArcStart(b))     => s.axis_distance_aa_e_s.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
        (ArcEnd(a), ArcEnd(b))       => s.axis_distance_aa_e_e.iter().find(|c| c.a == *a && c.b == *b && c.horizontal == horizontal).map(|c| c.cid),
    }
}

/// Locate the cid of the `PointPointDistance` backing constraint for
/// a dimension. Each supported endpoint pair maps to one of the
/// `distance_*` collections (see `arael-sketch/src/actions.rs`
/// distance dispatch for the canonical routing).
fn find_distance_cid(
    s: &Sketch,
    a: &crate::dimensions::DimensionEndpoint,
    b: &crate::dimensions::DimensionEndpoint,
) -> Option<u32> {
    use crate::dimensions::DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) =>
            // Exact order first; the reversed pair as a fallback so a
            // swapped-operand constraint stays deletable.
            s.distance_pp.iter().find(|c| c.a == *pa && c.b == *pb)
                .or_else(|| s.distance_pp.iter().find(|c| c.a == *pb && c.b == *pa))
                .map(|c| c.cid),
        (LineP1(la), LineP1(lb)) => s.distance_ll11.iter().find(|c| c.a == *la && c.b == *lb).map(|c| c.cid),
        (LineP1(la), LineP2(lb)) => s.distance_ll12.iter().find(|c| c.a == *la && c.b == *lb).map(|c| c.cid),
        (LineP2(la), LineP1(lb)) => s.distance_ll21.iter().find(|c| c.a == *la && c.b == *lb).map(|c| c.cid),
        (LineP2(la), LineP2(lb)) => s.distance_ll22.iter().find(|c| c.a == *la && c.b == *lb).map(|c| c.cid),
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) =>
            s.distance_lp1.iter().find(|c| c.line == *l && c.point == *p).map(|c| c.cid),
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) =>
            s.distance_lp2.iter().find(|c| c.line == *l && c.point == *p).map(|c| c.cid),
        (ArcCenter(ar), Point(p)) | (Point(p), ArcCenter(ar)) =>
            s.distance_arc_center_p.iter().find(|c| c.arc == *ar && c.point == *p).map(|c| c.cid),
        (ArcStart(ar), Point(p)) | (Point(p), ArcStart(ar)) =>
            s.distance_arc_start_p.iter().find(|c| c.arc == *ar && c.point == *p).map(|c| c.cid),
        (ArcEnd(ar), Point(p)) | (Point(p), ArcEnd(ar)) =>
            s.distance_arc_end_p.iter().find(|c| c.arc == *ar && c.point == *p).map(|c| c.cid),
        (ArcCenter(a), ArcCenter(b)) => s.distance_aa_ce_ce.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcCenter(a), ArcStart(b))  => s.distance_aa_ce_s.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcCenter(a), ArcEnd(b))    => s.distance_aa_ce_e.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcStart(a), ArcCenter(b))  => s.distance_aa_s_ce.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcStart(a), ArcStart(b))   => s.distance_aa_s_s.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcStart(a), ArcEnd(b))     => s.distance_aa_s_e.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcEnd(a), ArcCenter(b))    => s.distance_aa_e_ce.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcEnd(a), ArcStart(b))     => s.distance_aa_e_s.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcEnd(a), ArcEnd(b))       => s.distance_aa_e_e.iter().find(|c| c.a == *a && c.b == *b).map(|c| c.cid),
        (ArcCenter(ar), LineP1(l)) | (LineP1(l), ArcCenter(ar)) =>
            s.distance_arc_center_l1.iter().find(|c| c.arc == *ar && c.line == *l).map(|c| c.cid),
        (ArcCenter(ar), LineP2(l)) | (LineP2(l), ArcCenter(ar)) =>
            s.distance_arc_center_l2.iter().find(|c| c.arc == *ar && c.line == *l).map(|c| c.cid),
        (ArcStart(ar), LineP1(l)) | (LineP1(l), ArcStart(ar)) =>
            s.distance_arc_start_l1.iter().find(|c| c.arc == *ar && c.line == *l).map(|c| c.cid),
        (ArcStart(ar), LineP2(l)) | (LineP2(l), ArcStart(ar)) =>
            s.distance_arc_start_l2.iter().find(|c| c.arc == *ar && c.line == *l).map(|c| c.cid),
        (ArcEnd(ar), LineP1(l)) | (LineP1(l), ArcEnd(ar)) =>
            s.distance_arc_end_l1.iter().find(|c| c.arc == *ar && c.line == *l).map(|c| c.cid),
        (ArcEnd(ar), LineP2(l)) | (LineP2(l), ArcEnd(ar)) =>
            s.distance_arc_end_l2.iter().find(|c| c.arc == *ar && c.line == *l).map(|c| c.cid),
    }
}

/// Locate the cid of the point-to-line distance backing constraint
/// for a dimension. Covers `distance_pl`, `distance_lp1l`,
/// `distance_lp2l`, `distance_arc_center_l`, `distance_arc_start_l`,
/// `distance_arc_end_l`.
fn find_point_line_cid(
    s: &Sketch,
    ep: &crate::dimensions::DimensionEndpoint,
    line: Ref<Line>,
) -> Option<u32> {
    use crate::dimensions::DimensionEndpoint::*;
    match ep {
        Point(p) => s.distance_pl.iter().find(|c| c.point == *p && c.line == line).map(|c| c.cid),
        LineP1(l) => s.distance_lp1l.iter().find(|c| c.a == *l && c.b == line).map(|c| c.cid),
        LineP2(l) => s.distance_lp2l.iter().find(|c| c.a == *l && c.b == line).map(|c| c.cid),
        ArcCenter(ar) => s.distance_arc_center_l.iter().find(|c| c.arc == *ar && c.line == line).map(|c| c.cid),
        ArcStart(ar) => s.distance_arc_start_l.iter().find(|c| c.arc == *ar && c.line == line).map(|c| c.cid),
        ArcEnd(ar) => s.distance_arc_end_l.iter().find(|c| c.arc == *ar && c.line == line).map(|c| c.cid),
    }
}

impl Sketch {
    /// Create an empty sketch with default solver parameters.
    pub fn new() -> Self {
        let drift_sigma = 1000.0_f64;
        Sketch {
            points: Arena::new(),
            lines: Arena::new(),
            arcs: Arena::new(),
            drift_isigma: 1.0 / drift_sigma,
            constraint_isigma: 1000.0, // tight constraints
            min_length: 0.0001,
            next_point_id: 0,
            next_line_id: 0,
            next_arc_id: 0,
            coincident_pp: Vec::new(),
            coincident_lp1: Vec::new(),
            coincident_lp2: Vec::new(),
            coincident_ll11: Vec::new(),
            coincident_ll12: Vec::new(),
            coincident_ll21: Vec::new(),
            coincident_ll22: Vec::new(),
            distance_pp: Vec::new(),
            hdistance_pp: Vec::new(),
            vdistance_pp: Vec::new(),
            point_on_line: Vec::new(),
            midpoint: Vec::new(),
            midpoint_lp1: Vec::new(),
            midpoint_lp2: Vec::new(),
            midpoint_arc_start: Vec::new(),
            midpoint_arc_end: Vec::new(),
            midpoint_arc_point: Vec::new(),
            midpoint_lp1_arc: Vec::new(),
            midpoint_lp2_arc: Vec::new(),
            midpoint_arc_start_arc: Vec::new(),
            midpoint_arc_end_arc: Vec::new(),
            point_on_arc: Vec::new(),
            parallel: Vec::new(),
            perpendicular: Vec::new(),
            arc_line_parallel: Vec::new(),
            arc_arc_parallel: Vec::new(),
            collinear: Vec::new(),
            equal_length: Vec::new(),
            angle: Vec::new(),
            tangent_la: Vec::new(),
            concentric: Vec::new(),
            equal_radius: Vec::new(),
            tangent_aa: Vec::new(),
            symmetry_ll: Vec::new(),
            symmetry_pp: Vec::new(),
            symmetry_aa: Vec::new(),
            distance_pl: Vec::new(),
            distance_lp1l: Vec::new(),
            distance_lp2l: Vec::new(),
            distance_arc_center_l: Vec::new(),
            distance_arc_start_l: Vec::new(),
            distance_arc_end_l: Vec::new(),
            line_p1_on_line: Vec::new(),
            line_p2_on_line: Vec::new(),
            coincident_arc_center: Vec::new(),
            coincident_arc_start: Vec::new(),
            coincident_arc_end: Vec::new(),
            coincident_lp1_arc_center: Vec::new(),
            coincident_lp2_arc_center: Vec::new(),
            coincident_lp1_arc_start: Vec::new(),
            coincident_lp2_arc_start: Vec::new(),
            coincident_lp1_arc_end: Vec::new(),
            coincident_lp2_arc_end: Vec::new(),
            coincident_arc_center_start: Vec::new(),
            coincident_arc_center_end: Vec::new(),
            coincident_arc_start_center: Vec::new(),
            coincident_arc_end_center: Vec::new(),
            coincident_arc_start_start: Vec::new(),
            coincident_arc_start_end: Vec::new(),
            coincident_arc_end_start: Vec::new(),
            coincident_arc_end_end: Vec::new(),
            line_p1_on_arc: Vec::new(),
            line_p2_on_arc: Vec::new(),
            distance_ll11: Vec::new(),
            distance_ll12: Vec::new(),
            distance_ll21: Vec::new(),
            distance_ll22: Vec::new(),
            distance_lp1: Vec::new(),
            distance_lp2: Vec::new(),
            distance_arc_center_p: Vec::new(),
            distance_arc_start_p: Vec::new(),
            distance_arc_end_p: Vec::new(),
            distance_arc_center_l1: Vec::new(),
            distance_arc_center_l2: Vec::new(),
            distance_arc_start_l1: Vec::new(),
            distance_arc_start_l2: Vec::new(),
            distance_arc_end_l1: Vec::new(),
            distance_arc_end_l2: Vec::new(),
            distance_aa_ce_ce: Vec::new(),
            distance_aa_ce_s: Vec::new(),
            distance_aa_ce_e: Vec::new(),
            distance_aa_s_ce: Vec::new(),
            distance_aa_s_s: Vec::new(),
            distance_aa_s_e: Vec::new(),
            distance_aa_e_ce: Vec::new(),
            distance_aa_e_s: Vec::new(),
            distance_aa_e_e: Vec::new(),
            distance_concentric: Vec::new(),
            axis_distance_ll11: Vec::new(),
            axis_distance_ll12: Vec::new(),
            axis_distance_ll21: Vec::new(),
            axis_distance_ll22: Vec::new(),
            axis_distance_lp1: Vec::new(),
            axis_distance_lp2: Vec::new(),
            axis_distance_arc_center_p: Vec::new(),
            axis_distance_arc_start_p: Vec::new(),
            axis_distance_arc_end_p: Vec::new(),
            axis_distance_arc_center_l1: Vec::new(),
            axis_distance_arc_center_l2: Vec::new(),
            axis_distance_arc_start_l1: Vec::new(),
            axis_distance_arc_start_l2: Vec::new(),
            axis_distance_arc_end_l1: Vec::new(),
            axis_distance_arc_end_l2: Vec::new(),
            axis_distance_aa_ce_ce: Vec::new(),
            axis_distance_aa_ce_s: Vec::new(),
            axis_distance_aa_ce_e: Vec::new(),
            axis_distance_aa_s_ce: Vec::new(),
            axis_distance_aa_s_s: Vec::new(),
            axis_distance_aa_s_e: Vec::new(),
            axis_distance_aa_e_ce: Vec::new(),
            axis_distance_aa_e_s: Vec::new(),
            axis_distance_aa_e_e: Vec::new(),
            on_normal_ll: Vec::new(),
            on_normal_aa: Vec::new(),
            image_line_t: Vec::new(),
            image_line_tf: Vec::new(),
            image_line_r: Vec::new(),
            image_arc_t: Vec::new(),
            image_arc_tf: Vec::new(),
            image_arc_r: Vec::new(),
            image_point_t: Vec::new(),
            image_point_tf: Vec::new(),
            image_point_r: Vec::new(),
            dimensions: Vec::new(),
            next_dimension_id: 0,
            metas: Vec::new(),
            next_meta_id: 0,
            notices: Vec::new(),
            next_constraint_id: 1,
            user_params: Vec::new(),
            expr_constraints: Vec::new(),
            symbol_bag: None,
            expr_hb: TripletBlock::new(),
            cached_dof: None,
            cached_rank: None,
            structure_gen: 0,
        }
    }

    /// The current structural generation. A cache stores this alongside what
    /// it built and rebuilds when the two no longer match.
    pub fn structure_gen(&self) -> u64 {
        self.structure_gen
    }

    /// Queue a message for whoever ran the current mutation.
    pub fn push_notice(&mut self, msg: impl Into<String>) {
        self.notices.push(msg.into());
    }

    /// Take the queued messages, leaving none.
    pub fn take_notices(&mut self) -> std::vec::Vec<String> {
        std::mem::take(&mut self.notices)
    }

    /// Index of the meta-constraint named `name` (`M<n>`), if any.
    pub fn find_meta(&self, name: &str) -> Option<usize> {
        self.metas.iter().position(|m| m.name == name)
    }

    /// Index of the meta-constraint with this id, if it still exists.
    pub fn meta_index(&self, mid: u32) -> Option<usize> {
        self.metas.iter().position(|m| m.mid == mid)
    }

    /// The cached DOF, if one was computed at the current structure
    /// generation.
    pub fn cached_dof(&self) -> Option<usize> {
        self.cached_dof
            .and_then(|(g, d)| (g == self.structure_gen).then_some(d))
    }

    /// Store the DOF for the current structure generation.
    pub fn set_cached_dof(&mut self, dof: usize) {
        self.cached_dof = Some((self.structure_gen, dof));
    }

    /// Drop the caches: for mutations that change the instantaneous
    /// rank without a structural change (the generation cannot tell).
    pub fn clear_cached_dof(&mut self) {
        self.cached_dof = None;
        self.cached_rank = None;
    }

    /// The cached rank analysis, if one was computed at the current
    /// structure generation.
    pub fn cached_rank(&self) -> Option<&arael::rank::RankResult> {
        self.cached_rank
            .as_ref()
            .and_then(|(g, rr)| (*g == self.structure_gen).then_some(rr))
    }

    /// Compute (or serve) the rank analysis for the current structure
    /// generation. Fills the DOF cache as a byproduct, so the display
    /// and the drag-start probes read the same computation.
    pub fn ensure_rank(&mut self) -> Result<&arael::rank::RankResult, String> {
        let sgen = self.structure_gen;
        if !matches!(&self.cached_rank, Some((g, _)) if *g == sgen) {
            let rr = self.rank_analysis()?;
            self.cached_rank = Some((sgen, rr));
        }
        Ok(&self.cached_rank.as_ref().unwrap().1)
    }

    /// Walk every Vec-stored constraint in canonical order and assign a
    /// numeric id (`C<nid>`) to any constraint whose nid is still the 0
    /// sentinel. Call at the tail of every mutating action and after
    /// loading a sketch so freshly-deserialised sketches pick up names.
    /// Mint permanent ids for dimensions that lack one (did == 0):
    /// max existing + 1 upward. Runs with constraint naming, so every
    /// action and load fixup covers it.
    pub fn assign_dimension_ids(&mut self) {
        let mut next = self.dimensions.iter().map(|d| d.did).max().unwrap_or(0) + 1;
        for d in &mut self.dimensions {
            if d.did == 0 {
                d.did = next;
                next += 1;
            }
        }
    }

    /// Index of the dimension carrying this did, if it still exists.
    pub fn dimension_index_by_did(&self, did: u32) -> Option<usize> {
        if did == 0 { return None; }
        self.dimensions.iter().position(|d| d.did == did)
    }

    pub fn assign_constraint_names(&mut self) {
        self.assign_dimension_ids();
        // Registry order is field-declaration order, so numbering is
        // stable and every collection participates -- constraint types
        // once missing from the hand-kept list stayed at nid 0 forever.
        let mut next = self.next_constraint_id;
        self.for_each_constraint_collection(|_, _, coll| {
            for i in 0..coll.len() {
                let c = coll.item_mut(i);
                if c.nid() == 0 {
                    c.set_nid(next);
                    next += 1;
                }
            }
        });
        self.next_constraint_id = next;
    }

    /// Add a free point at the given position.
    pub fn add_point(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("P{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::new(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: false, quiet: false, name,
            drag_pull: 0.0,
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Add a fixed (non-optimizable) point at the given position.
    pub fn add_point_fixed(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("P{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::fixed(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: false, quiet: false, name,
            drag_pull: 0.0,
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Add a helper point (auto-removed when no constraints reference it).
    pub fn add_helper_point(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("Pc{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::new(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: true, quiet: false, name,
            drag_pull: 0.0,
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Add a line with two free endpoints.
    pub fn add_line(&mut self, p1: vect2d, p2: vect2d) -> Ref<Line> {
        let name = format!("L{}", self.next_line_id);
        self.next_line_id += 1;
        self.lines.push(Line {
            p1: Param::new(p1),
            p2: Param::new(p2),
            constraints: LineConstraints { horizontal: false, vertical: false, has_length: false, length: 0.0, has_angle: false, target_angle: 0.0, h_dir_sign: f64::NAN, v_dir_sign: f64::NAN },
            style: LineStyle::Solid, construction: false, quiet: false, name,
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Add an arc or circle. When `closed` is true, start/end angles are
    /// fixed (not optimized) since they are meaningless for a full circle.
    pub fn add_arc(&mut self, center: vect2d, radius: f64, start: f64, end: f64, closed: bool) -> Ref<Arc> {
        self.add_arc_with_dir(center, radius, start, end, closed, true)
    }

    /// Add an arc with explicit direction. ccw=true means CCW from start to end.
    /// For CW arcs, end_angle is adjusted so that end - start < 0.
    pub fn add_arc_with_dir(&mut self, center: vect2d, radius: f64, start: f64, end: f64, closed: bool, ccw: bool) -> Ref<Arc> {
        // Ensure end - start has the correct sign for the arc direction
        let end = if !closed && !ccw && end > start {
            end - std::f64::consts::TAU
        } else if !closed && ccw && end < start {
            end + std::f64::consts::TAU
        } else {
            end
        };
        let name = format!("A{}", self.next_arc_id);
        self.next_arc_id += 1;
        self.arcs.push(Arc {
            center: Param::new(center),
            radius: Param::new(radius),
            radius_b: Param::new(radius),
            rotation: Param::fixed(0.0),
            start_angle: if closed { Param::fixed(start) } else { Param::new(start) },
            end_angle: if closed { Param::fixed(end) } else { Param::new(end) },
            closed, ccw,
            is_ellipse: false,
            style: LineStyle::Solid, construction: false, quiet: false, name,
            constraints: ArcConstraints {
                has_target_radius: false, target_radius: 0.0,
                has_target_radius_b: false, target_radius_b: 0.0,
                has_target_sweep: false, target_sweep: 0.0, sweep_sign: 1.0, has_target_rotation: false, target_rotation: 0.0,
            },
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Add an ellipse (closed) or elliptic arc. rx = semi-major, ry = semi-minor,
    /// rot = rotation angle of the ellipse axes.
    pub fn add_ellipse(&mut self, center: vect2d, rx: f64, ry: f64, rot: f64, closed: bool) -> Ref<Arc> {
        let name = format!("EA{}", self.next_arc_id);
        self.next_arc_id += 1;
        self.arcs.push(Arc {
            center: Param::new(center),
            radius: Param::new(rx),
            radius_b: Param::new(ry),
            rotation: Param::new(rot),
            start_angle: if closed { Param::fixed(0.0) } else { Param::new(0.0) },
            end_angle: if closed { Param::fixed(std::f64::consts::TAU) } else { Param::new(std::f64::consts::TAU) },
            closed,
            is_ellipse: true,
            ccw: true,
            style: LineStyle::Solid, construction: false, quiet: false, name,
            constraints: ArcConstraints {
                has_target_radius: false, target_radius: 0.0,
                has_target_radius_b: false, target_radius_b: 0.0,
                has_target_sweep: false, target_sweep: 0.0, sweep_sign: 1.0, has_target_rotation: false, target_rotation: 0.0,
            },
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Add a partial elliptic arc with explicit center parameterization.
    pub fn add_elliptic_arc(&mut self, center: vect2d, rx: f64, ry: f64,
        rot: f64, start: f64, end: f64, ccw: bool) -> Ref<Arc>
    {
        let end = if !ccw && end > start {
            end - std::f64::consts::TAU
        } else if ccw && end < start {
            end + std::f64::consts::TAU
        } else {
            end
        };
        let name = format!("EA{}", self.next_arc_id);
        self.next_arc_id += 1;
        self.arcs.push(Arc {
            center: Param::new(center),
            radius: Param::new(rx),
            radius_b: Param::new(ry),
            rotation: Param::new(rot),
            start_angle: Param::new(start),
            end_angle: Param::new(end),
            closed: false,
            is_ellipse: true,
            ccw,
            style: LineStyle::Solid, construction: false, quiet: false, name,
            constraints: ArcConstraints {
                has_target_radius: false, target_radius: 0.0,
                has_target_radius_b: false, target_radius_b: 0.0,
                has_target_sweep: false, target_sweep: 0.0, sweep_sign: 1.0, has_target_rotation: false, target_rotation: 0.0,
            },
            cid: 0, hb: SelfBlock::new(),
        })
    }

    /// Remove a point and all constraints referencing it.
    pub fn delete_point(&mut self, r: Ref<Point>) {
        self.dimensions.retain(|d| !d.kind.references_point(r));
        self.points.remove(r);
        self.for_each_constraint_collection(|_, _, coll| {
            coll.retain_constraints(&mut |c| !c.references_point(r));
        });
        self.cleanup_helper_points();
    }

    /// Remove a line and all constraints referencing it.
    pub fn delete_line(&mut self, r: Ref<Line>) {
        self.dimensions.retain(|d| !d.kind.references_line(r));
        self.lines.remove(r);
        self.for_each_constraint_collection(|_, _, coll| {
            coll.retain_constraints(&mut |c| !c.references_line(r));
        });
        self.cleanup_helper_points();
    }

    /// Remove an arc and all constraints referencing it.
    pub fn delete_arc(&mut self, r: Ref<Arc>) {
        self.dimensions.retain(|d| !d.kind.references_arc(r));
        self.arcs.remove(r);
        self.for_each_constraint_collection(|_, _, coll| {
            coll.retain_constraints(&mut |c| !c.references_arc(r));
        });
        self.cleanup_helper_points();
    }

    /// Remove helper points that are no longer needed.
    /// A helper is removed if it lost its bridge constraint (semantic origin
    /// gone) or has no purpose constraint. Cascades until stable.
    pub fn cleanup_helper_points(&mut self) {
        loop {
            // Classify every helper-point reference: coincidence
            // constraints are bridges (they say what the helper stands
            // for), everything else referencing it is a purpose. The
            // registry walk covers every collection, so a new
            // constraint type participates automatically.
            let mut has_bridge: std::collections::HashSet<u32> = std::collections::HashSet::new();
            let mut has_purpose: std::collections::HashSet<u32> = std::collections::HashSet::new();
            self.for_each_constraint_collection_ref(|arenas, meta, coll| {
                for i in 0..coll.len() {
                    coll.item(i).each_point_ref(&mut |p| {
                        if arenas.points.get(p).is_some_and(|pt| pt.helper) {
                            if meta.coincidence {
                                has_bridge.insert(p.index());
                            } else {
                                has_purpose.insert(p.index());
                            }
                        }
                    });
                }
            });

            // Remove helpers that lost their bridge OR have no purpose
            let to_remove: std::vec::Vec<Ref<Point>> = self.points.refs()
                .filter(|r| self.points[*r].helper
                    && (!has_bridge.contains(&r.index()) || !has_purpose.contains(&r.index())))
                .collect();
            if to_remove.is_empty() { break; }

            for r in &to_remove {
                let r = *r;
                self.for_each_constraint_collection(|_, _, coll| {
                    coll.retain_constraints(&mut |c| !c.references_point(r));
                });
            }
            for r in to_remove { self.points.remove(r); }
        }
    }

    /// Remove duplicate constraints from all collections. Prints a warning if any are found.
    /// Remove duplicate constraints, keyed by each type's
    /// [`registry::DedupKey`]. Coincidence collections share one key
    /// space, so the same endpoint pair expressed through different
    /// collections is still a duplicate. Duplicates are bugs upstream
    /// (the validation gate rejects them); removal here is the
    /// backstop, and every removal is reported.
    pub fn dedup_constraints(&mut self) {
        use crate::registry::DedupKey;
        // Pass 1: detect, with readable descriptions.
        let mut dup_msgs: std::vec::Vec<String> = std::vec::Vec::new();
        {
            let this: &Sketch = self;
            let mut coincide_seen = std::collections::HashSet::new();
            this.for_each_constraint_collection_ref(|_, meta, coll| {
                let mut local_seen = std::collections::HashSet::new();
                for i in 0..coll.len() {
                    let c = coll.item(i);
                    let fresh = match c.dedup_key() {
                        k @ DedupKey::Coincidence(..) => coincide_seen.insert(k),
                        k => local_seen.insert(k),
                    };
                    if !fresh {
                        dup_msgs.push(format!("{}: {}", meta.name, c.describe(this)));
                    }
                }
            });
        }
        if dup_msgs.is_empty() {
            return;
        }
        for m in &dup_msgs {
            eprintln!("BUG: duplicate constraint removed: {}", m);
        }
        eprintln!("{}", std::backtrace::Backtrace::force_capture());
        // Pass 2: remove -- same keys, same order, so the first copy
        // survives exactly as pass 1 reported.
        let mut coincide_seen = std::collections::HashSet::new();
        self.for_each_constraint_collection(|_, _, coll| {
            let mut local_seen = std::collections::HashSet::new();
            coll.retain_constraints(&mut |c| match c.dedup_key() {
                k @ DedupKey::Coincidence(..) => coincide_seen.insert(k),
                k => local_seen.insert(k),
            });
        });
    }

    /// Recompute tangent_la sign fields from current geometry.
    /// Needed after loading old saves that default sign to 1.0.
    pub fn fixup_tangent_signs(&mut self) {
        for t in &mut self.tangent_la {
            let l = &self.lines[t.line];
            let a = &self.arcs[t.arc];
            let dx = l.p2.value.x - l.p1.value.x;
            let dy = l.p2.value.y - l.p1.value.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-12 { continue; }
            let dist = ((a.center.value.x - l.p1.value.x) * dy
                      - (a.center.value.y - l.p1.value.y) * dx) / len;
            t.sign = if dist >= 0.0 { 1.0 } else { -1.0 };
        }
    }

    /// Merge duplicate helper points at the same position and consolidate
    /// helper-point-bridged constraints into direct constraints.
    pub fn consolidate_helper_constraints(&mut self) {
        // Phase 1: Merge duplicate helper points at the same position.
        // If two helper points are at the same position, rewrite all constraints
        // referencing the second to reference the first, then remove the second.
        let helper_refs: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| self.points[*r].helper)
            .collect();
        let mut merged = std::collections::HashMap::<Ref<Point>, Ref<Point>>::new(); // old -> canonical
        for i in 0..helper_refs.len() {
            let ri = helper_refs[i];
            if merged.contains_key(&ri) { continue; }
            let pi = self.points[ri].pos.value;
            for j in (i+1)..helper_refs.len() {
                let rj = helper_refs[j];
                if merged.contains_key(&rj) { continue; }
                let pj = self.points[rj].pos.value;
                if (pi.x - pj.x).abs() < 1e-9 && (pi.y - pj.y).abs() < 1e-9 {
                    merged.insert(rj, ri);
                    eprintln!("INFO: merging duplicate helper point {} into {}", rj.index(), ri.index());
                }
            }
        }
        if !merged.is_empty() {
            // Rewrite all point refs in constraints -- every
            // collection, via the registry (midpoint_arc_point was
            // once missing here and kept a dangling ref).
            self.for_each_constraint_collection(|_, _, coll| {
                for i in 0..coll.len() {
                    for (old, canonical) in &merged {
                        coll.item_mut(i).remap_point(*old, *canonical);
                    }
                }
            });
            // Remove merged points
            for old in merged.keys() { self.points.remove(*old); }
            // Dedup again after remapping
            self.dedup_constraints();
        }

        // Phase 2: Replace helper-point bridges with direct constraints.
        // Collect ALL matching arcs (not just the first) since point merging
        // in Phase 1 can create multiple arc constraints per helper point.
        let helper_refs: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| self.points[*r].helper)
            .collect();
        for hr in &helper_refs {
            let hr = *hr;
            let lp1s: std::vec::Vec<Ref<Line>> = self.coincident_lp1.iter().filter(|c| c.point == hr).map(|c| c.line).collect();
            let lp2s: std::vec::Vec<Ref<Line>> = self.coincident_lp2.iter().filter(|c| c.point == hr).map(|c| c.line).collect();
            let acs: std::vec::Vec<Ref<Arc>> = self.coincident_arc_center.iter().filter(|c| c.point == hr).map(|c| c.arc).collect();
            let a_starts: std::vec::Vec<Ref<Arc>> = self.coincident_arc_start.iter().filter(|c| c.point == hr).map(|c| c.arc).collect();
            let a_ends: std::vec::Vec<Ref<Arc>> = self.coincident_arc_end.iter().filter(|c| c.point == hr).map(|c| c.arc).collect();

            macro_rules! consolidate {
                ($lines:expr, $arcs:expr, $lp_coll:ident, $arc_coll:ident, $direct_coll:ident, $DirectType:ident, $label:expr) => {
                    for &line in &$lines {
                        for &arc in &$arcs {
                            self.$direct_coll.push($DirectType { line, arc, nid: 0, cid: 0, hb: CrossBlock::new() });
                            self.$lp_coll.retain(|c| !(c.line == line && c.point == hr));
                            self.$arc_coll.retain(|c| !(c.point == hr && c.arc == arc));
                            eprintln!("INFO: consolidated helper {} -> {}", hr.index(), $label);
                        }
                    }
                };
            }
            consolidate!(lp1s, acs, coincident_lp1, coincident_arc_center, coincident_lp1_arc_center, CoincidentLP1ArcCenter, "LP1ArcCenter");
            consolidate!(lp2s, acs, coincident_lp2, coincident_arc_center, coincident_lp2_arc_center, CoincidentLP2ArcCenter, "LP2ArcCenter");
            consolidate!(lp1s, a_starts, coincident_lp1, coincident_arc_start, coincident_lp1_arc_start, CoincidentLP1ArcStart, "LP1ArcStart");
            consolidate!(lp2s, a_starts, coincident_lp2, coincident_arc_start, coincident_lp2_arc_start, CoincidentLP2ArcStart, "LP2ArcStart");
            consolidate!(lp1s, a_ends, coincident_lp1, coincident_arc_end, coincident_lp1_arc_end, CoincidentLP1ArcEnd, "LP1ArcEnd");
            consolidate!(lp2s, a_ends, coincident_lp2, coincident_arc_end, coincident_lp2_arc_end, CoincidentLP2ArcEnd, "LP2ArcEnd");
        }
        self.cleanup_helper_points();
        self.dedup_constraints();
    }

}

impl arael::model::ExtendedModel<f64> for Sketch {
    fn extended_deserialize(&mut self) {
        // After solver writes back optimized values, sync radius_b.value for non-ellipses
        let refs: Vec<_> = self.arcs.refs().collect();
        for r in refs {
            if !self.arcs[r].is_ellipse {
                self.arcs[r].radius_b.value = self.arcs[r].radius.value;
            }
        }
    }

    fn extended_update(&mut self, _params: &[f64]) {
        // For non-ellipse arcs, keep radius_b work value in sync with radius
        let refs: Vec<_> = self.arcs.refs().collect();
        for r in refs {
            if !self.arcs[r].is_ellipse {
                let rv = self.arcs[r].radius.work();
                *self.arcs[r].radius_b.work_mut() = rv;
            }
        }
    }

    fn extended_cost(&self, params: &[f64]) -> f64 {
        if self.expr_constraints.is_empty() { return 0.0; }
        let bag = self.symbol_bag.as_ref().expect("symbol_bag not built");
        let vars = bag.eval_vars(params);
        let isigma = self.constraint_isigma;
        let mut total = 0.0;
        for ec in &self.expr_constraints {
            match ec.cost(&vars, isigma) {
                Ok(c) => total += c,
                Err(e) => eprintln!("expr constraint eval error: {}: {}", ec.description, e),
            }
        }
        total
    }

    fn extended_compute(&mut self, params: &[f64], grad: &mut [f64]) {
        if self.expr_constraints.is_empty() { return; }
        let bag = self.symbol_bag.as_ref().expect("symbol_bag not built");
        let vars = bag.eval_vars(params);
        let isigma = self.constraint_isigma;
        let hb = &mut self.expr_hb;
        for ec in &self.expr_constraints {
            if let Err(e) = ec.compute(&vars, isigma, hb, grad) {
                eprintln!("expr constraint eval error: {}: {}", ec.description, e);
            }
        }
    }

    fn extended_jacobian(&mut self, params: &[f64], rows: &mut std::vec::Vec<arael::model::JacobianRow<f64>>, cid: &mut u32) {
        if self.expr_constraints.is_empty() { return; }
        let bag = self.symbol_bag.as_ref().expect("symbol_bag not built");
        let vars = bag.eval_vars(params);
        let isigma = self.constraint_isigma;
        for ec in &mut self.expr_constraints {
            ec.cid = *cid;
            match ec.jacobian_row(&vars, isigma) {
                Ok((residual, entries)) => {
                    rows.push(arael::model::JacobianRow { constraint: *cid, label: ec.label, residual, entries });
                }
                Err(e) => eprintln!("expr constraint eval error: {}: {}", ec.description, e),
            }
            *cid += 1;
        }
    }
}

impl Sketch {
    /// Return `(nid, cid)` pairs for every constraint in the sketch
    /// (across all constraint collections, entity constraints excluded).
    ///
    /// `nid` is the user-visible stable numeric constraint id (C1, C2,
    /// ...) and is serialised, so pre/post action comparisons are
    /// robust against the macro's per-pass cid renumbering. `cid` is
    /// the post-calc_jacobian identifier used by [`arael::model::Jacobian::rows`].
    ///
    /// Must be called AFTER `calc_jacobian`/`solve`/`compute_dof` so
    /// the `cid` field is populated on every constraint instance.
    /// Constraints whose `nid` is still 0 (never named) are omitted.
    /// Remove the constraint carrying this nid, from whichever
    /// collection holds it. Returns whether one was removed. Nids are
    /// unique and permanent, so this is the stable delete -- positional
    /// indices go stale with every retain.
    pub fn remove_constraint_by_nid(&mut self, nid: u32) -> bool {
        if nid == 0 {
            return false;
        }
        let mut removed = false;
        self.for_each_constraint_collection(|_, _, coll| {
            let before = coll.len();
            coll.retain_constraints(&mut |c| c.nid() != nid);
            removed |= coll.len() != before;
        });
        removed
    }

    /// Whether any collection holds a constraint with this nid.
    pub fn has_constraint_nid(&self, nid: u32) -> bool {
        if nid == 0 {
            return false;
        }
        let mut found = false;
        self.for_each_constraint_collection_ref(|_, _, coll| {
            if found {
                return;
            }
            for i in 0..coll.len() {
                if coll.item(i).nid() == nid {
                    found = true;
                    return;
                }
            }
        });
        found
    }

    pub fn constraint_nid_cid_pairs(&self) -> std::vec::Vec<(u32, u32)> {
        let mut out = std::vec::Vec::new();
        self.for_each_constraint_collection_ref(|_, _, coll| {
            for i in 0..coll.len() {
                let c = coll.item(i);
                if c.nid() != 0 {
                    out.push((c.nid(), c.cid()));
                }
            }
        });
        out
    }

    /// Build a map from CID to human-readable constraint description.
    /// Must be called AFTER calc_jacobian/solve/compute_dof, which populates
    /// the `cid` field on each constraint instance.
    pub fn constraint_labels(&self) -> std::collections::HashMap<u32, String> {
        let mut m = std::collections::HashMap::new();
        for r in self.points.refs() { let p = &self.points[r]; m.insert(p.cid, format!("point:{}", p.name)); }
        for r in self.lines.refs() { let l = &self.lines[r]; m.insert(l.cid, format!("line:{}", l.name)); }
        for r in self.arcs.refs() { let a = &self.arcs[r]; m.insert(a.cid, format!("arc:{}", a.name)); }
        self.for_each_constraint_collection_ref(|arenas, meta, coll| {
            for i in 0..coll.len() {
                let c = coll.item(i);
                let mut names: std::vec::Vec<String> = std::vec::Vec::new();
                c.each_point_ref(&mut |p| names.push(
                    arenas.points.get(p).map(|e| e.name.clone()).unwrap_or_else(|| "?".into())));
                c.each_line_ref(&mut |l| names.push(
                    arenas.lines.get(l).map(|e| e.name.clone()).unwrap_or_else(|| "?".into())));
                c.each_arc_ref(&mut |a| names.push(
                    arenas.arcs.get(a).map(|e| e.name.clone()).unwrap_or_else(|| "?".into())));
                m.insert(c.cid(), format!("{}:{}", meta.name, names.join(",")));
            }
        });
        m
    }

    /// Fix up arc fields after loading old files that lack radius_b/rotation.
    /// Detects the sentinel default (radius_b == 0.0) and sets radius_b to
    /// match radius, rotation to 0, as optimizable params.
    pub fn fixup_after_load(&mut self) {
        let refs: Vec<_> = self.arcs.refs().collect();
        for r in refs {
            if self.arcs[r].radius_b.value == 0.0 && !self.arcs[r].is_ellipse {
                let rv = self.arcs[r].radius.value;
                self.arcs[r].radius_b = Param::new(rv);
                self.arcs[r].rotation = Param::fixed(0.0);
            }
        }
    }

    /// Add an expression constraint. The expression should evaluate to 0
    /// when satisfied. Symbol resolution and differentiation happen at
    /// solve() time.
    pub fn add_expr_constraint(&mut self, expr: arael_sym::E, description: String) {
        self.expr_constraints.push(ExpressionConstraint::new_unresolved(expr, description));
    }

    /// Rebuild expr_constraints from dimensions that have expr_str.
    /// Called at the start of every solve() since the set of optimizable
    /// params can change between solves (lock/unlock).
    /// Entity names reachable from the sketch's expressions: prefixes
    /// of dotted symbols from user-param expressions, dimension
    /// expressions, range bounds and measured symbols, closed over
    /// dim / user-param references. A symbol bag built for these
    /// entities resolves and evaluates those expressions exactly like
    /// the full bag, without paying for every entity in the sketch.
    fn expr_entity_filter(&self) -> std::collections::HashSet<String> {
        let mut syms: std::collections::HashSet<String> = std::collections::HashSet::new();
        for p in &self.user_params {
            if p.expr_str.trim().parse::<f64>().is_ok() { continue; }
            if let Ok(parsed) = arael_sym::parse(&p.expr_str) {
                syms.extend(parsed.symbols());
            }
        }
        for dim in &self.dimensions {
            let mut uses_measured = dim.derived || dim.range.is_some();
            if let Some(ref es) = dim.expr_str {
                uses_measured = true;
                if let Ok(parsed) = arael_sym::parse(es) {
                    syms.extend(parsed.symbols());
                }
            }
            if let Some(rb) = &dim.range {
                let mut add = |syms: &mut std::collections::HashSet<String>, rv: &dimensions::RangeValue| {
                    if let dimensions::RangeValue::Live(src) = rv
                        && let Ok(parsed) = arael_sym::parse(src) {
                            syms.extend(parsed.symbols());
                        }
                };
                match rb {
                    dimensions::RangeBound::Min(v) | dimensions::RangeBound::Max(v) => add(&mut syms, v),
                    dimensions::RangeBound::Between(lo, hi) => {
                        add(&mut syms, lo);
                        add(&mut syms, hi);
                    }
                }
            }
            if uses_measured {
                syms.extend(dim.measured_symbol(self).symbols());
            }
        }
        for _ in 0..16 {
            let mut added = false;
            for dim in &self.dimensions {
                if !syms.contains(&dim.name) { continue; }
                if let Some(ref es) = dim.expr_str {
                    if let Ok(parsed) = arael_sym::parse(es) {
                        for sym in parsed.symbols() { added |= syms.insert(sym); }
                    }
                } else if dim.derived {
                    for sym in dim.measured_symbol(self).symbols() { added |= syms.insert(sym); }
                }
            }
            for p in &self.user_params {
                if !syms.contains(&p.name) || p.expr_str.trim().parse::<f64>().is_ok() { continue; }
                if let Ok(parsed) = arael_sym::parse(&p.expr_str) {
                    for sym in parsed.symbols() { added |= syms.insert(sym); }
                }
            }
            if !added { break; }
        }
        syms.iter()
            .filter_map(|sym| {
                let prefix = sym.split('.').next()?;
                (prefix.len() < sym.len()).then(|| prefix.to_string())
            })
            .collect()
    }

    fn rebuild_expr_constraints(&mut self) {
        self.expr_constraints.clear();
        let has_expr = self.dimensions.iter().any(|d| d.expr_str.is_some());
        let has_user_params = !self.user_params.is_empty();
        let has_range = self.dimensions.iter().any(|d| d.range.is_some());
        if !has_expr && !has_user_params && !has_range {
            for d in &mut self.dimensions { d.broken = false; }
            return;
        }

        // Reset broken flags so SymbolBag always starts fresh --
        // stale flags from a previous solve can hide circular refs.
        for d in &mut self.dimensions { d.broken = false; }
        for p in &mut self.user_params { p.broken = false; }

        // Need param indices assigned for SymbolBag; serialize to assign them.
        {
            let mut tmp = std::vec::Vec::new();
            self.serialize(&mut tmp);
        }
        let filter = self.expr_entity_filter();
        let mut bag = SymbolBag::build_filtered(self, Some(&filter));

        // Detect broken user params first (they feed into dimensions).
        // Process in order so earlier params that break get frozen before
        // downstream params/dims are checked.
        for i in 0..self.user_params.len() {
            let expr_str = &self.user_params[i].expr_str;
            // Pure numeric literals are never broken
            if expr_str.trim().parse::<f64>().is_ok() { continue; }
            let is_broken = if let Ok(parsed) = arael_sym::parse(expr_str) {
                let expanded = expr_constraint::expand_derived(&parsed, &bag);
                !expanded.symbols().iter().all(|sym|
                    bag.param_indices.contains_key(sym.as_str())
                    || bag.dim_values.contains_key(sym.as_str())
                )
            } else {
                true
            };
            self.user_params[i].broken = is_broken;
            if is_broken {
                bag.derived.remove(&self.user_params[i].name);
                bag.dim_values.insert(
                    self.user_params[i].name.clone(),
                    self.user_params[i].value,
                );
            }
        }

        // Detect broken references and create expression constraints.
        // Process in order so broken dims get frozen in the bag before
        // downstream dims that reference them are checked.
        for i in 0..self.dimensions.len() {
            // Derived dimensions don't create constraints
            if self.dimensions[i].derived { continue; }
            if let Some(ref expr_str) = self.dimensions[i].expr_str {
                let is_broken = if let Ok(parsed) = arael_sym::parse(expr_str) {
                    let expanded = expr_constraint::expand_derived(&parsed, &bag);
                    let all_resolved = expanded.symbols().iter().all(|sym|
                        bag.param_indices.contains_key(sym.as_str())
                        || bag.dim_values.contains_key(sym.as_str())
                    );
                    if all_resolved {
                        // Normal: measured - expr = 0
                        let measured = self.dimensions[i].measured_symbol(self);
                        let residual = measured - parsed;
                        let desc = format!("{} = {}", self.dimensions[i].name, expr_str);
                        self.expr_constraints.push(
                            ExpressionConstraint::new_unresolved(residual, desc));
                        false
                    } else {
                        true
                    }
                } else {
                    true
                };

                self.dimensions[i].broken = is_broken;
                if is_broken {
                    // Freeze in bag so downstream dims see a constant
                    bag.derived.remove(&self.dimensions[i].name);
                    bag.dim_values.insert(
                        self.dimensions[i].name.clone(),
                        self.dimensions[i].value,
                    );
                    // Fallback: constrain to last computed value
                    let measured = self.dimensions[i].measured_symbol(self);
                    let residual = measured - arael_sym::constant(self.dimensions[i].value);
                    let desc = format!("{} = {} [broken]", self.dimensions[i].name, self.dimensions[i].value);
                    self.expr_constraints.push(
                        ExpressionConstraint::new_unresolved(residual, desc));
                }
            } else {
                self.dimensions[i].broken = false;
            }
        }

        // Range dimensions: barrier residuals (piecewise-zero inside the
        // feasible region). Each contributes one direct ExpressionConstraint
        // whose `expr` IS the residual (not measured - expr, as in the
        // equality path above). Live bounds re-parse + resolve against the
        // current SymbolBag; a bound with unresolved free symbols marks
        // the dimension broken (same cascade as expression dims above).
        for i in 0..self.dimensions.len() {
            let rb = match self.dimensions[i].range.clone() {
                Some(rb) => rb,
                None => continue,
            };
            let resolve_value = |rv: &dimensions::RangeValue, bag: &SymbolBag|
                -> Option<arael_sym::E>
            {
                match rv {
                    dimensions::RangeValue::Literal(v) => Some(arael_sym::constant(*v)),
                    dimensions::RangeValue::Live(src) => {
                        let parsed = arael_sym::parse(src).ok()?;
                        let expanded = expr_constraint::expand_derived(&parsed, bag);
                        let all_resolved = expanded.symbols().iter().all(|sym|
                            bag.param_indices.contains_key(sym.as_str())
                            || bag.dim_values.contains_key(sym.as_str()));
                        if all_resolved { Some(expanded) } else { None }
                    }
                }
            };
            let resolved = match &rb {
                dimensions::RangeBound::Min(v) =>
                    resolve_value(v, &bag).map(dimensions::ResolvedBound::Min),
                dimensions::RangeBound::Max(v) =>
                    resolve_value(v, &bag).map(dimensions::ResolvedBound::Max),
                dimensions::RangeBound::Between(lo, hi) => {
                    match (resolve_value(lo, &bag), resolve_value(hi, &bag)) {
                        (Some(l), Some(h)) => Some(dimensions::ResolvedBound::Between(l, h)),
                        _ => None,
                    }
                }
            };
            let Some(resolved) = resolved else {
                self.dimensions[i].broken = true;
                continue;
            };
            self.dimensions[i].broken = false;
            let measured = self.dimensions[i].measured_symbol(self);
            let residual = Dimension::range_residual(&resolved, measured);
            let desc = format!("{} {}", self.dimensions[i].name,
                match &rb {
                    dimensions::RangeBound::Min(v) => format!(">= {}", v),
                    dimensions::RangeBound::Max(v) => format!("<= {}", v),
                    dimensions::RangeBound::Between(lo, hi) => format!("in {} to {}", lo, hi),
                });
            let mut ec = ExpressionConstraint::new_unresolved(residual, desc);
            // Range dimensions are one-sided barriers: the residual is
            // zero inside the feasible band and linear outside it. Tag
            // the label so DOF rank detection can strip them -- their
            // Jacobian row would otherwise flip in/out of rank as the
            // geometry crosses the bound, reporting DOF 2 at the bound
            // and DOF 3 off it for the same sketch.
            ec.label = "range";
            self.expr_constraints.push(ec);
        }
    }

    /// Validate an expression string: parse it and check all symbols resolve.
    /// Returns Err with a description if invalid.
    pub fn validate_expr(&mut self, expr_str: &str) -> Result<(), String> {
        let parsed = arael_sym::parse(expr_str).map_err(|e| e.to_string())?;
        {
            let mut tmp = std::vec::Vec::new();
            self.serialize(&mut tmp);
        }
        let bag = SymbolBag::build(self);
        let expanded = expr_constraint::expand_derived(&parsed, &bag);
        let unresolved: Vec<String> = expanded.symbols().into_iter().filter(|sym|
            !bag.param_indices.contains_key(sym.as_str())
            && !bag.dim_values.contains_key(sym.as_str())
        ).collect();
        if !unresolved.is_empty() {
            return Err(format!("Unknown symbol: {}", unresolved.join(", ")));
        }
        Ok(())
    }

    /// Validate a user parameter name. Returns Err if the name is empty,
    /// a duplicate, a system name pattern, or already used by an entity.
    pub fn validate_param_name(&self, name: &str, exclude_index: Option<usize>) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("Name cannot be empty".into());
        }
        // Must be a valid identifier: alphanumeric + underscore, not starting with digit
        if name.bytes().next().is_none_or(|b| b.is_ascii_digit()) {
            return Err("Name cannot start with a digit".into());
        }
        if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err("Name can only contain letters, digits, and underscores".into());
        }
        // Not a system name pattern (d0, L0, P0, A0, etc.)
        if is_system_name(name) {
            return Err(format!("'{}' conflicts with system naming (d/L/P/A + number)", name));
        }
        // Not a duplicate of another user param
        for (i, p) in self.user_params.iter().enumerate() {
            if Some(i) == exclude_index { continue; }
            if p.name == name {
                return Err(format!("Parameter '{}' already exists", name));
            }
        }
        // Not a dimension name
        for d in &self.dimensions {
            if d.name == name {
                return Err(format!("'{}' is already used by a dimension", name));
            }
        }
        Ok(())
    }

    /// Get a display-friendly name for a point. For helper points, resolves
    /// through bridge constraints to show the semantic origin (e.g. "L0.p1").
    pub fn point_display_name(&self, r: Ref<Point>) -> String {
        let p = &self.points[r];
        if !p.helper { return p.name.clone(); }
        for c in &self.coincident_lp1 {
            if c.point == r { return format!("{}.p1", self.lines[c.line].name); }
        }
        for c in &self.coincident_lp2 {
            if c.point == r { return format!("{}.p2", self.lines[c.line].name); }
        }
        for c in &self.coincident_arc_center {
            if c.point == r { return format!("{}.center", self.arcs[c.arc].name); }
        }
        for c in &self.coincident_arc_start {
            if c.point == r { return format!("{}.start", self.arcs[c.arc].name); }
        }
        for c in &self.coincident_arc_end {
            if c.point == r { return format!("{}.end", self.arcs[c.arc].name); }
        }
        p.name.clone()
    }

    /// Add an expression-based dimension. The expression string is parsed
    /// Build a `cid -> dimension_name` map for constraints that are
    /// backed by a dimension (`d<n>`). Deleting the dimension via
    /// `delete d<n>` removes both the dimension and its backing
    /// constraint, which is the user-facing way to edit or remove
    /// these constraints -- `C<nid>` is visible in `list` but is not
    /// a valid handle for `delete` (see `find_constraint_by_name` in arael-sketch-backend).
    ///
    /// Coverage focuses on the dimension kinds that write into
    /// constraint collections: `HDistance`, `VDistance`,
    /// `PointPointDistance`, `PointLineDistance`, `Angle`,
    /// `ConcentricDistance`, `LineLineDistance`. Dimensions that only
    /// feed expression constraints (`LineLength`, `ArcRadius`,
    /// `ArcRadiusB`, `ArcSweep`, `LineAngle`) don't have a backing
    /// cid in any collection and are omitted.
    pub fn dimension_cid_name_map(&self) -> std::collections::HashMap<u32, String> {
        use crate::dimensions::{DimensionEndpoint as E, DimensionKind as K};
        let mut out = std::collections::HashMap::new();
        for dim in &self.dimensions {
            // Each arm scans the collection(s) that the corresponding
            // add path pushes into. See `push_axis_distance` and
            // `add_dimension` in arael-sketch/src/actions.rs for the
            // inverse mapping.
            let cid = match &dim.kind {
                K::HDistance(a, b) => find_axis_cid(self, a, b, true),
                K::VDistance(a, b) => find_axis_cid(self, a, b, false),
                K::PointPointDistance(a, b) => find_distance_cid(self, a, b),
                K::PointLineDistance(ep, l) => find_point_line_cid(self, ep, *l),
                K::Angle(la, lb, _) => self.angle.iter()
                    .find(|c| c.a == *la && c.b == *lb)
                    .map(|c| c.cid),
                K::ConcentricDistance(a, b) => self.distance_concentric.iter()
                    .find(|c| (c.a == *a && c.b == *b) || (c.a == *b && c.b == *a))
                    .map(|c| c.cid),
                K::LineLineDistance(a, b) => {
                    // LineLineDistance uses point-to-line with LineP1(b) as anchor.
                    find_point_line_cid(self, &E::LineP1(*b), *a)
                }
                K::LineLength(_) | K::ArcRadius(_) | K::ArcRadiusB(_)
                    | K::ArcSweep(_) | K::LineAngle(_) | K::ArcRotation(_) => None,
            };
            if let Some(cid) = cid {
                out.insert(cid, dim.name.clone());
            }
        }
        // Expression-backed dimensions (LineLength, ArcRadius,
        // ArcRadiusB, ArcSweep, LineAngle, plus any dimension with an
        // expr_str or range bound) feed into self.expr_constraints.
        // Their descriptions always start with the dimension name
        // followed by a space -- parse that prefix to map cid -> name.
        for ec in &self.expr_constraints {
            if let Some((dim_name, _)) = ec.description.split_once(' ') {
                let is_dim = dim_name.starts_with('d')
                    && dim_name.len() > 1
                    && dim_name[1..].chars().all(|c| c.is_ascii_digit());
                if is_dim {
                    out.insert(ec.cid, dim_name.to_string());
                }
            }
        }
        out
    }

    /// Look up a constraint by its user-visible name (e.g. `"C1"`,
    /// `"CL0H"`) and return the descriptive part of its
    /// `list_constraints` entry. Returns `None` if no active
    /// constraint bears that name.
    ///
    /// The search walks `list_constraints` and strips the `"<name>: "`
    /// prefix, so every constraint that appears in the canonical list
    /// output is findable -- including dimension-managed distance
    /// constraints that have a `C<nid>` name but no `ConstraintId`
    /// variant.
    pub fn find_constraint_description(&self, name: &str) -> Option<String> {
        let prefix = format!("{}: ", name);
        for line in self.list_constraints() {
            if let Some(rest) = line.strip_prefix(&prefix) {
                return Some(rest.to_string());
            }
        }
        None
    }

    /// List all active constraints as human-readable strings.
    /// Constraints addressable by their own name: H/V flags, locks and
    /// the geometric `C<n>` collections. Dimension-managed constraints
    /// (the value flags length / xangle / radius / radius_b / sweep /
    /// rotation and the dimension-backed collections) are the
    /// dimension's business and listed with it -- except an orphan
    /// value flag with no dimension over it, which still shows here
    /// so it is not invisible.
    pub fn list_constraints(&self) -> Vec<String> {
        let mut out = Vec::new();
        let managed = |kind: DimensionKind| self.dimensions.iter().any(|d| d.kind == kind);
        // Entity-level flags
        for r in self.lines.refs() {
            let l = &self.lines[r];
            if l.constraints.horizontal { out.push(format!("{}: horizontal {}", format_flag_name(&l.name, 'H'), l.name)); }
            if l.constraints.vertical { out.push(format!("{}: vertical {}", format_flag_name(&l.name, 'V'), l.name)); }
            if l.constraints.has_length && !managed(DimensionKind::LineLength(r)) {
                out.push(format!("length {} = {}", l.name, l.constraints.length));
            }
            if l.constraints.has_angle && !managed(DimensionKind::LineAngle(r)) {
                out.push(format!("xangle {} = {:.4}", l.name, l.constraints.target_angle.to_degrees()));
            }
            if !l.p1.optimize { out.push(format!("lock {}.p1", l.name)); }
            if !l.p2.optimize { out.push(format!("lock {}.p2", l.name)); }
        }
        for r in self.points.refs() {
            let p = &self.points[r];
            if p.constraints.has_fix_x || p.constraints.has_fix_y {
                out.push(format!("lock {}", p.name));
            }
        }
        for r in self.arcs.refs() {
            let a = &self.arcs[r];
            if a.constraints.has_target_radius && !managed(DimensionKind::ArcRadius(r)) {
                out.push(format!("radius {} = {}", a.name, a.constraints.target_radius));
            }
            if a.constraints.has_target_radius_b && !managed(DimensionKind::ArcRadiusB(r)) {
                out.push(format!("radius_b {} = {}", a.name, a.constraints.target_radius_b));
            }
            if a.constraints.has_target_sweep && !managed(DimensionKind::ArcSweep(r)) {
                out.push(format!("sweep {} = {:.2} deg", a.name, a.constraints.target_sweep.to_degrees()));
            }
            if a.constraints.has_target_rotation && !managed(DimensionKind::ArcRotation(r)) {
                out.push(format!("xangle {} = {:.4}", a.name, a.constraints.target_rotation.to_degrees()));
            }
            if !a.center.optimize { out.push(format!("lock {}.center", a.name)); }
        }
        // Constraint collections, via the registry: one describe()
        // per type, one emission per constraint (the old hand walk
        // listed the midpoint_lp/arc families twice, the second copy
        // with swapped operands). Coincidence entries referencing a
        // helper point are bridges -- internal wiring, suppressed.
        // Dimension-backed collections belong to their dimension.
        // Sorted by nid: creation order, not collection order.
        let mut items: std::vec::Vec<(u32, String)> = std::vec::Vec::new();
        self.for_each_constraint_collection_ref(|arenas, meta, coll| {
            if meta.dimension_backed {
                return;
            }
            for i in 0..coll.len() {
                let c = coll.item(i);
                if meta.coincidence {
                    let mut bridges_helper = false;
                    c.each_point_ref(&mut |r| {
                        if arenas.points.get(r).is_some_and(|p| p.helper) {
                            bridges_helper = true;
                        }
                    });
                    if bridges_helper {
                        continue;
                    }
                }
                items.push((c.nid(), format!("C{}: {}", c.nid(), c.describe(self))));
            }
        });
        items.sort_by_key(|&(nid, _)| nid);
        out.extend(items.into_iter().map(|(_, s)| s));
        out
    }

    /// and the constraint is: `measured_property - parsed_expr = 0`.
    /// Returns Err if the expression fails to parse or references unknown symbols.
    pub fn add_expr_dimension(&mut self, kind: DimensionKind, expr_str: &str,
                              offset: vect2d, text_along: f64) -> Result<(), String> {
        self.validate_expr(expr_str)?;
        let parsed = arael_sym::parse(expr_str).unwrap(); // safe: validate_expr checked parse

        let name = format!("d{}", self.next_dimension_id);
        self.next_dimension_id += 1;

        // Build the measured property expression
        let dim = Dimension {
            did: 0, // minted by assign_dimension_ids
            kind, value: 0.0, offset, text_along,
            name: name.clone(), expr_str: Some(expr_str.to_string()),
            broken: false,
            derived: false,
            range: None,
        };
        let measured = dim.measured_symbol(self);
        self.dimensions.push(dim);

        // Residual: measured - expr = 0
        let residual = measured - parsed;
        self.add_expr_constraint(residual, format!("{} = {}", name, expr_str));
        Ok(())
    }

    /// Rebuild expression constraints and resolve them so they contribute
    /// to Jacobian / cost assembly. Needed before calling calc_jacobian or
    /// calc_grad_hessian_dense directly (outside of solve()), e.g. for DOF.
    pub fn prepare_expr_constraints(&mut self) {
        self.rebuild_expr_constraints();
        if !self.expr_constraints.is_empty() {
            let mut tmp = Vec::new();
            self.serialize(&mut tmp);
            let bag = SymbolBag::build(self);
            for ec in &mut self.expr_constraints {
                ec.resolve(&bag);
            }
            self.symbol_bag = Some(bag);
        }
    }


    /// Format an informative error message when Hessian decomposition fails.
    fn hessian_error_msg(n: usize, hessian: &[f64], err: &str) -> String {
        let nan_count = hessian.iter().filter(|v| v.is_nan()).count();
        let inf_count = hessian.iter().filter(|v| v.is_infinite()).count();
        format!("DOF computation failed: {} ({}x{} Hessian, {} NaN, {} Inf). \
                 This likely indicates a solver bug -- please report it.",
                err, n, n, nan_count, inf_count)
    }

    /// Legacy DOF computation via eigendecomposition of the Hessian J^T J.
    ///
    /// Kept for comparison and for the `dof eigenvalues` diagnostic. Rank
    /// detection from Hessian eigenvalues is numerically unstable at high
    /// constraint scales because forming J^T J squares the condition
    /// number. For routine DOF counting, prefer [`Sketch::compute_dof`]
    /// which uses SVD of the Jacobian directly.
    ///
    /// Uses nalgebra for n<32, faer for n>=32. Benchmark at n=896 (polygon128):
    ///   faer eigenvalues-only:    45ms
    ///   faer full eigen:          95ms
    ///   nalgebra eigenvalues-only: 110ms
    ///   nalgebra full eigen:      220ms
    pub fn compute_dof_eigenvalues(&mut self, analyze: bool) -> Result<DofResult, String> {
        self.compute_dof_eigenvalues_opt(analyze, true)
    }

    /// Eigenvalue-based DOF analysis with explicit preconditioning flag.
    ///
    /// When `preconditioned` is true (the default via
    /// [`Self::compute_dof_eigenvalues`]), the Hessian is scaled as
    /// `H_N = D^{-1} H D^{-1}` where `D = diag(sqrt(diag(H)))`. This is
    /// symmetric Jacobi preconditioning -- it preserves the null-space
    /// (and hence rank) exactly, but tames the condition number by
    /// folding per-parameter scale differences into the diagonal.
    /// `J^T J` would otherwise square an already-wide Jacobian
    /// condition number, making eigenvalue-based rank detection fail
    /// at large sketch scales. Right eigenvectors are back-transformed
    /// to raw parameter space as `v_raw = D^{-1} v_N` (then renormed
    /// to unit length) so the reported directions remain physical.
    ///
    /// When `preconditioned` is false, the eigenvalues and eigenvectors
    /// are of the raw Hessian -- useful for debugging residual scaling
    /// choices.
    pub fn compute_dof_eigenvalues_opt(&mut self, analyze: bool, preconditioned: bool) -> Result<DofResult, String> {
        // Strip range barriers before the Hessian is assembled -- their
        // contribution is state-dependent (zero inside the feasible
        // band, non-zero outside) and would make the reported DOF
        // swing as geometry crosses a bound. See the matching comment
        // in compute_dof. Restored after computation regardless of
        // outcome.
        self.prepare_expr_constraints();
        let saved_ranges: Vec<_> = {
            let (kept, ranges): (Vec<_>, Vec<_>) = std::mem::take(&mut self.expr_constraints)
                .into_iter().partition(|ec| ec.label != "range");
            self.expr_constraints = kept;
            ranges
        };
        let result = self.compute_dof_eigenvalues_opt_inner(analyze, preconditioned);
        self.expr_constraints.extend(saved_ranges);
        result
    }

    fn compute_dof_eigenvalues_opt_inner(&mut self, analyze: bool, preconditioned: bool) -> Result<DofResult, String> {
        use arael::simple_lm::LmProblem;
        let mut timer = Timer::new();

        self.update_tangent_flags();
        self.update_perpendicular_flags();
        self.update_line_dir_flags();

        let saved_drift = self.drift_isigma;
        self.drift_isigma = 0.0;

        let mut params = Vec::new();
        self.serialize(&mut params);
        let n = params.len();
        if n == 0 {
            self.drift_isigma = saved_drift;
            return Ok(DofResult { dof: 0, param_names: Vec::new(), eigenvalues: Vec::new(), eigenvectors: Vec::new() });
        }

        let param_names = if analyze {
            let bag = SymbolBag::build(self);
            let mut names = vec![String::new(); n];
            for (name, &idx) in &bag.param_indices {
                let i = idx as usize;
                if i < n && names[i].is_empty() { names[i] = name.clone(); }
            }
            names
        } else {
            Vec::new()
        };

        let t_prep = timer.lap();
        let mut grad = vec![0.0f64; n];
        let mut hessian = vec![0.0f64; n * n];
        self.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);
        self.drift_isigma = saved_drift;
        let t_hessian = timer.lap();
        // Degenerate geometry (a /len residual on collapsed entities)
        // yields NaN; the spectral sorts below must never see it.
        if hessian.iter().any(|v| !v.is_finite()) {
            return Err("Hessian contains non-finite values (degenerate geometry)".into());
        }

        // Symmetric Jacobi preconditioning: scale by `sqrt(diag(H))`
        // which equals the Jacobian's column L2 norms. Preserves
        // null-space exactly (zero eigenvalues stay zero) but narrows
        // the non-zero spectrum, so rank detection becomes robust to
        // per-parameter scale mismatches. Skipped in `raw` mode.
        let d: Vec<f64> = if preconditioned {
            (0..n).map(|i| hessian[i * n + i].max(0.0).sqrt().max(1e-15)).collect()
        } else {
            vec![1.0; n]
        };
        if preconditioned {
            for i in 0..n {
                for k in 0..n {
                    hessian[i * n + k] /= d[i] * d[k];
                }
            }
        }

        // Determine rank via spectral gap in the lower portion of the spectrum.
        let rank_from_evs = |evs: &[f64]| -> usize {
            let mut sorted: Vec<f64> = evs.iter().map(|v| v.abs()).collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let max_ev = sorted.last().copied().unwrap_or(0.0);
            let upper_bound = max_ev * 0.01;
            // Same zero-floor trick as rank_from_svs: let near-zero
            // eigenvalues participate in the gap search instead of
            // being silently skipped when `lo < 1e-20`.
            let floor = max_ev * 1e-20;
            let mut best_gap = 0.0f64;
            let mut best_cut = 0;
            for i in 0..sorted.len().saturating_sub(1) {
                let lo = sorted[i].max(floor);
                let hi = sorted[i + 1].max(floor);
                if lo > upper_bound { break; }
                let gap = hi / lo;
                if gap > best_gap {
                    best_gap = gap;
                    best_cut = i + 1;
                }
            }
            if best_gap < 1e3 {
                best_cut = sorted.iter().filter(|&&v| v < 1e-15).count();
            }

            evs.len() - best_cut
        };
        // Eigenvectors come out in the normalised parameter space.
        // Back-transform via `v_raw = D^{-1} v_N` (then renormalise to
        // unit length) so the reported direction is a physical one
        // the user can act on. When preconditioning is off, D is all
        // ones and this is a no-op.
        let unscale_evecs = |mut evs: Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            if !preconditioned { return evs; }
            for v in evs.iter_mut() {
                for (i, vi) in v.iter_mut().enumerate() {
                    *vi /= d[i];
                }
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-20 {
                    for vi in v.iter_mut() { *vi /= norm; }
                }
            }
            evs
        };

        let (method, result) = if n < 32 && analyze {
            let h = nalgebra::DMatrix::from_row_slice(n, n, &hessian);
            let eigen = nalgebra::SymmetricEigen::new(h);
            let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
            let rank = rank_from_evs(&eigenvalues);
            let dof = n.saturating_sub(rank);
            let eigenvectors: Vec<Vec<f64>> = unscale_evecs((0..n)
                .map(|col| eigen.eigenvectors.column(col).iter().cloned().collect())
                .collect());
            ("nalgebra eigen", DofResult { dof, param_names, eigenvalues, eigenvectors })
        } else if n < 32 {
            let h = nalgebra::DMatrix::from_row_slice(n, n, &hessian);
            let evs = h.symmetric_eigenvalues();
            let evs_vec: Vec<f64> = evs.iter().cloned().collect();
            let rank = rank_from_evs(&evs_vec);
            let dof = n.saturating_sub(rank);
            ("nalgebra eigenvalues-only", DofResult { dof, param_names: Vec::new(), eigenvalues: Vec::new(), eigenvectors: Vec::new() })
        } else if analyze {
            let faer_h = faer::Mat::from_fn(n, n, |i, k| hessian[i * n + k]);
            match faer_h.self_adjoint_eigen(faer::Side::Lower) {
                Ok(eigen) => {
                    let s = eigen.S().column_vector();
                    let u = eigen.U();
                    let eigenvalues: Vec<f64> = (0..n).map(|i| s[i]).collect();
                    let rank = rank_from_evs(&eigenvalues);
                    let dof = n.saturating_sub(rank);
                    let eigenvectors: Vec<Vec<f64>> = unscale_evecs((0..n)
                        .map(|col| (0..n).map(|row| u[(row, col)]).collect())
                        .collect());
                    ("faer eigen", DofResult { dof, param_names, eigenvalues, eigenvectors })
                }
                Err(e) => return Err(Self::hessian_error_msg(n, &hessian, &format!("{:?}", e))),
            }
        } else {
            let faer_h = faer::Mat::from_fn(n, n, |i, k| hessian[i * n + k]);
            match faer_h.self_adjoint_eigenvalues(faer::Side::Lower) {
                Ok(evs) => {
                    let evs_vec: Vec<f64> = evs.to_vec();
                    let rank = rank_from_evs(&evs_vec);
                    let dof = n.saturating_sub(rank);
                    ("faer eigenvalues-only", DofResult { dof, param_names: Vec::new(), eigenvalues: Vec::new(), eigenvectors: Vec::new() })
                }
                Err(e) => return Err(Self::hessian_error_msg(n, &hessian, &format!("{:?}", e))),
            }
        };
        if timer.on() {
            let t_eigen = timer.lap();
            eprintln!("[DOF-EIG] n={} analyze={} method={} prep={:?} hessian={:?} eigen={:?} total={:?} dof={}",
                n, analyze, method, t_prep, t_hessian, t_eigen, timer.total(), result.dof);
        }
        Ok(result)
    }

    /// Compute degrees of freedom. When `analyze` is true, also returns
    /// parameter names, eigenvalues, and eigenvectors for free direction
    /// classification. When false, only the DOF count is computed (fast path).
    ///
    /// Rank is determined from the singular values of the Jacobian J.
    /// The returned `eigenvalues` field holds sigma_i^2 (mathematically
    /// equivalent to eigenvalues of J^T J) and `eigenvectors` holds the
    /// right singular vectors of J (columns of V, equivalent to
    /// eigenvectors of J^T J). SVD is used because forming J^T J squares
    /// the condition number and destroys rank-detection precision at
    /// high constraint scales (observed: scale=10000 misreports DOF).
    ///
    /// Build the Jacobian the DOF machinery ranks: derived state
    /// prepared, drift disabled during assembly, range barrier rows
    /// stripped.
    fn dof_jacobian(&mut self) -> (arael::model::Jacobian<f64>, usize) {
        self.prepare_expr_constraints();
        self.update_tangent_flags();
        self.update_perpendicular_flags();
        self.update_line_dir_flags();

        let saved_drift = self.drift_isigma;
        self.drift_isigma = 0.0;

        let mut params = Vec::new();
        self.serialize(&mut params);
        let n = params.len();
        let mut jacobian = self.calc_jacobian(&params);
        self.drift_isigma = saved_drift;
        // Strip one-sided barrier rows (range dimensions, tagged
        // label = "range") before rank detection. Their Jacobian row
        // is zero inside the feasible band and non-zero outside, so
        // including them would make the reported DOF swing by one as
        // the user drags the geometry across a bound -- e.g. a rect
        // with interval width 1..999 would show DOF 2 at width 1 and
        // DOF 3 at width 2. The geometric DOF of the sketch is the
        // same either way; bounds are inequality constraints that
        // shouldn't count.
        jacobian.rows.retain(|r| r.label != "range");
        (jacobian, n)
    }

    /// Rank analysis of the current sketch: DOF (`nullity`) plus the
    /// null-space basis that answers candidate-row probes (see the
    /// probe module). The result is tied to the current parameter
    /// layout and geometry -- structural edits invalidate it, value
    /// drift degrades it gracefully. Updates the cached DOF.
    /// Serialize the parameters and evaluate the current cost.
    pub fn current_cost(&mut self) -> f64 {
        use arael::simple_lm::{LmProblem, RootProblem};
        let mut params = std::vec::Vec::new();
        self.serialize(&mut params);
        self.calc_cost(&params)
    }

    pub fn rank_analysis(&mut self) -> Result<arael::rank::RankResult, String> {
        let mut timer = Timer::new();
        let (jacobian, n) = self.dof_jacobian();
        let t_jac = timer.lap();
        let opts = arael::rank::RankOptions {
            null_hint: self.cached_dof(),
            ..Default::default()
        };
        let rr = jacobian.numeric_rank(&opts).map_err(|e| e.to_string())?;
        self.set_cached_dof(rr.nullity);
        if timer.on() {
            eprintln!("[RANK] m={} n={} method={:?} jacobian={:?} rank={:?} dof={}",
                jacobian.num_residuals(), n, rr.method, t_jac, timer.lap(), rr.nullity);
        }
        Ok(rr)
    }

    /// Uses nalgebra SVD for n<32, faer SVD for n>=32.
    pub fn compute_dof(&mut self, analyze: bool) -> Result<DofResult, String> {
        let mut timer = Timer::new();

        let (jacobian, n) = self.dof_jacobian();
        if n == 0 {
            return Ok(DofResult { dof: 0, param_names: Vec::new(), eigenvalues: Vec::new(), eigenvectors: Vec::new() });
        }
        // Degenerate geometry (a /len residual on collapsed entities)
        // yields NaN; the SVD and the spectral sorts below must never
        // see it. The rank_analysis path has the same guard inside
        // numeric_rank.
        if jacobian.rows.iter().any(|r| r.entries.iter().any(|&(_, v)| !v.is_finite())) {
            return Err("Jacobian contains non-finite values (degenerate geometry)".into());
        }

        let param_names = if analyze {
            let bag = SymbolBag::build(self);
            let mut names = vec![String::new(); n];
            for (name, &idx) in &bag.param_indices {
                let i = idx as usize;
                if i < n && names[i].is_empty() { names[i] = name.clone(); }
            }
            names
        } else {
            Vec::new()
        };

        let t_prep = timer.lap();
        let m = jacobian.num_residuals();

        if m == 0 {
            // No residuals: every parameter is free.
            let result = DofResult {
                dof: n,
                param_names,
                eigenvalues: vec![0.0; n],
                eigenvectors: (0..n).map(|i| {
                    let mut v = vec![0.0; n];
                    v[i] = 1.0;
                    v
                }).collect(),
            };
            self.set_cached_dof(result.dof);
            return Ok(result);
        }


        // Determine rank from the singular value spectrum of the
        // current-params Jacobian. The gap algorithm mirrors the old
        // eigenvalue version but operates on sigma directly: a sharp
        // jump between "near zero" and "constrained" shows up as a
        // large gap ratio, without the condition-number squaring
        // that killed precision in the J^T J approach.
        //
        // This is the *instantaneous* rank at the current params,
        // which can over-count DOF at configurations where two
        // Jacobian rows happen to be tangent-aligned -- classic
        // example: `hdistance L0.p2 L1.p2 = X` at a horizontal L1
        // has a row parallel to the length row at that instant, so
        // adding it doesn't lower the SVD rank even though it's
        // structurally a new constraint (rotate the triangle and it
        // decouples).
        //
        // The natural fix is to SVD a slightly-perturbed copy of the
        // params instead ("structural rank"); we tried it and backed
        // it out. Even tiny perturbations introduce a new spectrum
        // of small-but-nonzero sigmas in real sketches (robot.cmd
        // produced a sigma at 4e-5 with rel 5e-10 of max), and the
        // rank threshold that would classify the horizontal-line
        // sigma as "real" also classifies the robot sigma as "real",
        // reporting DOF=2 when the sketch actually has DOF=3. We
        // couldn't find a single threshold that handled both cases
        // without breaking another. A correct fix would track a
        // *structural* rank delta per constraint at creation time
        // and sum those instead of trusting the instantaneous SVD;
        // that's left for future work. The blocker analysis uses
        // this same current-params Jacobian so its row-span check
        // agrees with the DOF check that triggered it.
        // The gap decision itself lives in arael::rank::rank_cut; both
        // the fast path (numeric_rank) and the analyze path below use it.
        // Column-normalise the Jacobian before running rank detection.
        // Per-parameter scale differences (angular columns with
        // `radius * isigma` entries vs length columns with `isigma`
        // entries) can span >= 10^10, which makes the raw SVD spectrum
        // span the same range -- and the gap algorithm then picks a
        // spurious intra-rank gap on ill-conditioned sketches. The
        // column-normalised SVD preserves the Jacobian's null-space
        // (rank) but has a spectrum bounded by the row-space geometry,
        // where the gap algorithm is robust. Raw SVD remains available
        // via `dof singular raw` for residual-design debugging.
        let (dof, method) = if analyze {
            // Diagnostic path: full dense spectrum, since the caller
            // wants every eigenvalue/eigenvector anyway.
            let mut sorted: Vec<f64> = jacobian.singular_values_column_normalised();
            if sorted.iter().any(|v| !v.is_finite()) {
                return Err("non-finite singular values (degenerate geometry?)".into());
            }
            sorted.sort_by(|a: &f64, b| a.total_cmp(b));
            let (cut, _) = arael::rank::rank_cut(&sorted);
            let rank = sorted.len() - cut;
            (n.saturating_sub(rank), "dense svd".to_string())
        } else {
            let opts = arael::rank::RankOptions {
                null_hint: self.cached_dof(),
                ..Default::default()
            };
            let rr = jacobian.numeric_rank(&opts).map_err(|e| e.to_string())?;
            let method = match rr.method {
                arael::rank::RankMethod::Dense => "dense svd".to_string(),
                arael::rank::RankMethod::Iterative { block, grew } => {
                    format!("subspace iteration (block={} grew={})", block, grew)
                }
                arael::rank::RankMethod::Components { count, largest_n } => {
                    format!("{} independent components (largest n={})", count, largest_n)
                }
            };
            (rr.nullity, method)
        };

        let result = if analyze {
            let svd = jacobian.svd();
            let svs = &svd.singular_values;
            let k = svs.len();
            let mut eigenvalues = vec![0.0; n];
            let mut eigenvectors: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; n]).collect();
            for i in 0..k {
                eigenvalues[i] = svs[i] * svs[i];
                // svd.v is n x k row-major (right singular vectors as columns).
                // Column i contains the direction for singular value svs[i];
                // copy it into eigenvectors[i] (direction-major, param-indexed).
                for row in 0..n {
                    eigenvectors[i][row] = svd.v[row * k + i];
                }
            }
            // Fill any remaining slots with orthogonal basis vectors (not
            // mathematically meaningful beyond rank, but keeps dimensions
            // consistent for callers that iterate).
            for i in k..n {
                eigenvectors[i][i] = 1.0;
            }
            DofResult { dof, param_names, eigenvalues, eigenvectors }
        } else {
            DofResult { dof, param_names: Vec::new(), eigenvalues: Vec::new(), eigenvectors: Vec::new() }
        };
        if timer.on() {
            let t_rank = timer.lap();
            eprintln!("[DOF] m={} n={} analyze={} method={} prep+jac={:?} rank={:?} total={:?} dof={}",
                m, n, analyze, method, t_prep, t_rank, timer.total(), result.dof);
        }
        self.set_cached_dof(result.dof);
        Ok(result)
    }

    /// Return cached DOF or compute it (count only, no eigenvector
    /// analysis). Computing goes through the rank cache, so a dof()
    /// call also leaves the probe basis warm.
    pub fn dof(&mut self) -> Result<usize, String> {
        if let Some(d) = self.cached_dof() { return Ok(d); }
        Ok(self.ensure_rank()?.nullity)
    }

    /// Update tangent_la shared-endpoint flags by scanning coincident collections.
    pub fn update_tangent_flags(&mut self) {
        for t in &mut self.tangent_la {
            t.p1_arc_start = self.coincident_lp1_arc_start.iter().any(|c| c.line == t.line && c.arc == t.arc);
            t.p1_arc_end = self.coincident_lp1_arc_end.iter().any(|c| c.line == t.line && c.arc == t.arc);
            t.p2_arc_start = self.coincident_lp2_arc_start.iter().any(|c| c.line == t.line && c.arc == t.arc);
            t.p2_arc_end = self.coincident_lp2_arc_end.iter().any(|c| c.line == t.line && c.arc == t.arc);
            // Compute dir_sign for shared-endpoint directed tangent constraint.
            // Only set once (when 0.0) to remember the initial tangent direction;
            // recomputing each time would follow perturbations instead of preventing flips.
            let shared = t.p1_arc_start || t.p1_arc_end || t.p2_arc_start || t.p2_arc_end;
            if shared && t.dir_sign.is_nan() {
                let l = &self.lines[t.line];
                let a = &self.arcs[t.arc];
                // Compute arc endpoint and tangent from arc parameters
                let angle = if t.p1_arc_start || t.p2_arc_start { a.start_angle.value } else { a.end_angle.value };
                let ap = a.point_at(angle);
                let tv = a.tangent_at(angle);
                // Direction from arc endpoint to the line's other end
                let (dx, dy) = if t.p1_arc_start || t.p1_arc_end {
                    (l.p2.value.x - ap.x, l.p2.value.y - ap.y)
                } else {
                    (l.p1.value.x - ap.x, l.p1.value.y - ap.y)
                };
                let dot = dx * tv.x + dy * tv.y;
                t.dir_sign = if dot >= 0.0 { 1.0 } else { -1.0 };
            }
        }
    }

    /// Compute dir_sign for perpendicular constraints on first use.
    pub fn update_perpendicular_flags(&mut self) {
        for p in &mut self.perpendicular {
            if p.dir_sign.is_nan() {
                let la = &self.lines[p.a];
                let lb = &self.lines[p.b];
                let cross = (la.p2.value.x - la.p1.value.x) * (lb.p2.value.y - lb.p1.value.y)
                          - (la.p2.value.y - la.p1.value.y) * (lb.p2.value.x - lb.p1.value.x);
                p.dir_sign = if cross >= 0.0 { 1.0 } else { -1.0 };
            }
        }
    }

    /// Initialize h_dir_sign / v_dir_sign on lines with horizontal/vertical
    /// constraints the first time they're seen. Sketches loaded from JSON
    /// that predate the dir_sign fields deserialize with NaN; this backfills
    /// them from whatever orientation the loaded geometry has.
    pub fn update_line_dir_flags(&mut self) {
        for l in self.lines.iter_mut() {
            if l.constraints.horizontal && l.constraints.h_dir_sign.is_nan() {
                let dx = l.p2.value.x - l.p1.value.x;
                l.constraints.h_dir_sign = if dx >= 0.0 { 1.0 } else { -1.0 };
            }
            if l.constraints.vertical && l.constraints.v_dir_sign.is_nan() {
                let dy = l.p2.value.y - l.p1.value.y;
                l.constraints.v_dir_sign = if dy >= 0.0 { 1.0 } else { -1.0 };
            }
        }
    }

    /// Add hidden helper Points coincident with every "free" Line/Arc
    /// endpoint -- one that doesn't already appear in any coincident_*
    /// constraint linking it to another entity. Returns a state token
    /// that must be passed to `remove_drag_auto_anchors` at drag-end.
    ///
    /// # Why this exists (a hack we couldn't avoid)
    ///
    /// In long equal-length chains, the `length L = c` Jacobian is
    /// rank-deficient when a segment is perpendicular to the chain's
    /// drag direction (`sqrt(dx^2 + dy^2)` has zero derivative w.r.t.
    /// `dx` when `dx ~ 0`). At that pose nothing in the constraint
    /// system pulls a free endpoint along the chain's translation
    /// axis: `length` doesn't (rank-deficient), `equal` doesn't (only
    /// couples lengths), `coincident` only acts on joined endpoints,
    /// and the `drift` regularizer is six orders of magnitude weaker
    /// than the constraints. Result: per-frame drag translates the
    /// chain forward but leaves the free endpoint behind, so the
    /// last segment rotates 90 degrees away from the chain's axis,
    /// then snaps back, then flips again -- visibly unstable.
    ///
    /// Boosting `drift_isigma` is the "obvious" fix and doesn't work
    /// at long chains (the boost needed to lock down 20 articulation
    /// modes also makes the chain feel rigid). Adding a free Point
    /// coincident with the free endpoint -- the user's empirical
    /// workaround -- does fix it: the new Point's drift residual
    /// gets folded into the soft chain eigenmode through the
    /// coincident, so the Newton step now displaces the free
    /// endpoint along with the chain instead of leaving it stationary.
    /// We don't fully understand why this works at chain20 but a
    /// straight drift boost doesn't (Hessian spectra differ only in
    /// the 4th decimal at start-of-solve), only that it does. So we
    /// automate the workaround.
    ///
    /// # Mechanics
    ///
    /// Walks every coincident_* vec to identify joined endpoints,
    /// then for each Line p1/p2 (optimizable, not joined) and each
    /// Arc start/end (optimizable, not joined) pushes a hidden helper
    /// `Point` plus a matching `coincident_lp1`/`coincident_lp2`/
    /// `coincident_arc_start`/`coincident_arc_end`. The helper is
    /// placed at the endpoint's current position with a 0.001 offset
    /// so the parser doesn't reject it as a no-op coincidence.
    ///
    /// Call AFTER pushing the drag apparatus onto the sketch so the
    /// dragged endpoint is already "joined" (to the drag helper) and
    /// won't get a redundant auto-anchor.
    pub fn add_drag_auto_anchors(&mut self) -> DragAutoAnchorState {
        let mut state = DragAutoAnchorState {
            helper_points: std::vec::Vec::new(),
        };

        // Build sets of joined endpoints by walking every coincident_*
        // that touches a Line endpoint or Arc start/end.
        let mut joined_lp1: std::collections::HashSet<Ref<Line>> = std::collections::HashSet::new();
        let mut joined_lp2: std::collections::HashSet<Ref<Line>> = std::collections::HashSet::new();
        let mut joined_arc_start: std::collections::HashSet<Ref<Arc>> = std::collections::HashSet::new();
        let mut joined_arc_end: std::collections::HashSet<Ref<Arc>> = std::collections::HashSet::new();

        for c in &self.coincident_lp1 { joined_lp1.insert(c.line); }
        for c in &self.coincident_lp2 { joined_lp2.insert(c.line); }
        for c in &self.coincident_ll11 { joined_lp1.insert(c.a); joined_lp1.insert(c.b); }
        for c in &self.coincident_ll12 { joined_lp1.insert(c.a); joined_lp2.insert(c.b); }
        for c in &self.coincident_ll21 { joined_lp2.insert(c.a); joined_lp1.insert(c.b); }
        for c in &self.coincident_ll22 { joined_lp2.insert(c.a); joined_lp2.insert(c.b); }
        for c in &self.coincident_lp1_arc_center { joined_lp1.insert(c.line); }
        for c in &self.coincident_lp2_arc_center { joined_lp2.insert(c.line); }
        for c in &self.coincident_lp1_arc_start  { joined_lp1.insert(c.line); joined_arc_start.insert(c.arc); }
        for c in &self.coincident_lp2_arc_start  { joined_lp2.insert(c.line); joined_arc_start.insert(c.arc); }
        for c in &self.coincident_lp1_arc_end    { joined_lp1.insert(c.line); joined_arc_end.insert(c.arc); }
        for c in &self.coincident_lp2_arc_end    { joined_lp2.insert(c.line); joined_arc_end.insert(c.arc); }
        for c in &self.coincident_arc_start { joined_arc_start.insert(c.arc); }
        for c in &self.coincident_arc_end   { joined_arc_end.insert(c.arc); }
        for c in &self.coincident_arc_center_start { joined_arc_start.insert(c.b); }
        for c in &self.coincident_arc_start_center { joined_arc_start.insert(c.a); }
        for c in &self.coincident_arc_center_end { joined_arc_end.insert(c.b); }
        for c in &self.coincident_arc_end_center { joined_arc_end.insert(c.a); }
        for c in &self.coincident_arc_start_start { joined_arc_start.insert(c.a); joined_arc_start.insert(c.b); }
        for c in &self.coincident_arc_start_end   { joined_arc_start.insert(c.a); joined_arc_end.insert(c.b); }
        for c in &self.coincident_arc_end_start   { joined_arc_end.insert(c.a); joined_arc_start.insert(c.b); }
        for c in &self.coincident_arc_end_end     { joined_arc_end.insert(c.a); joined_arc_end.insert(c.b); }

        // Lines
        let line_refs: std::vec::Vec<Ref<Line>> = self.lines.refs().collect();
        for r in line_refs {
            let l = &self.lines[r];
            if l.p1.optimize && !joined_lp1.contains(&r) {
                let pos = vect2d::new(l.p1.value.x + 0.001, l.p1.value.y);
                let p = self.add_helper_point(pos);
                self.coincident_lp1.push(CoincidentLP1 {
                    line: r, point: p, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
                });
                state.helper_points.push(p);
            }
            let l = &self.lines[r];
            if l.p2.optimize && !joined_lp2.contains(&r) {
                let pos = vect2d::new(l.p2.value.x + 0.001, l.p2.value.y);
                let p = self.add_helper_point(pos);
                self.coincident_lp2.push(CoincidentLP2 {
                    line: r, point: p, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
                });
                state.helper_points.push(p);
            }
        }

        // Arcs (only optimizable start/end angles -- skip closed circles
        // where the angles are fixed)
        let arc_refs: std::vec::Vec<Ref<Arc>> = self.arcs.refs().collect();
        for r in arc_refs {
            let a = &self.arcs[r];
            if a.start_angle.optimize && !joined_arc_start.contains(&r) {
                let pt = a.start_pos();
                let pos = vect2d::new(pt.x + 0.001, pt.y);
                let p = self.add_helper_point(pos);
                self.coincident_arc_start.push(CoincidentArcStart {
                    point: p, arc: r, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
                });
                state.helper_points.push(p);
            }
            let a = &self.arcs[r];
            if a.end_angle.optimize && !joined_arc_end.contains(&r) {
                let pt = a.end_pos();
                let pos = vect2d::new(pt.x + 0.001, pt.y);
                let p = self.add_helper_point(pos);
                self.coincident_arc_end.push(CoincidentArcEnd {
                    point: p, arc: r, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
                });
                state.helper_points.push(p);
            }
        }

        state
    }

    /// Roll back the auto-anchors set up by `add_drag_auto_anchors`.
    pub fn remove_drag_auto_anchors(&mut self, state: &DragAutoAnchorState) {
        // By identity, not truncate-to-length: anything else pushed or
        // removed during the gesture cannot desync the rollback, and
        // the same state can roll back a deserialized clone. The
        // anchors only ever bridge through these four collections
        // (see add_drag_auto_anchors), and nothing else can reference
        // an invisible anchor helper during a gesture -- one pass over
        // each beats a full registry sweep per anchor point.
        if state.helper_points.is_empty() {
            return;
        }
        let set: std::collections::HashSet<arael::refs::Ref<Point>> =
            state.helper_points.iter().copied().collect();
        self.coincident_lp1.retain(|c| !set.contains(&c.point));
        self.coincident_lp2.retain(|c| !set.contains(&c.point));
        self.coincident_arc_start.retain(|c| !set.contains(&c.point));
        self.coincident_arc_end.retain(|c| !set.contains(&c.point));
        for p in &state.helper_points {
            if self.points.get(*p).is_some() {
                self.points.remove(*p);
            }
        }
    }

    /// Solve the sketch constraints using Levenberg-Marquardt.
    /// Uses sparse faer Cholesky for n >= 64 params, dense Cholesky otherwise.
    /// When starting cost is high, uses graduated optimization (1% -> 10% ->
    /// 100% constraint strength) to avoid ill-conditioning from the large
    /// constraint/drift sigma ratio.
    pub fn solve(&mut self) -> arael::simple_lm::LmResult<f64> {
        self.prepare_derived();
        self.solve_prepared()
    }

    /// Rebuild the state derived from the sketch's STRUCTURE: the expression
    /// constraints, the symbol bag they resolve against, and the tangent /
    /// perpendicular / line-direction flags.
    ///
    /// [`solve`](Self::solve) runs this every time, which is what a `Sketch`
    /// held on its own needs -- its fields are public, so nothing can know
    /// whether the structure moved. [`SketchCell`] does know, and skips it
    /// when it has not.
    ///
    /// The flag passes read geometry only to initialize a `dir_sign` that is
    /// `NaN`, and are written never to recompute one -- following the
    /// perturbations would let a constraint flip -- so they are already
    /// idempotent on values.
    pub fn prepare_derived(&mut self) {
        // Rebuild expression constraints from dimensions with expr_str
        // (needed after load/undo since expr_constraints is not serialized)
        self.rebuild_expr_constraints();
        self.update_tangent_flags();
        self.update_perpendicular_flags();
        self.update_line_dir_flags();
        if !self.expr_constraints.is_empty() {
            let filter = self.expr_entity_filter();
            let bag = SymbolBag::build_filtered(self, Some(&filter));
            for ec in &mut self.expr_constraints {
                ec.resolve(&bag);
            }
            self.symbol_bag = Some(bag);
        }
    }

    /// Solve without refreshing the derived state first. Only correct when the
    /// caller knows it is current -- [`SketchCell::solve`] does.
    pub fn solve_prepared(&mut self) -> arael::simple_lm::LmResult<f64> {
        // A session that lives just for this call. The graduated stages are
        // three solves of one structure -- it cannot change between them --
        // so they share the analysis instead of each paying for it. Building
        // one is free: nothing is analyzed until the first solve.
        let mut session = arael::simple_lm::LmSession::new(arael::simple_lm::SparseFaer::new());
        self.solve_prepared_with(&mut session)
    }

    /// The same, through a warm [`LmSession`](arael::simple_lm::LmSession)
    /// when one is supplied: the sparsity pattern, ordering and symbolic
    /// factorization are reused instead of rebuilt. The session must have been
    /// built for THIS structure -- [`SketchCell::solve`] keys one on the
    /// structural generation, which is what makes that safe.
    ///
    /// The graduated stages share it: they scale the residuals, which changes
    /// values and not the pattern.
    pub fn solve_prepared_with(
        &mut self,
        session: &mut arael::simple_lm::LmSession<f64, arael::simple_lm::SparseFaer<f64>>,
    ) -> arael::simple_lm::LmResult<f64> {
        let mut timer = Timer::new();
        use arael::simple_lm::LmProblem;

        let t_prel = timer.lap();

        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        self.serialize(&mut params64);
        let n = params64.len();

        if n == 0 {
            return arael::simple_lm::LmResult {
                x: params64, start_cost: 0.0, end_cost: 0.0,
                iterations: 0, accepted_iterations: 0,
                status: arael::simple_lm::LmStatus::Converged, final_lambda: 0.0,
                timing: None,
                solver: None,
            };
        }

        // Compute starting cost to decide strategy
        let start_cost = self.calc_cost(&params64);

        // Graduated optimization: when starting cost is high, the Hessian
        // condition number (constraint_isigma/drift_isigma)^2 can make LM
        // oscillate. Solve with reduced constraint strength first to get
        // close to the solution, then ramp up to full strength.
        let full_isigma = self.constraint_isigma;
        let graduated = start_cost > n as f64 * 1e-3;
        // (gradiation, cost_threshold)
        // Per-parameter cost at which a full-strength stage is converged.
        // The number has units: at drift_isigma 1e-3 a per-parameter cost of
        // 1e-8 is every entity within ~0.1 units of where it sat, and at
        // constraint_isigma 1e3 constraint violations near 1e-7 units. The
        // early stages loosen it by strength^2 -- cost scales with the square
        // of the residual weight, so each stage's threshold means the same
        // geometric accuracy.
        let base_cost = 1e-8;
        let stages: &[(f64, f64)] = if graduated {
            &[
              (0.01, n as f64 * base_cost * 1e4),
              (0.1, n as f64 * base_cost * 1e2),
              (1.0, n as f64 * base_cost * 1e0)
            ]
        } else {
            &[(1.0, n as f64 * base_cost)]
        };

        let mut total_iters = 0usize;
        let mut total_accepted = 0usize;
        let mut result = arael::simple_lm::LmResult {
            x: params64.clone(),
            start_cost,
            end_cost: start_cost,
            iterations: 0,
            accepted_iterations: 0,
            // Both are overwritten from each stage's solve below; these are
            // the no-stage placeholders.
            status: arael::simple_lm::LmStatus::Converged,
            final_lambda: 0.0,
            timing: None,
            solver: None,
        };

        let t_prep = timer.lap();

        for (scale, cost_threshold) in stages {
            self.constraint_isigma = full_isigma * scale;

            let mut params = std::vec::Vec::new();
            self.serialize(&mut params);

            // Start at the ladder's floor and let rejections raise lambda.
            // Pre-damping by distance from the solution was the old behavior,
            // and it over-damped exactly the solves that needed full
            // Gauss-Newton steps: a rejection costs one factorization, a high
            // starting lambda costs a dozen iterations of walking it back down.
            let lambda = 1e-6;

            let config = arael::simple_lm::LmConfig::<f64> {
                initial_lambda: lambda,
                //abs_precision: 1e-6,
                //rel_precision: 1e-4,
                abs_precision: 0.0,
                rel_precision: 0.0,
                cost_threshold: *cost_threshold,
                min_iters: 1,
                //gradient_tolerance: Some(1e-6),
                verbose: verbose(),
                gather_timing: verbose(),
                ..Default::default()
            };
            // The session follows the BACKEND, not a size threshold of our
            // own: simple_lm::solve is dense only at 6 parameters or fewer and
            // is the same sparse backend above that, so anything larger has an
            // analysis worth keeping.
            let stage_result = if n > 6 {
                session.solve_x0(&params, self, &config)
            } else {
                arael::simple_lm::solve(&params, self, &config)
            };
            match stage_result {
                Ok(r) => {
                    self.deserialize(&r.x);
                    total_iters += r.iterations;
                    total_accepted += r.accepted_iterations;
                    result.end_cost = r.end_cost;
                    result.x = r.x;
                    result.status = r.status;
                    result.final_lambda = r.final_lambda;
                    // Accumulate phase timing across the stages.
                    if let Some(t) = r.timing {
                        let acc = result.timing.get_or_insert_with(Default::default);
                        acc.total += t.total;
                        acc.assembly += t.assembly;
                        acc.analysis += t.analysis;
                        acc.linear_solve += t.linear_solve;
                        acc.cost_eval += t.cost_eval;
                        acc.advance += t.advance;
                        acc.assembly_count += t.assembly_count;
                        acc.analysis_count += t.analysis_count;
                        acc.linear_solve_count += t.linear_solve_count;
                        acc.cost_eval_count += t.cost_eval_count;
                        acc.advance_count += t.advance_count;
                        if acc.steps.is_empty() {
                            acc.first_assembly = t.first_assembly;
                            acc.first_linear_solve = t.first_linear_solve;
                            acc.first_cost_eval = t.first_cost_eval;
                            acc.first_advance = t.first_advance;
                        }
                        acc.steps.extend(t.steps);
                    }
                }
                Err(e) => {
                    // A broken stage (degenerate diagonal or setup
                    // failure): keep the best accepted state when there is
                    // one, report Aborted through the summary status, and
                    // stop the ramp -- the interactive sketch must not
                    // panic, and the previous committed values stand.
                    eprintln!("sketch solve stage failed: {}", e);
                    if let Some(p) = e.into_partial() {
                        self.deserialize(&p.x);
                        total_iters += p.iterations;
                        total_accepted += p.accepted_iterations;
                        result.end_cost = p.end_cost;
                        result.x = p.x;
                        result.final_lambda = p.final_lambda;
                    }
                    result.status = arael::simple_lm::LmStatus::Aborted;
                    break;
                }
            }
        }

        let t_solve = timer.lap();

        self.constraint_isigma = full_isigma;
        self.normalise_ellipse_rotations();
        self.update_expr_dim_values();
        result.iterations = total_iters;
        result.accepted_iterations = total_accepted;
        let t_finish = timer.lap();
        if timer.on() {
            println!(
                "solve end, total {:?}, final cost {}, iters {} ({} accepted): prel={:?}, prep={:?}, solve={:?}, finish={:?}",
                timer.total(), result.end_cost, result.iterations,
                result.accepted_iterations, t_prel, t_prep, t_solve, t_finish);
        }
        result
    }

    /// Wrap every ellipse's `rotation` param into `(-pi, pi]`. Angles
    /// can accumulate integer multiples of `2*pi` across repeated drags
    /// or expression-driven updates; without this post-pass the solver
    /// can settle on rotations like `3.5*pi` that are numerically fine
    /// but render as nonsense in `info` / `list` output. Wrapping by
    /// whole `2*pi` turns leaves every geometric quantity (point
    /// positions, tangents, curvatures) unchanged; the drift anchor is
    /// the committed `.value` itself, so next solve's drift picks up
    /// the wrapped angle as its new anchor automatically.
    fn normalise_ellipse_rotations(&mut self) {
        use arael::utils::rad2rad;
        let refs: std::vec::Vec<_> = self.arcs.refs().collect();
        for r in refs {
            let a = &mut self.arcs[r];
            if !a.is_ellipse { continue; }
            // Only `rotation` is normalised; `start_angle` / `end_angle`
            // parameterise the traversal of the ellipse and may span more
            // than 2*pi (e.g. a "large" elliptic arc where the sweep
            // itself exceeds pi). Wrapping them would silently change
            // which side of the ellipse is drawn.
            a.rotation.value = rad2rad(a.rotation.value);
        }
    }

    /// Evaluate expression/derived dimensions and user params, cache their computed values.
    pub fn update_expr_dim_values(&mut self) {
        let has_work = self.dimensions.iter().any(|d| d.expr_str.is_some() || d.derived || d.range.is_some())
            || self.user_params.iter().any(|p| !p.broken);
        if !has_work { return; }
        // The full bag walks every entity (tens of thousands of
        // formatted symbols on a big sketch) to evaluate a handful of
        // expressions. Collect the entity names the expressions can
        // reach -- prefixes of dotted symbols, closed over dim and
        // user-param references -- and build the bag for those only.
        let entities = self.expr_entity_filter();
        let bag = SymbolBag::build_filtered(self, Some(&entities));
        let mut params = Vec::new();
        self.serialize(&mut params);
        let vars = bag.eval_vars(&params);
        // Update user params first (dims may reference them)
        for p in &mut self.user_params {
            if p.broken { continue; }
            if p.expr_str.trim().parse::<f64>().is_ok() { continue; }
            if let Ok(parsed) = arael_sym::parse(&p.expr_str) {
                let expanded = expr_constraint::expand_derived(&parsed, &bag);
                match expanded.eval(&vars) {
                    Ok(val) => p.value = val,
                    Err(_) => p.broken = true,
                }
            }
        }
        for dim in &mut self.dimensions {
            if dim.broken { continue; }
            if let Some(ref expr_str) = dim.expr_str
                && let Ok(parsed) = arael_sym::parse(expr_str) {
                    let expanded = expr_constraint::expand_derived(&parsed, &bag);
                    match expanded.eval(&vars) {
                        Ok(val) => dim.value = val,
                        Err(_) => dim.broken = true,
                    }
                }
        }
        // Update derived numeric dims from measured geometry
        let derived_vals: Vec<(usize, f64)> = (0..self.dimensions.len())
            .filter(|&i| self.dimensions[i].derived && self.dimensions[i].expr_str.is_none() && !self.dimensions[i].broken)
            .filter_map(|i| {
                let measured = self.dimensions[i].measured_symbol(self);
                let expanded = expr_constraint::expand_derived(&measured, &bag);
                expanded.eval(&vars).ok().map(|v| (i, v))
            })
            .collect();
        for (i, val) in derived_vals {
            self.dimensions[i].value = val;
        }
        // Range dimensions track the current measured value in `value` for
        // display (the bound itself lives in `range`). Same eval shape as
        // the derived-dim update above.
        let range_vals: Vec<(usize, f64)> = (0..self.dimensions.len())
            .filter(|&i| self.dimensions[i].range.is_some())
            .filter_map(|i| {
                let measured = self.dimensions[i].measured_symbol(self);
                let expanded = expr_constraint::expand_derived(&measured, &bag);
                expanded.eval(&vars).ok().map(|v| (i, v))
            })
            .collect();
        for (i, val) in range_vals {
            self.dimensions[i].value = val;
        }
    }
}

#[cfg(test)]
mod jacobian_tests {
    use super::*;
    use arael::simple_lm::LmProblem;
    use arael::vect::vect2d;

    #[test]
    fn timer_is_inert_when_verbose_off() {
        assert!(!verbose());
        let mut t = Timer::new();
        assert!(!t.on());
        assert_eq!(t.lap(), std::time::Duration::ZERO);
        assert_eq!(t.total(), std::time::Duration::ZERO);
    }

    /// Build a sketch with lines, coincident constraint, and an expression
    /// dimension, then validate Jacobian against Hessian and cost.
    fn make_test_sketch() -> (Sketch, Vec<f64>) {
        let mut sketch = Sketch::new();
        let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
        let l1 = sketch.add_line(vect2d::new(3.0, 0.0), vect2d::new(5.0, 2.0));
        // Coincident: L0.p2 == L1.p1
        sketch.coincident_ll21.push(CoincidentLL21 {
            a: l0,
            b: l1,
            nid: 0, cid: 0,
            hb: arael::model::CrossBlock::new(),
        });
        // Length dimension on L0 (creates an expression constraint)
        sketch.lines[l0].constraints.has_length = true;
        sketch.lines[l0].constraints.length = 5.0;
        sketch.dimensions.push(Dimension {
            did: 0,
            kind: DimensionKind::LineLength(l0),
            value: 5.0, offset: vect2d::new(0.0, 1.0), text_along: 0.0,
            name: "d0".into(), expr_str: None, broken: false, derived: false,
            range: None,
        });

        sketch.prepare_expr_constraints();
        let mut params = Vec::new();
        sketch.serialize(&mut params);
        (sketch, params)
    }

    #[test]
    fn sketch_jacobian_cost_matches() {
        let (mut sketch, mut params) = make_test_sketch();
        // Perturb so residuals are non-zero
        params[0] += 0.1;
        params[1] += 0.2;
        params[4] -= 0.3;

        let j = sketch.calc_jacobian(&params);
        let cost_j: f64 = j.rows.iter().map(|r| r.residual * r.residual).sum();
        let cost_c = sketch.calc_cost(&params);
        assert!(
            (cost_j - cost_c).abs() < 1e-10,
            "cost mismatch: jacobian={}, calc_cost={}", cost_j, cost_c
        );
    }

    #[test]
    fn sketch_jacobian_jtj_matches_hessian() {
        let (mut sketch, mut params) = make_test_sketch();
        params[0] += 0.1;
        params[1] += 0.2;
        params[4] -= 0.3;

        let j = sketch.calc_jacobian(&params);
        let dense = j.to_dense();
        let m = j.num_residuals();
        let n = j.num_params;

        // J^T * J
        let mut jtj = vec![0.0f64; n * n];
        for i in 0..n {
            for k in 0..n {
                let mut s = 0.0;
                for r in 0..m { s += dense[r * n + i] * dense[r * n + k]; }
                jtj[i * n + k] = s;
            }
        }

        // Hessian = 2 * J^T * J
        let mut grad = vec![0.0f64; n];
        let mut hessian = vec![0.0f64; n * n];
        sketch.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

        for i in 0..n {
            for k in 0..n {
                let expected = 2.0 * jtj[i * n + k];
                let actual = hessian[i * n + k];
                assert!(
                    (expected - actual).abs() < 1e-8,
                    "H[{},{}] mismatch: 2*JtJ={}, H={}", i, k, expected, actual
                );
            }
        }
    }

    #[test]
    fn sketch_jacobian_gradient_matches() {
        let (mut sketch, mut params) = make_test_sketch();
        params[0] += 0.1;
        params[1] += 0.2;

        let j = sketch.calc_jacobian(&params);
        let n = j.num_params;

        let mut grad_j = vec![0.0f64; n];
        for row in &j.rows {
            for &(idx, d) in &row.entries {
                grad_j[idx as usize] += 2.0 * row.residual * d;
            }
        }

        let mut grad = vec![0.0f64; n];
        let mut hessian = vec![0.0f64; n * n];
        sketch.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

        for i in 0..n {
            assert!(
                (grad_j[i] - grad[i]).abs() < 1e-8,
                "grad[{}] mismatch: J={}, GH={}", i, grad_j[i], grad[i]
            );
        }
    }
}
