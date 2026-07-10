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
pub mod symbol_bag;
pub use symbol_bag::SymbolBag;
pub mod expr_constraint;
pub use expr_constraint::ExpressionConstraint;
pub mod blocker;
pub use blocker::{BlockerReport, analyze as analyze_blockers};

use arael::model::{CrossBlock, JacobianModel, Param, SelfBlock, TripletBlock};

const TIMING_DEBUG: bool = false;
use arael::vect::vect2d;
use arael::refs::{Ref, Arena};

// Entity and constraint types must share one module scope because the
// #[arael::model] macro emits private `_PARAM_COUNT` constants that
// CrossBlock<A, B> expansions need to reference.
include!("entities.rs");
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
    coincident_lp1_len: usize,
    coincident_lp2_len: usize,
    coincident_arc_start_len: usize,
    coincident_arc_end_len: usize,
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
    #[serde(default)]
    pub verbose: bool,
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
    // Dimension annotations
    #[arael(skip)]
    pub dimensions: std::vec::Vec<Dimension>,
    pub next_dimension_id: u32,
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
    // Cached DOF count — set by compute_dof(), cleared on structural mutation
    #[arael(skip)]
    #[serde(skip)]
    pub cached_dof: Option<usize>,
}

fn default_min_length() -> f64 { 0.0001 }
fn default_next_constraint_id() -> u32 { 1 }

/// Format a synthetic constraint name for a flag-style constraint on a
/// named entity. Pattern: "C<entity><flag>", e.g. "CL0H", "CL3V".
pub fn format_flag_name(entity_name: &str, flag: char) -> String {
    format!("C{}{}", entity_name, flag)
}

/// Parse a synthetic constraint name into its entity-name + flag-char
/// components. Returns None for names that don't match "C<entity>F"
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
            s.distance_pp.iter().find(|c| c.a == *pa && c.b == *pb).map(|c| c.cid),
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
        (LineP1(_), ArcCenter(_)) | (ArcCenter(_), LineP1(_))
        | (LineP2(_), ArcCenter(_)) | (ArcCenter(_), LineP2(_))
        | (LineP1(_), ArcStart(_)) | (ArcStart(_), LineP1(_))
        | (LineP2(_), ArcStart(_)) | (ArcStart(_), LineP2(_))
        | (LineP1(_), ArcEnd(_)) | (ArcEnd(_), LineP1(_))
        | (LineP2(_), ArcEnd(_)) | (ArcEnd(_), LineP2(_)) => None,
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
            verbose: false,
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
            dimensions: Vec::new(),
            next_dimension_id: 0,
            next_constraint_id: 1,
            user_params: Vec::new(),
            expr_constraints: Vec::new(),
            symbol_bag: None,
            expr_hb: TripletBlock::new(),
            cached_dof: None,
        }
    }

    /// Walk every Vec-stored constraint in canonical order and assign a
    /// numeric id (C<nid>) to any constraint whose nid is still the 0
    /// sentinel. Call at the tail of every mutating action and after
    /// loading a sketch so freshly-deserialised sketches pick up names.
    pub fn assign_constraint_names(&mut self) {
        macro_rules! assign {
            ($($field:ident),* $(,)?) => {
                $(
                    for c in self.$field.iter_mut() {
                        if c.nid == 0 {
                            c.nid = self.next_constraint_id;
                            self.next_constraint_id += 1;
                        }
                    }
                )*
            };
        }
        assign!(
            coincident_pp,
            coincident_lp1, coincident_lp2,
            coincident_ll11, coincident_ll12, coincident_ll21, coincident_ll22,
            distance_pp, hdistance_pp, vdistance_pp,
            point_on_line,
            midpoint, midpoint_lp1, midpoint_lp2,
            midpoint_arc_start, midpoint_arc_end,
            midpoint_arc_point,
            midpoint_lp1_arc, midpoint_lp2_arc,
            midpoint_arc_start_arc, midpoint_arc_end_arc,
            point_on_arc,
            parallel, perpendicular, collinear,
            equal_length, angle,
            tangent_la, concentric, equal_radius, tangent_aa,
            symmetry_ll, symmetry_pp, symmetry_aa,
            distance_pl, distance_lp1l, distance_lp2l,
            distance_arc_center_l, distance_arc_start_l, distance_arc_end_l,
            line_p1_on_line, line_p2_on_line,
            coincident_arc_center, coincident_arc_start, coincident_arc_end,
            coincident_lp1_arc_center, coincident_lp2_arc_center,
            coincident_lp1_arc_start, coincident_lp2_arc_start,
            coincident_lp1_arc_end, coincident_lp2_arc_end,
            coincident_arc_center_start, coincident_arc_center_end,
            coincident_arc_start_center, coincident_arc_end_center,
            coincident_arc_start_start, coincident_arc_start_end,
            coincident_arc_end_start, coincident_arc_end_end,
            line_p1_on_arc, line_p2_on_arc,
            distance_ll11, distance_ll12, distance_ll21, distance_ll22,
            distance_lp1, distance_lp2,
            distance_arc_center_p, distance_arc_start_p, distance_arc_end_p,
            distance_arc_center_l1, distance_arc_center_l2,
            distance_arc_start_l1, distance_arc_start_l2,
            distance_arc_end_l1, distance_arc_end_l2,
            distance_aa_ce_ce, distance_aa_ce_s, distance_aa_ce_e,
            distance_aa_s_ce, distance_aa_s_s, distance_aa_s_e,
            distance_aa_e_ce, distance_aa_e_s, distance_aa_e_e,
            distance_concentric,
            axis_distance_ll11, axis_distance_ll12, axis_distance_ll21, axis_distance_ll22,
            axis_distance_lp1, axis_distance_lp2,
            axis_distance_arc_center_p, axis_distance_arc_start_p, axis_distance_arc_end_p,
            axis_distance_arc_center_l1, axis_distance_arc_center_l2,
            axis_distance_arc_start_l1, axis_distance_arc_start_l2,
            axis_distance_arc_end_l1, axis_distance_arc_end_l2,
            axis_distance_aa_ce_ce, axis_distance_aa_ce_s, axis_distance_aa_ce_e,
            axis_distance_aa_s_ce, axis_distance_aa_s_s, axis_distance_aa_s_e,
            axis_distance_aa_e_ce, axis_distance_aa_e_s, axis_distance_aa_e_e,
        );
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
        self.coincident_pp.retain(|c| c.a != r && c.b != r);
        self.coincident_lp1.retain(|c| c.point != r);
        self.coincident_lp2.retain(|c| c.point != r);
        self.distance_pp.retain(|c| c.a != r && c.b != r);
        self.hdistance_pp.retain(|c| c.a != r && c.b != r);
        self.vdistance_pp.retain(|c| c.a != r && c.b != r);
        self.point_on_line.retain(|c| c.point != r);
        self.midpoint.retain(|c| c.point != r);
        self.midpoint_arc_point.retain(|c| c.point != r);
        self.point_on_arc.retain(|c| c.point != r);
        self.distance_pl.retain(|c| c.point != r);
        self.coincident_arc_center.retain(|c| c.point != r);
        self.coincident_arc_start.retain(|c| c.point != r);
        self.coincident_arc_end.retain(|c| c.point != r);
        self.distance_lp1.retain(|c| c.point != r);
        self.distance_lp2.retain(|c| c.point != r);
        self.distance_arc_center_p.retain(|c| c.point != r);
        self.distance_arc_start_p.retain(|c| c.point != r);
        self.distance_arc_end_p.retain(|c| c.point != r);
        self.symmetry_pp.retain(|c| c.a != r && c.c != r);
        self.axis_distance_lp1.retain(|c| c.point != r);
        self.axis_distance_lp2.retain(|c| c.point != r);
        self.axis_distance_arc_center_p.retain(|c| c.point != r);
        self.axis_distance_arc_start_p.retain(|c| c.point != r);
        self.axis_distance_arc_end_p.retain(|c| c.point != r);
    }

    /// Remove a line and all constraints referencing it.
    pub fn delete_line(&mut self, r: Ref<Line>) {
        self.dimensions.retain(|d| !d.kind.references_line(r));
        self.lines.remove(r);
        self.coincident_lp1.retain(|c| c.line != r);
        self.coincident_lp2.retain(|c| c.line != r);
        self.coincident_ll11.retain(|c| c.a != r && c.b != r);
        self.coincident_ll12.retain(|c| c.a != r && c.b != r);
        self.coincident_ll21.retain(|c| c.a != r && c.b != r);
        self.coincident_ll22.retain(|c| c.a != r && c.b != r);
        self.point_on_line.retain(|c| c.line != r);
        self.midpoint.retain(|c| c.line != r);
        self.midpoint_lp1.retain(|c| c.line != r && c.target != r);
        self.midpoint_lp2.retain(|c| c.line != r && c.target != r);
        self.midpoint_arc_start.retain(|c| c.line != r);
        self.midpoint_arc_end.retain(|c| c.line != r);
        self.midpoint_lp1_arc.retain(|c| c.line != r);
        self.midpoint_lp2_arc.retain(|c| c.line != r);
        self.parallel.retain(|c| c.a != r && c.b != r);
        self.perpendicular.retain(|c| c.a != r && c.b != r);
        self.collinear.retain(|c| c.a != r && c.b != r);
        self.equal_length.retain(|c| c.a != r && c.b != r);
        self.angle.retain(|c| c.a != r && c.b != r);
        self.tangent_la.retain(|c| c.line != r);
        self.symmetry_ll.retain(|c| c.a != r && c.b != r && c.c != r);
        self.symmetry_pp.retain(|c| c.line != r);
        self.symmetry_aa.retain(|c| c.line != r);
        self.distance_pl.retain(|c| c.line != r);
        self.distance_lp1l.retain(|c| c.a != r && c.b != r);
        self.distance_lp2l.retain(|c| c.a != r && c.b != r);
        self.distance_arc_center_l.retain(|c| c.line != r);
        self.distance_arc_start_l.retain(|c| c.line != r);
        self.distance_arc_end_l.retain(|c| c.line != r);
        self.line_p1_on_line.retain(|c| c.a != r && c.b != r);
        self.line_p2_on_line.retain(|c| c.a != r && c.b != r);
        self.coincident_lp1_arc_center.retain(|c| c.line != r);
        self.coincident_lp2_arc_center.retain(|c| c.line != r);
        self.coincident_lp1_arc_start.retain(|c| c.line != r);
        self.coincident_lp2_arc_start.retain(|c| c.line != r);
        self.coincident_lp1_arc_end.retain(|c| c.line != r);
        self.coincident_lp2_arc_end.retain(|c| c.line != r);
        self.line_p1_on_arc.retain(|c| c.line != r);
        self.line_p2_on_arc.retain(|c| c.line != r);
        self.distance_ll11.retain(|c| c.a != r && c.b != r);
        self.distance_ll12.retain(|c| c.a != r && c.b != r);
        self.distance_ll21.retain(|c| c.a != r && c.b != r);
        self.distance_ll22.retain(|c| c.a != r && c.b != r);
        self.distance_lp1.retain(|c| c.line != r);
        self.distance_lp2.retain(|c| c.line != r);
        self.distance_arc_center_l1.retain(|c| c.line != r);
        self.distance_arc_center_l2.retain(|c| c.line != r);
        self.distance_arc_start_l1.retain(|c| c.line != r);
        self.distance_arc_start_l2.retain(|c| c.line != r);
        self.distance_arc_end_l1.retain(|c| c.line != r);
        self.distance_arc_end_l2.retain(|c| c.line != r);
        self.axis_distance_ll11.retain(|c| c.a != r && c.b != r);
        self.axis_distance_ll12.retain(|c| c.a != r && c.b != r);
        self.axis_distance_ll21.retain(|c| c.a != r && c.b != r);
        self.axis_distance_ll22.retain(|c| c.a != r && c.b != r);
        self.axis_distance_lp1.retain(|c| c.line != r);
        self.axis_distance_lp2.retain(|c| c.line != r);
        self.axis_distance_arc_center_l1.retain(|c| c.line != r);
        self.axis_distance_arc_center_l2.retain(|c| c.line != r);
        self.axis_distance_arc_start_l1.retain(|c| c.line != r);
        self.axis_distance_arc_start_l2.retain(|c| c.line != r);
        self.axis_distance_arc_end_l1.retain(|c| c.line != r);
        self.axis_distance_arc_end_l2.retain(|c| c.line != r);
        self.cleanup_helper_points();
    }

    /// Remove an arc and all constraints referencing it.
    pub fn delete_arc(&mut self, r: Ref<Arc>) {
        self.dimensions.retain(|d| !d.kind.references_arc(r));
        self.arcs.remove(r);
        self.point_on_arc.retain(|c| c.arc != r);
        self.line_p1_on_arc.retain(|c| c.arc != r);
        self.line_p2_on_arc.retain(|c| c.arc != r);
        self.tangent_la.retain(|c| c.arc != r);
        self.concentric.retain(|c| c.a != r && c.b != r);
        self.equal_radius.retain(|c| c.a != r && c.b != r);
        self.tangent_aa.retain(|c| c.a != r && c.b != r);
        self.symmetry_aa.retain(|c| c.a != r && c.c != r);
        self.midpoint_arc_start.retain(|c| c.arc != r);
        self.midpoint_arc_end.retain(|c| c.arc != r);
        self.midpoint_arc_point.retain(|c| c.arc != r);
        self.midpoint_lp1_arc.retain(|c| c.arc != r);
        self.midpoint_lp2_arc.retain(|c| c.arc != r);
        self.midpoint_arc_start_arc.retain(|c| c.a != r && c.b != r);
        self.midpoint_arc_end_arc.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_center.retain(|c| c.arc != r);
        self.coincident_arc_start.retain(|c| c.arc != r);
        self.coincident_arc_end.retain(|c| c.arc != r);
        self.coincident_lp1_arc_center.retain(|c| c.arc != r);
        self.coincident_lp2_arc_center.retain(|c| c.arc != r);
        self.coincident_lp1_arc_start.retain(|c| c.arc != r);
        self.coincident_lp2_arc_start.retain(|c| c.arc != r);
        self.coincident_lp1_arc_end.retain(|c| c.arc != r);
        self.coincident_lp2_arc_end.retain(|c| c.arc != r);
        self.coincident_arc_center_start.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_center_end.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_start_center.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_end_center.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_start_start.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_start_end.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_end_start.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_end_end.retain(|c| c.a != r && c.b != r);
        self.distance_arc_center_p.retain(|c| c.arc != r);
        self.distance_arc_start_p.retain(|c| c.arc != r);
        self.distance_arc_end_p.retain(|c| c.arc != r);
        self.distance_arc_center_l.retain(|c| c.arc != r);
        self.distance_arc_start_l.retain(|c| c.arc != r);
        self.distance_arc_end_l.retain(|c| c.arc != r);
        self.distance_arc_center_l1.retain(|c| c.arc != r);
        self.distance_arc_center_l2.retain(|c| c.arc != r);
        self.distance_arc_start_l1.retain(|c| c.arc != r);
        self.distance_arc_start_l2.retain(|c| c.arc != r);
        self.distance_arc_end_l1.retain(|c| c.arc != r);
        self.distance_arc_end_l2.retain(|c| c.arc != r);
        self.distance_aa_ce_ce.retain(|c| c.a != r && c.b != r);
        self.distance_aa_ce_s.retain(|c| c.a != r && c.b != r);
        self.distance_aa_ce_e.retain(|c| c.a != r && c.b != r);
        self.distance_aa_s_ce.retain(|c| c.a != r && c.b != r);
        self.distance_aa_s_s.retain(|c| c.a != r && c.b != r);
        self.distance_aa_s_e.retain(|c| c.a != r && c.b != r);
        self.distance_aa_e_ce.retain(|c| c.a != r && c.b != r);
        self.distance_aa_e_s.retain(|c| c.a != r && c.b != r);
        self.distance_aa_e_e.retain(|c| c.a != r && c.b != r);
        self.distance_concentric.retain(|c| c.a != r && c.b != r);
        self.axis_distance_arc_center_p.retain(|c| c.arc != r);
        self.axis_distance_arc_start_p.retain(|c| c.arc != r);
        self.axis_distance_arc_end_p.retain(|c| c.arc != r);
        self.axis_distance_arc_center_l1.retain(|c| c.arc != r);
        self.axis_distance_arc_center_l2.retain(|c| c.arc != r);
        self.axis_distance_arc_start_l1.retain(|c| c.arc != r);
        self.axis_distance_arc_start_l2.retain(|c| c.arc != r);
        self.axis_distance_arc_end_l1.retain(|c| c.arc != r);
        self.axis_distance_arc_end_l2.retain(|c| c.arc != r);
        self.axis_distance_aa_ce_ce.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_ce_s.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_ce_e.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_s_ce.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_s_s.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_s_e.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_e_ce.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_e_s.retain(|c| c.a != r && c.b != r);
        self.axis_distance_aa_e_e.retain(|c| c.a != r && c.b != r);
        self.cleanup_helper_points();
    }

    /// Remove helper points that are no longer needed.
    /// A helper is removed if it lost its bridge constraint (semantic origin
    /// gone) or has no purpose constraint. Cascades until stable.
    pub fn cleanup_helper_points(&mut self) {
        loop {
            // Find which helpers have a bridge (know what they represent)
            let mut has_bridge: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for c in &self.coincident_lp1 { if self.points.get(c.point).is_some_and(|p| p.helper) { has_bridge.insert(c.point.index()); } }
            for c in &self.coincident_lp2 { if self.points.get(c.point).is_some_and(|p| p.helper) { has_bridge.insert(c.point.index()); } }
            for c in &self.coincident_arc_center { if self.points.get(c.point).is_some_and(|p| p.helper) { has_bridge.insert(c.point.index()); } }
            for c in &self.coincident_arc_start { if self.points.get(c.point).is_some_and(|p| p.helper) { has_bridge.insert(c.point.index()); } }
            for c in &self.coincident_arc_end { if self.points.get(c.point).is_some_and(|p| p.helper) { has_bridge.insert(c.point.index()); } }
            for c in &self.coincident_pp {
                if self.points.get(c.a).is_some_and(|p| p.helper) { has_bridge.insert(c.a.index()); }
                if self.points.get(c.b).is_some_and(|p| p.helper) { has_bridge.insert(c.b.index()); }
            }

            // Find which helpers have a purpose constraint
            let mut has_purpose: std::collections::HashSet<u32> = std::collections::HashSet::new();
            let mut add_pt = |r: Ref<Point>| { has_purpose.insert(r.index()); };
            for c in &self.distance_pp { add_pt(c.a); add_pt(c.b); }
            for c in &self.hdistance_pp { add_pt(c.a); add_pt(c.b); }
            for c in &self.vdistance_pp { add_pt(c.a); add_pt(c.b); }
            for c in &self.point_on_line { add_pt(c.point); }
            for c in &self.midpoint { add_pt(c.point); }
            for c in &self.point_on_arc { add_pt(c.point); }
            for c in &self.distance_pl { add_pt(c.point); }
            for c in &self.distance_lp1 { add_pt(c.point); }
            for c in &self.distance_lp2 { add_pt(c.point); }
            for c in &self.symmetry_pp { add_pt(c.a); add_pt(c.c); }

            // Remove helpers that lost their bridge OR have no purpose
            let to_remove: std::vec::Vec<Ref<Point>> = self.points.refs()
                .filter(|r| self.points[*r].helper
                    && (!has_bridge.contains(&r.index()) || !has_purpose.contains(&r.index())))
                .collect();
            if to_remove.is_empty() { break; }

            for r in &to_remove {
                self.coincident_pp.retain(|c| c.a != *r && c.b != *r);
                self.coincident_lp1.retain(|c| c.point != *r);
                self.coincident_lp2.retain(|c| c.point != *r);
                self.coincident_arc_center.retain(|c| c.point != *r);
                self.coincident_arc_start.retain(|c| c.point != *r);
                self.coincident_arc_end.retain(|c| c.point != *r);
                self.symmetry_pp.retain(|c| c.a != *r && c.c != *r);
                self.distance_pp.retain(|c| c.a != *r && c.b != *r);
                self.hdistance_pp.retain(|c| c.a != *r && c.b != *r);
                self.vdistance_pp.retain(|c| c.a != *r && c.b != *r);
                self.point_on_line.retain(|c| c.point != *r);
                self.midpoint.retain(|c| c.point != *r);
                self.point_on_arc.retain(|c| c.point != *r);
                self.distance_pl.retain(|c| c.point != *r);
                self.distance_lp1.retain(|c| c.point != *r);
                self.distance_lp2.retain(|c| c.point != *r);
            }
            for r in to_remove { self.points.remove(r); }
        }
    }

    /// Remove duplicate constraints from all collections. Prints a warning if any are found.
    pub fn dedup_constraints(&mut self) {
        let mut total_removed = 0usize;
        macro_rules! dedup_ab {
            ($coll:expr, $name:expr, $container_a:expr, $container_b:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.a.index(), c.b.index())) {
                        let na = $container_a.get(c.a).map(|e| e.name.as_str()).unwrap_or("?");
                        let nb = $container_b.get(c.b).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: a={}, b={}", $name, na, nb);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.a.index(), c.b.index())));
            };
        }
        macro_rules! dedup_lp {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.line.index(), c.point.index())) {
                        let nl = self.lines.get(c.line).map(|e| e.name.as_str()).unwrap_or("?");
                        let np = self.points.get(c.point).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: line={}, point={}", $name, nl, np);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.line.index(), c.point.index())));
            };
        }
        macro_rules! dedup_la {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.line.index(), c.arc.index())) {
                        let nl = self.lines.get(c.line).map(|e| e.name.as_str()).unwrap_or("?");
                        let na = self.arcs.get(c.arc).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: line={}, arc={}", $name, nl, na);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.line.index(), c.arc.index())));
            };
        }
        macro_rules! dedup_pa {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.point.index(), c.arc.index())) {
                        let np = self.points.get(c.point).map(|e| e.name.as_str()).unwrap_or("?");
                        let na = self.arcs.get(c.arc).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: point={}, arc={}", $name, np, na);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.point.index(), c.arc.index())));
            };
        }
        macro_rules! dedup_pl {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.point.index(), c.line.index())) {
                        let np = self.points.get(c.point).map(|e| e.name.as_str()).unwrap_or("?");
                        let nl = self.lines.get(c.line).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: point={}, line={}", $name, np, nl);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.point.index(), c.line.index())));
            };
        }
        dedup_ab!(self.coincident_pp, "coincident_pp", self.points, self.points);
        // PP is symmetric: (a,b) == (b,a)
        {
            let before = self.coincident_pp.len();
            let mut seen = std::collections::HashSet::new();
            self.coincident_pp.retain(|c| {
                let (a, b) = (c.a.index().min(c.b.index()), c.a.index().max(c.b.index()));
                seen.insert((a, b))
            });
            let removed = before - self.coincident_pp.len();
            if removed > 0 { eprintln!("BUG: removed {} cross-duplicate coincident_pp constraints", removed); total_removed += removed; }
        }
        dedup_lp!(self.coincident_lp1, "coincident_lp1");
        dedup_lp!(self.coincident_lp2, "coincident_lp2");
        dedup_ab!(self.coincident_ll11, "coincident_ll11", self.lines, self.lines);
        dedup_ab!(self.coincident_ll12, "coincident_ll12", self.lines, self.lines);
        dedup_ab!(self.coincident_ll21, "coincident_ll21", self.lines, self.lines);
        dedup_ab!(self.coincident_ll22, "coincident_ll22", self.lines, self.lines);
        // Cross-Vec dedup for LL: ll11(a,b)==ll11(b,a), ll22(a,b)==ll22(b,a), ll12(a,b)==ll21(b,a)
        {
            let mut seen = std::collections::HashSet::new();
            // Normalize: represent each endpoint pair as (min_id, max_id) where id encodes line+endpoint
            let ep_id = |line: u32, is_p2: bool| -> u64 { (line as u64) << 1 | (is_p2 as u64) };
            let mut add = |line_a: u32, p2_a: bool, line_b: u32, p2_b: bool| -> bool {
                let a = ep_id(line_a, p2_a);
                let b = ep_id(line_b, p2_b);
                let key = (a.min(b), a.max(b));
                seen.insert(key)
            };
            let before = self.coincident_ll11.len() + self.coincident_ll12.len()
                + self.coincident_ll21.len() + self.coincident_ll22.len();
            self.coincident_ll11.retain(|c| add(c.a.index(), false, c.b.index(), false));
            self.coincident_ll12.retain(|c| add(c.a.index(), false, c.b.index(), true));
            self.coincident_ll21.retain(|c| add(c.a.index(), true, c.b.index(), false));
            self.coincident_ll22.retain(|c| add(c.a.index(), true, c.b.index(), true));
            let after = self.coincident_ll11.len() + self.coincident_ll12.len()
                + self.coincident_ll21.len() + self.coincident_ll22.len();
            let removed = before - after;
            if removed > 0 { eprintln!("BUG: removed {} cross-duplicate LL coincident constraints", removed); total_removed += removed; }
        }
        dedup_ab!(self.distance_pp, "distance_pp", self.points, self.points);
        dedup_ab!(self.hdistance_pp, "hdistance_pp", self.points, self.points);
        dedup_ab!(self.vdistance_pp, "vdistance_pp", self.points, self.points);
        dedup_pl!(self.point_on_line, "point_on_line");
        dedup_pl!(self.midpoint, "midpoint");
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_lp1.retain(|c| seen.insert((c.line.index(), c.target.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_lp2.retain(|c| seen.insert((c.line.index(), c.target.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_arc_start.retain(|c| seen.insert((c.arc.index(), c.line.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_arc_end.retain(|c| seen.insert((c.arc.index(), c.line.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_arc_point.retain(|c| seen.insert((c.point.index(), c.arc.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_lp1_arc.retain(|c| seen.insert((c.line.index(), c.arc.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_lp2_arc.retain(|c| seen.insert((c.line.index(), c.arc.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_arc_start_arc.retain(|c| seen.insert((c.a.index(), c.b.index())));
        }
        {
            let mut seen = std::collections::HashSet::new();
            self.midpoint_arc_end_arc.retain(|c| seen.insert((c.a.index(), c.b.index())));
        }
        dedup_pa!(self.point_on_arc, "point_on_arc");
        dedup_ab!(self.parallel, "parallel", self.lines, self.lines);
        dedup_la!(self.arc_line_parallel, "arc_line_parallel");
        dedup_ab!(self.arc_arc_parallel, "arc_arc_parallel", self.arcs, self.arcs);
        dedup_ab!(self.perpendicular, "perpendicular", self.lines, self.lines);
        dedup_ab!(self.collinear, "collinear", self.lines, self.lines);
        {
            let mut seen = std::collections::HashSet::new();
            self.symmetry_ll.retain(|c| seen.insert((c.a.index(), c.b.index(), c.c.index())));
        }
        dedup_ab!(self.equal_length, "equal_length", self.lines, self.lines);
        dedup_ab!(self.angle, "angle", self.lines, self.lines);
        dedup_la!(self.tangent_la, "tangent_la");
        dedup_la!(self.line_p1_on_arc, "line_p1_on_arc");
        dedup_la!(self.line_p2_on_arc, "line_p2_on_arc");
        dedup_ab!(self.concentric, "concentric", self.arcs, self.arcs);
        dedup_ab!(self.equal_radius, "equal_radius", self.arcs, self.arcs);
        dedup_ab!(self.tangent_aa, "tangent_aa", self.arcs, self.arcs);
        dedup_pl!(self.distance_pl, "distance_pl");
        dedup_ab!(self.line_p1_on_line, "line_p1_on_line", self.lines, self.lines);
        dedup_ab!(self.line_p2_on_line, "line_p2_on_line", self.lines, self.lines);
        dedup_pa!(self.coincident_arc_center, "coincident_arc_center");
        dedup_pa!(self.coincident_arc_start, "coincident_arc_start");
        dedup_pa!(self.coincident_arc_end, "coincident_arc_end");
        dedup_la!(self.coincident_lp1_arc_center, "coincident_lp1_arc_center");
        dedup_la!(self.coincident_lp2_arc_center, "coincident_lp2_arc_center");
        dedup_la!(self.coincident_lp1_arc_start, "coincident_lp1_arc_start");
        dedup_la!(self.coincident_lp2_arc_start, "coincident_lp2_arc_start");
        dedup_la!(self.coincident_lp1_arc_end, "coincident_lp1_arc_end");
        dedup_la!(self.coincident_lp2_arc_end, "coincident_lp2_arc_end");
        dedup_ab!(self.coincident_arc_center_start, "coincident_arc_center_start", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_center_end, "coincident_arc_center_end", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_start_center, "coincident_arc_start_center", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_end_center, "coincident_arc_end_center", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_start_start, "coincident_arc_start_start", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_start_end, "coincident_arc_start_end", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_end_start, "coincident_arc_end_start", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_end_end, "coincident_arc_end_end", self.arcs, self.arcs);
        let _ = total_removed;
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
        let mut merged = std::collections::HashMap::<u32, Ref<Point>>::new(); // old -> canonical
        for i in 0..helper_refs.len() {
            let ri = helper_refs[i];
            if merged.contains_key(&ri.index()) { continue; }
            let pi = self.points[ri].pos.value;
            for j in (i+1)..helper_refs.len() {
                let rj = helper_refs[j];
                if merged.contains_key(&rj.index()) { continue; }
                let pj = self.points[rj].pos.value;
                if (pi.x - pj.x).abs() < 1e-9 && (pi.y - pj.y).abs() < 1e-9 {
                    merged.insert(rj.index(), ri);
                    eprintln!("INFO: merging duplicate helper point {} into {}", rj.index(), ri.index());
                }
            }
        }
        if !merged.is_empty() {
            // Rewrite all point refs in constraints
            let remap = |r: &mut Ref<Point>| {
                if let Some(canonical) = merged.get(&r.index()) { *r = *canonical; }
            };
            for c in &mut self.coincident_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.coincident_lp1 { remap(&mut c.point); }
            for c in &mut self.coincident_lp2 { remap(&mut c.point); }
            for c in &mut self.distance_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.hdistance_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.vdistance_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.point_on_line { remap(&mut c.point); }
            for c in &mut self.midpoint { remap(&mut c.point); }
            for c in &mut self.point_on_arc { remap(&mut c.point); }
            for c in &mut self.distance_pl { remap(&mut c.point); }
            for c in &mut self.coincident_arc_center { remap(&mut c.point); }
            for c in &mut self.coincident_arc_start { remap(&mut c.point); }
            for c in &mut self.coincident_arc_end { remap(&mut c.point); }
            for c in &mut self.distance_arc_center_p { remap(&mut c.point); }
            for c in &mut self.distance_arc_start_p { remap(&mut c.point); }
            for c in &mut self.distance_arc_end_p { remap(&mut c.point); }
            for c in &mut self.axis_distance_lp1 { remap(&mut c.point); }
            for c in &mut self.axis_distance_lp2 { remap(&mut c.point); }
            for c in &mut self.axis_distance_arc_center_p { remap(&mut c.point); }
            for c in &mut self.axis_distance_arc_start_p { remap(&mut c.point); }
            for c in &mut self.axis_distance_arc_end_p { remap(&mut c.point); }
            for c in &mut self.symmetry_pp { remap(&mut c.a); remap(&mut c.c); }
            for c in &mut self.distance_lp1 { remap(&mut c.point); }
            for c in &mut self.distance_lp2 { remap(&mut c.point); }
            // Remove merged points
            for old in merged.keys() { self.points.remove(Ref::new(*old)); }
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

impl arael::model::ExtendedModel for Sketch {
    fn extended_deserialize64(&mut self) {
        // After solver writes back optimized values, sync radius_b.value for non-ellipses
        let refs: Vec<_> = self.arcs.refs().collect();
        for r in refs {
            if !self.arcs[r].is_ellipse {
                self.arcs[r].radius_b.value = self.arcs[r].radius.value;
            }
        }
    }

    fn extended_update64(&mut self, _params: &[f64]) {
        // For non-ellipse arcs, keep radius_b work value in sync with radius
        let refs: Vec<_> = self.arcs.refs().collect();
        for r in refs {
            if !self.arcs[r].is_ellipse {
                let rv = self.arcs[r].radius.work();
                *self.arcs[r].radius_b.work_mut() = rv;
            }
        }
    }

    fn extended_cost64(&self, params: &[f64]) -> f64 {
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

    fn extended_compute64(&mut self, params: &[f64], grad: &mut [f64]) {
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

    fn extended_jacobian64(&mut self, params: &[f64], rows: &mut std::vec::Vec<arael::model::JacobianRow<f64>>, cid: &mut u32) {
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
    pub fn constraint_nid_cid_pairs(&self) -> std::vec::Vec<(u32, u32)> {
        let mut out = std::vec::Vec::new();
        macro_rules! collect {
            ($($field:ident),* $(,)?) => {
                $(
                    for c in &self.$field {
                        if c.nid != 0 {
                            out.push((c.nid, c.cid));
                        }
                    }
                )*
            };
        }
        collect!(
            coincident_pp,
            coincident_lp1, coincident_lp2,
            coincident_ll11, coincident_ll12, coincident_ll21, coincident_ll22,
            distance_pp, hdistance_pp, vdistance_pp,
            point_on_line,
            midpoint, midpoint_lp1, midpoint_lp2,
            midpoint_arc_start, midpoint_arc_end,
            midpoint_arc_point,
            midpoint_lp1_arc, midpoint_lp2_arc,
            midpoint_arc_start_arc, midpoint_arc_end_arc,
            point_on_arc,
            parallel, perpendicular, collinear,
            equal_length, angle,
            tangent_la, concentric, equal_radius, tangent_aa,
            symmetry_ll, symmetry_pp, symmetry_aa,
            distance_pl, distance_lp1l, distance_lp2l,
            distance_arc_center_l, distance_arc_start_l, distance_arc_end_l,
            line_p1_on_line, line_p2_on_line,
            coincident_arc_center, coincident_arc_start, coincident_arc_end,
            coincident_lp1_arc_center, coincident_lp2_arc_center,
            coincident_lp1_arc_start, coincident_lp2_arc_start,
            coincident_lp1_arc_end, coincident_lp2_arc_end,
            coincident_arc_center_start, coincident_arc_center_end,
            coincident_arc_start_center, coincident_arc_end_center,
            coincident_arc_start_start, coincident_arc_start_end,
            coincident_arc_end_start, coincident_arc_end_end,
            line_p1_on_arc, line_p2_on_arc,
            distance_ll11, distance_ll12, distance_ll21, distance_ll22,
            distance_lp1, distance_lp2,
            distance_arc_center_p, distance_arc_start_p, distance_arc_end_p,
            distance_arc_center_l1, distance_arc_center_l2,
            distance_arc_start_l1, distance_arc_start_l2,
            distance_arc_end_l1, distance_arc_end_l2,
            distance_aa_ce_ce, distance_aa_ce_s, distance_aa_ce_e,
            distance_aa_s_ce, distance_aa_s_s, distance_aa_s_e,
            distance_aa_e_ce, distance_aa_e_s, distance_aa_e_e,
            distance_concentric,
            axis_distance_ll11, axis_distance_ll12, axis_distance_ll21, axis_distance_ll22,
            axis_distance_lp1, axis_distance_lp2,
            axis_distance_arc_center_p, axis_distance_arc_start_p, axis_distance_arc_end_p,
            axis_distance_arc_center_l1, axis_distance_arc_center_l2,
            axis_distance_arc_start_l1, axis_distance_arc_start_l2,
            axis_distance_arc_end_l1, axis_distance_arc_end_l2,
            axis_distance_aa_ce_ce, axis_distance_aa_ce_s, axis_distance_aa_ce_e,
            axis_distance_aa_s_ce, axis_distance_aa_s_s, axis_distance_aa_s_e,
            axis_distance_aa_e_ce, axis_distance_aa_e_s, axis_distance_aa_e_e,
        );
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
        // Helper to get entity names
        let pn = |r: Ref<Point>| self.points.get(r).map(|p| p.name.as_str()).unwrap_or("?").to_string();
        let ln = |r: Ref<Line>| self.lines.get(r).map(|l| l.name.as_str()).unwrap_or("?").to_string();
        let an = |r: Ref<Arc>| self.arcs.get(r).map(|a| a.name.as_str()).unwrap_or("?").to_string();
        for c in &self.coincident_pp { m.insert(c.cid, format!("coinc_pp:{},{}", pn(c.a), pn(c.b))); }
        for c in &self.coincident_lp1 { m.insert(c.cid, format!("coinc_lp1:{},{}", ln(c.line), pn(c.point))); }
        for c in &self.coincident_lp2 { m.insert(c.cid, format!("coinc_lp2:{},{}", ln(c.line), pn(c.point))); }
        for c in &self.coincident_ll11 { m.insert(c.cid, format!("coinc_ll11:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.coincident_ll12 { m.insert(c.cid, format!("coinc_ll12:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.coincident_ll21 { m.insert(c.cid, format!("coinc_ll21:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.coincident_ll22 { m.insert(c.cid, format!("coinc_ll22:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.distance_pp { m.insert(c.cid, format!("dist_pp:{},{}", pn(c.a), pn(c.b))); }
        for c in &self.hdistance_pp { m.insert(c.cid, format!("hdist_pp:{},{}", pn(c.a), pn(c.b))); }
        for c in &self.vdistance_pp { m.insert(c.cid, format!("vdist_pp:{},{}", pn(c.a), pn(c.b))); }
        for c in &self.point_on_line { m.insert(c.cid, format!("on_line:{},{}", pn(c.point), ln(c.line))); }
        for c in &self.midpoint { m.insert(c.cid, format!("midpoint:{},{}", pn(c.point), ln(c.line))); }
        for c in &self.point_on_arc { m.insert(c.cid, format!("on_arc:{},{}", pn(c.point), an(c.arc))); }
        for c in &self.parallel { m.insert(c.cid, format!("parallel:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.arc_line_parallel { m.insert(c.cid, format!("parallel:{},{}", an(c.arc), ln(c.line))); }
        for c in &self.arc_arc_parallel { m.insert(c.cid, format!("parallel:{},{}", an(c.a), an(c.b))); }
        for c in &self.perpendicular { m.insert(c.cid, format!("perp:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.collinear { m.insert(c.cid, format!("collinear:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.equal_length { m.insert(c.cid, format!("eq_len:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.angle { m.insert(c.cid, format!("angle:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.tangent_la { m.insert(c.cid, format!("tangent_la:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_lp2_arc_start { m.insert(c.cid, format!("coinc_lp2_as:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.tangent_aa { m.insert(c.cid, format!("tangent_aa:{},{}", an(c.a), an(c.b))); }
        for c in &self.concentric { m.insert(c.cid, format!("concentric:{},{}", an(c.a), an(c.b))); }
        for c in &self.equal_radius { m.insert(c.cid, format!("eq_radius:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_center { m.insert(c.cid, format!("coinc_arc_center:{},{}", pn(c.point), an(c.arc))); }
        for c in &self.coincident_arc_start { m.insert(c.cid, format!("coinc_arc_start:{},{}", pn(c.point), an(c.arc))); }
        for c in &self.coincident_arc_end { m.insert(c.cid, format!("coinc_arc_end:{},{}", pn(c.point), an(c.arc))); }
        for c in &self.coincident_lp1_arc_center { m.insert(c.cid, format!("coinc_lp1_ac:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_lp2_arc_center { m.insert(c.cid, format!("coinc_lp2_ac:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_lp1_arc_start { m.insert(c.cid, format!("coinc_lp1_as:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_lp2_arc_start { m.insert(c.cid, format!("coinc_lp2_as:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_lp1_arc_end { m.insert(c.cid, format!("coinc_lp1_ae:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_lp2_arc_end { m.insert(c.cid, format!("coinc_lp2_ae:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.coincident_arc_center_start { m.insert(c.cid, format!("coinc_ac_as:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_center_end { m.insert(c.cid, format!("coinc_ac_ae:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_start_center { m.insert(c.cid, format!("coinc_as_ac:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_end_center { m.insert(c.cid, format!("coinc_ae_ac:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_start_start { m.insert(c.cid, format!("coinc_as_as:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_start_end { m.insert(c.cid, format!("coinc_as_ae:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_end_start { m.insert(c.cid, format!("coinc_ae_as:{},{}", an(c.a), an(c.b))); }
        for c in &self.coincident_arc_end_end { m.insert(c.cid, format!("coinc_ae_ae:{},{}", an(c.a), an(c.b))); }
        for c in &self.line_p1_on_line { m.insert(c.cid, format!("lp1_on_l:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.line_p2_on_line { m.insert(c.cid, format!("lp2_on_l:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.line_p1_on_arc { m.insert(c.cid, format!("lp1_on_a:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.line_p2_on_arc { m.insert(c.cid, format!("lp2_on_a:{},{}", ln(c.line), an(c.arc))); }
        for c in &self.distance_pl { m.insert(c.cid, format!("dist_pl:{},{}", pn(c.point), ln(c.line))); }
        for c in &self.distance_lp1l { m.insert(c.cid, format!("dist_lp1l:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.distance_lp2l { m.insert(c.cid, format!("dist_lp2l:{},{}", ln(c.a), ln(c.b))); }
        for c in &self.distance_arc_center_l { m.insert(c.cid, format!("dist_ac_l:{},{}", an(c.arc), ln(c.line))); }
        for c in &self.distance_arc_start_l { m.insert(c.cid, format!("dist_as_l:{},{}", an(c.arc), ln(c.line))); }
        for c in &self.distance_arc_end_l { m.insert(c.cid, format!("dist_ae_l:{},{}", an(c.arc), ln(c.line))); }
        for c in &self.distance_concentric { m.insert(c.cid, format!("dist_concentric:{},{}", an(c.a), an(c.b))); }
        for c in &self.symmetry_ll { m.insert(c.cid, format!("sym_ll:{},{},{}", ln(c.a), ln(c.b), ln(c.c))); }
        for c in &self.symmetry_pp { m.insert(c.cid, format!("sym_pp:{},{},{}", pn(c.a), ln(c.line), pn(c.c))); }
        for c in &self.symmetry_aa { m.insert(c.cid, format!("sym_aa:{},{},{}", an(c.a), ln(c.line), an(c.c))); }
        for ec in &self.expr_constraints { m.insert(ec.cid, format!("expr:{}", ec.description)); }
        for (i, ec) in self.expr_constraints.iter().enumerate() {
            // expr_constraints are assigned CIDs starting after all macro-generated ones
            // but we don't have easy access to that base — approximate with description
            let _ = i;
            let label = format!("expr:{}", ec.description);
            // We don't know the CID here — it would need to be tracked. Skip for now.
            let _ = label;
        }
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
            self.serialize64(&mut tmp);
        }
        let mut bag = SymbolBag::build(self);

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
            self.serialize64(&mut tmp);
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
    pub fn list_constraints(&self) -> Vec<String> {
        let mut out = Vec::new();
        // Entity-level flags
        for r in self.lines.refs() {
            let l = &self.lines[r];
            if l.constraints.horizontal { out.push(format!("{}: horizontal {}", format_flag_name(&l.name, 'H'), l.name)); }
            if l.constraints.vertical { out.push(format!("{}: vertical {}", format_flag_name(&l.name, 'V'), l.name)); }
            if l.constraints.has_length { out.push(format!("length {} = {}", l.name, l.constraints.length)); }
            if l.constraints.has_angle { out.push(format!("xangle {} = {:.4}", l.name, l.constraints.target_angle.to_degrees())); }
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
            if a.constraints.has_target_radius { out.push(format!("radius {} = {}", a.name, a.constraints.target_radius)); }
            if a.constraints.has_target_sweep { out.push(format!("sweep {} = {:.2} deg", a.name, a.constraints.target_sweep.to_degrees())); }
            if !a.center.optimize { out.push(format!("lock {}.center", a.name)); }
        }
        // Constraint vectors (use macros to reduce repetition)
        macro_rules! list_cross {
            ($vec:expr, $name:literal, $fa:ident, $fb:ident) => {
                for c in &$vec {
                    let na = &self.lines[c.$fa].name;
                    let nb = &self.lines[c.$fb].name;
                    out.push(format!("C{}: {} {} {}", c.nid, $name, na, nb));
                }
            };
        }
        list_cross!(self.parallel, "parallel", a, b);
        for c in &self.arc_line_parallel {
            let na = &self.arcs[c.arc].name;
            let nb = &self.lines[c.line].name;
            out.push(format!("C{}: parallel {} {}", c.nid, na, nb));
        }
        for c in &self.arc_arc_parallel {
            let na = &self.arcs[c.a].name;
            let nb = &self.arcs[c.b].name;
            out.push(format!("C{}: parallel {} {}", c.nid, na, nb));
        }
        list_cross!(self.perpendicular, "perpendicular", a, b);
        list_cross!(self.collinear, "collinear", a, b);
        list_cross!(self.equal_length, "equal", a, b);
        // Coincident: suppress bridge constraints (helper point bridges)
        for c in &self.coincident_pp {
            if !self.points[c.a].helper && !self.points[c.b].helper {
                out.push(format!("C{}: coincident {} {}", c.nid, self.points[c.a].name, self.points[c.b].name));
            }
        }
        for c in &self.coincident_ll11 { out.push(format!("C{}: coincident {}.p1 {}.p1", c.nid, self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_ll12 { out.push(format!("C{}: coincident {}.p1 {}.p2", c.nid, self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_ll21 { out.push(format!("C{}: coincident {}.p2 {}.p1", c.nid, self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_ll22 { out.push(format!("C{}: coincident {}.p2 {}.p2", c.nid, self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_lp1 {
            if !self.points[c.point].helper {
                out.push(format!("C{}: coincident {}.p1 {}", c.nid, self.lines[c.line].name, self.points[c.point].name));
            }
        }
        for c in &self.coincident_lp2 {
            if !self.points[c.point].helper {
                out.push(format!("C{}: coincident {}.p2 {}", c.nid, self.lines[c.line].name, self.points[c.point].name));
            }
        }
        // Point-Arc coincident: suppress helper bridges
        for c in &self.coincident_arc_center {
            if !self.points[c.point].helper {
                out.push(format!("C{}: coincident {} {}.center", c.nid, self.points[c.point].name, self.arcs[c.arc].name));
            }
        }
        for c in &self.coincident_arc_start {
            if !self.points[c.point].helper {
                out.push(format!("C{}: coincident {} {}.start", c.nid, self.points[c.point].name, self.arcs[c.arc].name));
            }
        }
        for c in &self.coincident_arc_end {
            if !self.points[c.point].helper {
                out.push(format!("C{}: coincident {} {}.end", c.nid, self.points[c.point].name, self.arcs[c.arc].name));
            }
        }
        // Line-Arc coincident
        for c in &self.coincident_lp1_arc_center { out.push(format!("C{}: coincident {}.p1 {}.center", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.coincident_lp2_arc_center { out.push(format!("C{}: coincident {}.p2 {}.center", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.coincident_lp1_arc_start { out.push(format!("C{}: coincident {}.p1 {}.start", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.coincident_lp2_arc_start { out.push(format!("C{}: coincident {}.p2 {}.start", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.coincident_lp1_arc_end { out.push(format!("C{}: coincident {}.p1 {}.end", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.coincident_lp2_arc_end { out.push(format!("C{}: coincident {}.p2 {}.end", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        // Arc-Arc coincident
        for c in &self.coincident_arc_center_start { out.push(format!("C{}: coincident {}.center {}.start", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_center_end { out.push(format!("C{}: coincident {}.center {}.end", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_start_center { out.push(format!("C{}: coincident {}.start {}.center", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_end_center { out.push(format!("C{}: coincident {}.end {}.center", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_start_start { out.push(format!("C{}: coincident {}.start {}.start", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_start_end { out.push(format!("C{}: coincident {}.start {}.end", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_end_start { out.push(format!("C{}: coincident {}.end {}.start", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.coincident_arc_end_end { out.push(format!("C{}: coincident {}.end {}.end", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        // Line endpoint on line/arc
        for c in &self.line_p1_on_line { out.push(format!("C{}: point_on {}.p1 {}", c.nid, self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.line_p2_on_line { out.push(format!("C{}: point_on {}.p2 {}", c.nid, self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.line_p1_on_arc { out.push(format!("C{}: point_on {}.p1 {}", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.line_p2_on_arc { out.push(format!("C{}: point_on {}.p2 {}", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.angle { out.push(format!("C{}: angle {} {} = {:.1}deg", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.angle.to_degrees())); }
        for c in &self.tangent_la { out.push(format!("C{}: tangent {} {}", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.tangent_aa { out.push(format!("C{}: tangent {} {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.concentric { out.push(format!("C{}: concentric {} {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.equal_radius { out.push(format!("C{}: equal {} {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.symmetry_ll { out.push(format!("C{}: symmetry {} {} {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, self.lines[c.c].name)); }
        for c in &self.symmetry_pp { out.push(format!("C{}: symmetry {} {} {}", c.nid, self.point_display_name(c.a), self.lines[c.line].name, self.point_display_name(c.c))); }
        for c in &self.symmetry_aa { out.push(format!("C{}: symmetry {} {} {}", c.nid, self.arcs[c.a].name, self.lines[c.line].name, self.arcs[c.c].name)); }
        // Point-based constraints: use display names to resolve helpers
        for c in &self.point_on_line { out.push(format!("C{}: point_on {} {}", c.nid, self.point_display_name(c.point), self.lines[c.line].name)); }
        for c in &self.point_on_arc { out.push(format!("C{}: point_on {} {}", c.nid, self.point_display_name(c.point), self.arcs[c.arc].name)); }
        for c in &self.midpoint { out.push(format!("C{}: midpoint {} {}", c.nid, self.point_display_name(c.point), self.lines[c.line].name)); }
        for c in &self.midpoint_lp1 { out.push(format!("C{}: midpoint {}.p1 {}", c.nid, self.lines[c.line].name, self.lines[c.target].name)); }
        for c in &self.midpoint_lp2 { out.push(format!("C{}: midpoint {}.p2 {}", c.nid, self.lines[c.line].name, self.lines[c.target].name)); }
        for c in &self.midpoint_arc_start { out.push(format!("C{}: midpoint {}.start {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name)); }
        for c in &self.midpoint_arc_end { out.push(format!("C{}: midpoint {}.end {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name)); }
        for c in &self.midpoint_arc_point { out.push(format!("C{}: midpoint {} {}", c.nid, self.point_display_name(c.point), self.arcs[c.arc].name)); }
        for c in &self.midpoint_lp1_arc { out.push(format!("C{}: midpoint {}.p1 {}", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.midpoint_lp2_arc { out.push(format!("C{}: midpoint {}.p2 {}", c.nid, self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.midpoint_arc_start_arc { out.push(format!("C{}: midpoint {}.start {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.midpoint_arc_end_arc { out.push(format!("C{}: midpoint {}.end {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.distance_pp { out.push(format!("C{}: distance {} {} = {}", c.nid, self.point_display_name(c.a), self.point_display_name(c.b), c.distance.abs())); }
        for c in &self.hdistance_pp { out.push(format!("C{}: hdistance {} {} = {}", c.nid, self.point_display_name(c.a), self.point_display_name(c.b), c.distance.abs())); }
        for c in &self.vdistance_pp { out.push(format!("C{}: vdistance {} {} = {}", c.nid, self.point_display_name(c.a), self.point_display_name(c.b), c.distance.abs())); }
        for c in &self.distance_pl { out.push(format!("C{}: distance {} {} = {}", c.nid, self.point_display_name(c.point), self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_lp1l { out.push(format!("C{}: distance {}.p1 {} = {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.distance_lp2l { out.push(format!("C{}: distance {}.p2 {} = {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.distance_arc_center_l { out.push(format!("C{}: distance {}.center {} = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_start_l { out.push(format!("C{}: distance {}.start {} = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_end_l { out.push(format!("C{}: distance {}.end {} = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        // Line-endpoint distance constraints
        for c in &self.distance_ll11 { out.push(format!("C{}: distance {}.p1 {}.p1 = {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.distance_ll12 { out.push(format!("C{}: distance {}.p1 {}.p2 = {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.distance_ll21 { out.push(format!("C{}: distance {}.p2 {}.p1 = {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.distance_ll22 { out.push(format!("C{}: distance {}.p2 {}.p2 = {}", c.nid, self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.distance_lp1 { out.push(format!("C{}: distance {}.p1 {} = {}", c.nid, self.lines[c.line].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.distance_lp2 { out.push(format!("C{}: distance {}.p2 {} = {}", c.nid, self.lines[c.line].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.distance_arc_center_p { out.push(format!("C{}: distance {}.center {} = {}", c.nid, self.arcs[c.arc].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.distance_arc_start_p { out.push(format!("C{}: distance {}.start {} = {}", c.nid, self.arcs[c.arc].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.distance_arc_end_p { out.push(format!("C{}: distance {}.end {} = {}", c.nid, self.arcs[c.arc].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.distance_arc_center_l1 { out.push(format!("C{}: distance {}.center {}.p1 = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_center_l2 { out.push(format!("C{}: distance {}.center {}.p2 = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_start_l1 { out.push(format!("C{}: distance {}.start {}.p1 = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_start_l2 { out.push(format!("C{}: distance {}.start {}.p2 = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_end_l1 { out.push(format!("C{}: distance {}.end {}.p1 = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_arc_end_l2 { out.push(format!("C{}: distance {}.end {}.p2 = {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.distance_aa_ce_ce { out.push(format!("C{}: distance {}.center {}.center = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_ce_s { out.push(format!("C{}: distance {}.center {}.start = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_ce_e { out.push(format!("C{}: distance {}.center {}.end = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_s_ce { out.push(format!("C{}: distance {}.start {}.center = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_s_s { out.push(format!("C{}: distance {}.start {}.start = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_s_e { out.push(format!("C{}: distance {}.start {}.end = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_e_ce { out.push(format!("C{}: distance {}.end {}.center = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_e_s { out.push(format!("C{}: distance {}.end {}.start = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_aa_e_e { out.push(format!("C{}: distance {}.end {}.end = {}", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.distance_concentric {
            out.push(format!("C{}: distance {} {} = {} (concentric)", c.nid, self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs()));
        }
        // Midpoint variants
        for c in &self.midpoint_lp1 { out.push(format!("C{}: midpoint {}.p1 {}", c.nid, self.lines[c.target].name, self.lines[c.line].name)); }
        for c in &self.midpoint_lp2 { out.push(format!("C{}: midpoint {}.p2 {}", c.nid, self.lines[c.target].name, self.lines[c.line].name)); }
        for c in &self.midpoint_arc_start { out.push(format!("C{}: midpoint {}.start {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name)); }
        for c in &self.midpoint_arc_end { out.push(format!("C{}: midpoint {}.end {}", c.nid, self.arcs[c.arc].name, self.lines[c.line].name)); }
        // Axis distance constraints
        let axis_label = |h: bool| if h { "hdistance" } else { "vdistance" };
        for c in &self.axis_distance_ll11 { out.push(format!("C{}: {} {}.p1 {}.p1 = {}", c.nid, axis_label(c.horizontal), self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_ll12 { out.push(format!("C{}: {} {}.p1 {}.p2 = {}", c.nid, axis_label(c.horizontal), self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_ll21 { out.push(format!("C{}: {} {}.p2 {}.p1 = {}", c.nid, axis_label(c.horizontal), self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_ll22 { out.push(format!("C{}: {} {}.p2 {}.p2 = {}", c.nid, axis_label(c.horizontal), self.lines[c.a].name, self.lines[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_lp1 { out.push(format!("C{}: {} {}.p1 {} = {}", c.nid, axis_label(c.horizontal), self.lines[c.line].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.axis_distance_lp2 { out.push(format!("C{}: {} {}.p2 {} = {}", c.nid, axis_label(c.horizontal), self.lines[c.line].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.axis_distance_arc_center_p { out.push(format!("C{}: {} {}.center {} = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.axis_distance_arc_start_p { out.push(format!("C{}: {} {}.start {} = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.axis_distance_arc_end_p { out.push(format!("C{}: {} {}.end {} = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.point_display_name(c.point), c.distance.abs())); }
        for c in &self.axis_distance_arc_center_l1 { out.push(format!("C{}: {} {}.center {}.p1 = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.axis_distance_arc_center_l2 { out.push(format!("C{}: {} {}.center {}.p2 = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.axis_distance_arc_start_l1 { out.push(format!("C{}: {} {}.start {}.p1 = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.axis_distance_arc_start_l2 { out.push(format!("C{}: {} {}.start {}.p2 = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.axis_distance_arc_end_l1 { out.push(format!("C{}: {} {}.end {}.p1 = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.axis_distance_arc_end_l2 { out.push(format!("C{}: {} {}.end {}.p2 = {}", c.nid, axis_label(c.horizontal), self.arcs[c.arc].name, self.lines[c.line].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_ce_ce { out.push(format!("C{}: {} {}.center {}.center = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_ce_s { out.push(format!("C{}: {} {}.center {}.start = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_ce_e { out.push(format!("C{}: {} {}.center {}.end = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_s_ce { out.push(format!("C{}: {} {}.start {}.center = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_s_s { out.push(format!("C{}: {} {}.start {}.start = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_s_e { out.push(format!("C{}: {} {}.start {}.end = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_e_ce { out.push(format!("C{}: {} {}.end {}.center = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_e_s { out.push(format!("C{}: {} {}.end {}.start = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
        for c in &self.axis_distance_aa_e_e { out.push(format!("C{}: {} {}.end {}.end = {}", c.nid, axis_label(c.horizontal), self.arcs[c.a].name, self.arcs[c.b].name, c.distance.abs())); }
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
            self.serialize64(&mut tmp);
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
        let t_total = if TIMING_DEBUG { Some(std::time::Instant::now()) } else { None };

        self.update_tangent_flags();
        self.update_perpendicular_flags();
        self.update_line_dir_flags();

        let saved_drift = self.drift_isigma;
        self.drift_isigma = 0.0;

        let mut params = Vec::new();
        self.serialize64(&mut params);
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

        let t_hessian = if TIMING_DEBUG { Some(std::time::Instant::now()) } else { None };
        let mut grad = vec![0.0f64; n];
        let mut hessian = vec![0.0f64; n * n];
        self.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);
        self.drift_isigma = saved_drift;
        let t_hessian = t_hessian.map(|t| t.elapsed());

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
        let t_eigen = if TIMING_DEBUG { Some(std::time::Instant::now()) } else { None };
        let rank_from_evs = |evs: &[f64]| -> usize {
            let mut sorted: Vec<f64> = evs.iter().map(|v| v.abs()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        if TIMING_DEBUG {
            let t_h = t_hessian.unwrap().as_secs_f64() * 1000.0;
            let t_e = t_eigen.unwrap().elapsed().as_secs_f64() * 1000.0;
            let t_t = t_total.unwrap().elapsed().as_secs_f64() * 1000.0;
            eprintln!("[DOF-EIG] n={} analyze={} method={} hessian={:.3}ms eigen={:.3}ms total={:.3}ms dof={}",
                n, analyze, method, t_h, t_e, t_t, result.dof);
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
    /// Uses nalgebra SVD for n<32, faer SVD for n>=32.
    pub fn compute_dof(&mut self, analyze: bool) -> Result<DofResult, String> {
        let t_total = if TIMING_DEBUG { Some(std::time::Instant::now()) } else { None };

        self.prepare_expr_constraints();
        self.update_tangent_flags();
        self.update_perpendicular_flags();
        self.update_line_dir_flags();

        let saved_drift = self.drift_isigma;
        self.drift_isigma = 0.0;

        let mut params = Vec::new();
        self.serialize64(&mut params);
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

        let t_jac = if TIMING_DEBUG { Some(std::time::Instant::now()) } else { None };
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
        let m = jacobian.num_residuals();
        let t_jac = t_jac.map(|t| t.elapsed());

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
            self.cached_dof = Some(result.dof);
            return Ok(result);
        }


        let t_svd = if TIMING_DEBUG { Some(std::time::Instant::now()) } else { None };
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
        let rank_from_svs = |svs: &[f64]| -> usize {
            let mut sorted: Vec<f64> = svs.iter().map(|v| v.abs()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let max_sv = sorted.last().copied().unwrap_or(0.0);
            let upper_bound = max_sv * 0.01;
            // Floor near-zero sigmas at `max_sv * 1e-20` so they
            // participate in the gap search instead of being skipped.
            // No plausible real sigma lives below `max_sv * 1e-15`, so a
            // floor at `1e-20` leaves 5 orders of magnitude of headroom
            // above the floor for real rank sigmas to remain resolvable
            // while letting the zero-to-first-real gap compete on its
            // own merits. Multiple adjacent zeros collapse to ratio ~1
            // against each other (both pulled to the same floor),
            // leaving only the last-zero -> first-real transition as a
            // huge-gap candidate -- which is exactly what identifies the
            // rank cut.
            let floor = max_sv * 1e-20;
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
            svs.len() - best_cut
        };
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
        let rank = rank_from_svs(&jacobian.singular_values_column_normalised());
        let dof = n.saturating_sub(rank);

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
        let method = if n < 32 { "Jacobian::svd (nalgebra)" } else { "Jacobian::svd (faer)" };

        if TIMING_DEBUG {
            let t_j = t_jac.unwrap().as_secs_f64() * 1000.0;
            let t_s = t_svd.unwrap().elapsed().as_secs_f64() * 1000.0;
            let t_t = t_total.unwrap().elapsed().as_secs_f64() * 1000.0;
            eprintln!("[DOF] m={} n={} analyze={} method={} jacobian={:.3}ms svd={:.3}ms total={:.3}ms dof={}",
                m, n, analyze, method, t_j, t_s, t_t, result.dof);
        }
        self.cached_dof = Some(result.dof);
        Ok(result)
    }

    /// Return cached DOF or compute it (count only, no eigenvector analysis).
    pub fn dof(&mut self) -> Result<usize, String> {
        if let Some(d) = self.cached_dof { return Ok(d); }
        let result = self.compute_dof(false)?;
        Ok(result.dof)
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
                let cr = a.rotation.value.cos();
                let sr = a.rotation.value.sin();
                let ct = angle.cos();
                let st = angle.sin();
                let ax = a.center.value.x + a.radius.value * ct * cr - a.radius_b.value * st * sr;
                let ay = a.center.value.y + a.radius.value * ct * sr + a.radius_b.value * st * cr;
                let tx = -a.radius.value * st * cr - a.radius_b.value * ct * sr;
                let ty = -a.radius.value * st * sr + a.radius_b.value * ct * cr;
                // Direction from arc endpoint to the line's other end
                let (dx, dy) = if t.p1_arc_start || t.p1_arc_end {
                    (l.p2.value.x - ax, l.p2.value.y - ay)
                } else {
                    (l.p1.value.x - ax, l.p1.value.y - ay)
                };
                let dot = dx * tx + dy * ty;
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
        // Arc endpoint world position: c + R * cos(t) * cos(rot) - R_b * sin(t) * sin(rot),
        //                              c + R * cos(t) * sin(rot) + R_b * sin(t) * cos(rot)
        // where t is the start_angle or end_angle.
        fn arc_pt(a: &Arc, angle: f64) -> vect2d {
            let ct = angle.cos();
            let st = angle.sin();
            let cr = a.rotation.value.cos();
            let sr = a.rotation.value.sin();
            vect2d::new(
                a.center.value.x + a.radius.value * ct * cr - a.radius_b.value * st * sr,
                a.center.value.y + a.radius.value * ct * sr + a.radius_b.value * st * cr,
            )
        }

        let mut state = DragAutoAnchorState {
            helper_points: std::vec::Vec::new(),
            coincident_lp1_len: self.coincident_lp1.len(),
            coincident_lp2_len: self.coincident_lp2.len(),
            coincident_arc_start_len: self.coincident_arc_start.len(),
            coincident_arc_end_len: self.coincident_arc_end.len(),
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
                let pt = arc_pt(a, a.start_angle.value);
                let pos = vect2d::new(pt.x + 0.001, pt.y);
                let p = self.add_helper_point(pos);
                self.coincident_arc_start.push(CoincidentArcStart {
                    point: p, arc: r, nid: 0, cid: 0, hb: arael::model::CrossBlock::new(),
                });
                state.helper_points.push(p);
            }
            let a = &self.arcs[r];
            if a.end_angle.optimize && !joined_arc_end.contains(&r) {
                let pt = arc_pt(a, a.end_angle.value);
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
    pub fn remove_drag_auto_anchors(&mut self, state: DragAutoAnchorState) {
        self.coincident_lp1.truncate(state.coincident_lp1_len);
        self.coincident_lp2.truncate(state.coincident_lp2_len);
        self.coincident_arc_start.truncate(state.coincident_arc_start_len);
        self.coincident_arc_end.truncate(state.coincident_arc_end_len);
        for p in state.helper_points {
            self.points.remove(p);
        }
    }

    /// Solve the sketch constraints using Levenberg-Marquardt.
    /// Uses sparse faer Cholesky for n >= 64 params, dense Cholesky otherwise.
    /// When starting cost is high, uses graduated optimization (1% -> 10% ->
    /// 100% constraint strength) to avoid ill-conditioning from the large
    /// constraint/drift sigma ratio.
    pub fn solve(&mut self) -> arael::simple_lm::LmResult<f64> {
        use arael::simple_lm::LmProblem;
        // Rebuild expression constraints from dimensions with expr_str
        // (needed after load/undo since expr_constraints is not serialized)
        self.rebuild_expr_constraints();
        self.update_tangent_flags();
        self.update_perpendicular_flags();
        self.update_line_dir_flags();

        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        self.serialize64(&mut params64);
        let n = params64.len();

        if n == 0 {
            return arael::simple_lm::LmResult {
                x: params64, start_cost: 0.0, end_cost: 0.0,
                iterations: 0, accepted_iterations: 0,
                status: arael::simple_lm::LmStatus::Converged, final_lambda: 0.0,
                timing: None,
            };
        }

        // Build symbol bag and resolve expression constraints
        if !self.expr_constraints.is_empty() {
            let bag = SymbolBag::build(self);
            for ec in &mut self.expr_constraints {
                ec.resolve(&bag);
            }
            self.symbol_bag = Some(bag);
        }

        // Compute starting cost to decide strategy
        let start_cost = self.calc_cost(&params64);

        // Graduated optimization: when starting cost is high, the Hessian
        // condition number (constraint_isigma/drift_isigma)^2 can make LM
        // oscillate. Solve with reduced constraint strength first to get
        // close to the solution, then ramp up to full strength.
        let full_isigma = self.constraint_isigma;
        let graduated = start_cost > n as f64 * 1e-3;
        let stages: &[f64] = if graduated {
            &[0.01, 0.1, 1.0]
        } else {
            &[1.0]
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
        };

        for &scale in stages {
            self.constraint_isigma = full_isigma * scale;

            let mut params = std::vec::Vec::new();
            self.serialize64(&mut params);
            let cost = self.calc_cost(&params);

            let lambda = if cost > 1.0 {
                (cost * 1e-6).clamp(1e-4, 1.0)
            } else {
                1e-6
            };

            let config = arael::simple_lm::LmConfig::<f64> {
                initial_lambda: lambda,
                abs_precision: 1e-6,
                rel_precision: 1e-4,
                cost_threshold: n as f64 * 1e-6,
                min_iters: if cost > (n as f64 * 1e-4) { 32 } else { 8 },
                verbose: self.verbose,
                ..Default::default()
            };
            let stage_result = if n >= 64 {
                arael::simple_lm::solve_sparse_faer(&params, self, &config)
            } else {
                arael::simple_lm::solve(&params, self, &config)
            };
            self.deserialize64(&stage_result.x);
            total_iters += stage_result.iterations;
            total_accepted += stage_result.accepted_iterations;
            result.end_cost = stage_result.end_cost;
            result.x = stage_result.x;
            result.status = stage_result.status;
            result.final_lambda = stage_result.final_lambda;
        }

        self.constraint_isigma = full_isigma;
        self.normalise_ellipse_rotations();
        self.update_expr_dim_values();
        result.iterations = total_iters;
        result.accepted_iterations = total_accepted;
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
        let bag = SymbolBag::build(self);
        let mut params = Vec::new();
        self.serialize64(&mut params);
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
            kind: DimensionKind::LineLength(l0),
            value: 5.0, offset: vect2d::new(0.0, 1.0), text_along: 0.0,
            name: "d0".into(), expr_str: None, broken: false, derived: false,
            range: None,
        });

        sketch.prepare_expr_constraints();
        let mut params = Vec::new();
        sketch.serialize64(&mut params);
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
