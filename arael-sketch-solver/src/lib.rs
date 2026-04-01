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

use arael::model::{Model, Param, SelfBlock, CrossBlock, TripletBlock};
use arael::vect::vect2d;
use arael::refs::{Ref, Arena};

// Entity and constraint types must share one module scope because the
// #[arael::model] macro emits private `_PARAM_COUNT` constants that
// CrossBlock<A, B> expansions need to reference.
include!("entities.rs");
include!("constraints.rs");

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(root, extended)]
pub struct Sketch {
    pub points: Arena<Point>,
    pub lines: Arena<Line>,
    pub arcs: Arena<Arc>,
    // Solver parameters
    pub drift_isigma: f64,
    pub constraint_isigma: f64,
    #[arael(skip)]
    #[serde(default)]
    pub verbose: bool,
    // Auto-naming counters
    #[arael(skip)]
    pub next_point_id: u32,
    #[arael(skip)]
    pub next_line_id: u32,
    #[arael(skip)]
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
    pub point_on_arc: std::vec::Vec<PointOnArc>,
    pub parallel: std::vec::Vec<Parallel>,
    pub perpendicular: std::vec::Vec<Perpendicular>,
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
    pub distance_pl: std::vec::Vec<DistancePL>,
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
    // Dimension annotations
    #[arael(skip)]
    pub dimensions: std::vec::Vec<Dimension>,
    #[arael(skip)]
    pub next_dimension_id: u32,
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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
            point_on_arc: Vec::new(),
            parallel: Vec::new(),
            perpendicular: Vec::new(),
            collinear: Vec::new(),
            equal_length: Vec::new(),
            angle: Vec::new(),
            tangent_la: Vec::new(),
            concentric: Vec::new(),
            equal_radius: Vec::new(),
            tangent_aa: Vec::new(),
            symmetry_ll: Vec::new(),
            symmetry_pp: Vec::new(),
            distance_pl: Vec::new(),
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
            dimensions: Vec::new(),
            next_dimension_id: 0,
            user_params: Vec::new(),
            expr_constraints: Vec::new(),
            symbol_bag: None,
            expr_hb: TripletBlock::new(),
        }
    }

    /// Add a free point at the given position.
    pub fn add_point(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("P{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::new(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: false, name,
            hb: SelfBlock::new(),
        })
    }

    /// Add a fixed (non-optimizable) point at the given position.
    pub fn add_point_fixed(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("P{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::fixed(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: false, name,
            hb: SelfBlock::new(),
        })
    }

    /// Add a helper point (auto-removed when no constraints reference it).
    pub fn add_helper_point(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("Pc{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::new(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: true, name,
            hb: SelfBlock::new(),
        })
    }

    /// Add a line with two free endpoints.
    pub fn add_line(&mut self, p1: vect2d, p2: vect2d) -> Ref<Line> {
        let name = format!("L{}", self.next_line_id);
        self.next_line_id += 1;
        self.lines.push(Line {
            p1: Param::new(p1),
            p2: Param::new(p2),
            constraints: LineConstraints { horizontal: false, vertical: false, has_length: false, length: 0.0 },
            style: LineStyle::Solid, name,
            hb: SelfBlock::new(),
        })
    }

    /// Add an arc or circle. When `closed` is true, start/end angles are
    /// fixed (not optimized) since they are meaningless for a full circle.
    pub fn add_arc(&mut self, center: vect2d, radius: f64, start: f64, end: f64, closed: bool) -> Ref<Arc> {
        let name = format!("A{}", self.next_arc_id);
        self.next_arc_id += 1;
        self.arcs.push(Arc {
            center: Param::new(center),
            radius: Param::new(radius),
            start_angle: if closed { Param::fixed(start) } else { Param::new(start) },
            end_angle: if closed { Param::fixed(end) } else { Param::new(end) },
            closed,
            style: LineStyle::Solid, name,
            constraints: ArcConstraints { has_target_radius: false, target_radius: 0.0 },
            hb: SelfBlock::new(),
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
        self.point_on_arc.retain(|c| c.point != r);
        self.distance_pl.retain(|c| c.point != r);
        self.coincident_arc_center.retain(|c| c.point != r);
        self.coincident_arc_start.retain(|c| c.point != r);
        self.coincident_arc_end.retain(|c| c.point != r);
        self.distance_lp1.retain(|c| c.point != r);
        self.distance_lp2.retain(|c| c.point != r);
        self.symmetry_pp.retain(|c| c.a != r && c.c != r);
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
        self.parallel.retain(|c| c.a != r && c.b != r);
        self.perpendicular.retain(|c| c.a != r && c.b != r);
        self.collinear.retain(|c| c.a != r && c.b != r);
        self.equal_length.retain(|c| c.a != r && c.b != r);
        self.angle.retain(|c| c.a != r && c.b != r);
        self.tangent_la.retain(|c| c.line != r);
        self.symmetry_ll.retain(|c| c.a != r && c.b != r && c.c != r);
        self.symmetry_pp.retain(|c| c.line != r);
        self.distance_pl.retain(|c| c.line != r);
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
        self.midpoint_arc_start.retain(|c| c.arc != r);
        self.midpoint_arc_end.retain(|c| c.arc != r);
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
        self.cleanup_helper_points();
    }

    /// Remove helper points that have no remaining constraints referencing them.
    pub fn cleanup_helper_points(&mut self) {
        // Collect all point refs that appear in any constraint
        let mut referenced: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut add_pt = |r: Ref<Point>| { referenced.insert(r.index()); };
        for c in &self.coincident_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.coincident_lp1 { add_pt(c.point); }
        for c in &self.coincident_lp2 { add_pt(c.point); }
        for c in &self.distance_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.hdistance_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.vdistance_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.point_on_line { add_pt(c.point); }
        for c in &self.midpoint { add_pt(c.point); }
        for c in &self.point_on_arc { add_pt(c.point); }
        for c in &self.distance_pl { add_pt(c.point); }
        for c in &self.coincident_arc_center { add_pt(c.point); }
        for c in &self.coincident_arc_start { add_pt(c.point); }
        for c in &self.coincident_arc_end { add_pt(c.point); }

        // Remove unreferenced helper points
        let to_remove: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| {
                let p = &self.points[*r];
                p.helper && !referenced.contains(&r.index())
            })
            .collect();
        for r in to_remove {
            self.points.remove(r);
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
        dedup_pa!(self.point_on_arc, "point_on_arc");
        dedup_ab!(self.parallel, "parallel", self.lines, self.lines);
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
            // Remove merged points
            for (old, _) in &merged { self.points.remove(Ref::new(*old)); }
            // Dedup again after remapping
            self.dedup_constraints();
        }

        // Phase 2: Replace helper-point bridges with direct constraints
        let helper_refs: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| self.points[*r].helper)
            .collect();
        for hr in &helper_refs {
            let hr = *hr;
            let lp1: Option<Ref<Line>> = self.coincident_lp1.iter().find(|c| c.point == hr).map(|c| c.line);
            let lp2: Option<Ref<Line>> = self.coincident_lp2.iter().find(|c| c.point == hr).map(|c| c.line);
            let ac: Option<Ref<Arc>> = self.coincident_arc_center.iter().find(|c| c.point == hr).map(|c| c.arc);
            let a_start: Option<Ref<Arc>> = self.coincident_arc_start.iter().find(|c| c.point == hr).map(|c| c.arc);
            let a_end: Option<Ref<Arc>> = self.coincident_arc_end.iter().find(|c| c.point == hr).map(|c| c.arc);

            macro_rules! consolidate {
                ($line_opt:expr, $arc_opt:expr, $lp_coll:ident, $arc_coll:ident, $direct_coll:ident, $DirectType:ident, $label:expr) => {
                    if let (Some(line), Some(arc)) = ($line_opt, $arc_opt) {
                        self.$direct_coll.push($DirectType { line, arc, hb: CrossBlock::new() });
                        self.$lp_coll.retain(|c| !(c.line == line && c.point == hr));
                        self.$arc_coll.retain(|c| !(c.point == hr && c.arc == arc));
                        eprintln!("INFO: consolidated helper {} -> {}", hr.index(), $label);
                    }
                };
            }
            consolidate!(lp1, ac, coincident_lp1, coincident_arc_center, coincident_lp1_arc_center, CoincidentLP1ArcCenter, "LP1ArcCenter");
            consolidate!(lp2, ac, coincident_lp2, coincident_arc_center, coincident_lp2_arc_center, CoincidentLP2ArcCenter, "LP2ArcCenter");
            consolidate!(lp1, a_start, coincident_lp1, coincident_arc_start, coincident_lp1_arc_start, CoincidentLP1ArcStart, "LP1ArcStart");
            consolidate!(lp2, a_start, coincident_lp2, coincident_arc_start, coincident_lp2_arc_start, CoincidentLP2ArcStart, "LP2ArcStart");
            consolidate!(lp1, a_end, coincident_lp1, coincident_arc_end, coincident_lp1_arc_end, CoincidentLP1ArcEnd, "LP1ArcEnd");
            consolidate!(lp2, a_end, coincident_lp2, coincident_arc_end, coincident_lp2_arc_end, CoincidentLP2ArcEnd, "LP2ArcEnd");
        }
        self.cleanup_helper_points();
        self.dedup_constraints();
    }

}

impl arael::model::ExtendedModel for Sketch {
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

    fn extended_compute64(&mut self, params: &[f64]) {
        if self.expr_constraints.is_empty() { return; }
        let bag = self.symbol_bag.as_ref().expect("symbol_bag not built");
        let vars = bag.eval_vars(params);
        let isigma = self.constraint_isigma;
        let hb = &mut self.expr_hb as *mut TripletBlock<f64>;
        for ec in &self.expr_constraints {
            if let Err(e) = ec.compute(&vars, isigma, unsafe { &mut *hb }) {
                eprintln!("expr constraint eval error: {}: {}", ec.description, e);
            }
        }
    }
}

impl Sketch {
    /// Add an expression constraint. The expression should evaluate to 0
    /// when satisfied. Symbols are resolved against current entity names
    /// and dimensions.
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
        if !has_expr && !has_user_params {
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
        if name.bytes().next().map_or(true, |b| b.is_ascii_digit()) {
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

    /// Add an expression-based dimension. The expression string is parsed
    /// List all active constraints as human-readable strings.
    pub fn list_constraints(&self) -> Vec<String> {
        let mut out = Vec::new();
        // Entity-level flags
        for r in self.lines.refs() {
            let l = &self.lines[r];
            if l.constraints.horizontal { out.push(format!("horizontal {}", l.name)); }
            if l.constraints.vertical { out.push(format!("vertical {}", l.name)); }
            if l.constraints.has_length { out.push(format!("length {} = {}", l.name, l.constraints.length)); }
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
            if !a.center.optimize { out.push(format!("lock {}.center", a.name)); }
        }
        // Constraint vectors (use macros to reduce repetition)
        macro_rules! list_cross {
            ($vec:expr, $name:literal, $fa:ident, $fb:ident) => {
                for c in &$vec {
                    let na = &self.lines[c.$fa].name;
                    let nb = &self.lines[c.$fb].name;
                    out.push(format!("{} {} {}", $name, na, nb));
                }
            };
        }
        list_cross!(self.parallel, "parallel", a, b);
        list_cross!(self.perpendicular, "perpendicular", a, b);
        list_cross!(self.collinear, "collinear", a, b);
        list_cross!(self.equal_length, "equal_length", a, b);
        for c in &self.coincident_pp {
            out.push(format!("coincident {} {}", self.points[c.a].name, self.points[c.b].name));
        }
        for c in &self.coincident_ll11 { out.push(format!("coincident {}.p1 {}.p1", self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_ll12 { out.push(format!("coincident {}.p1 {}.p2", self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_ll21 { out.push(format!("coincident {}.p2 {}.p1", self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_ll22 { out.push(format!("coincident {}.p2 {}.p2", self.lines[c.a].name, self.lines[c.b].name)); }
        for c in &self.coincident_lp1 { out.push(format!("coincident {}.p1 {}", self.lines[c.line].name, self.points[c.point].name)); }
        for c in &self.coincident_lp2 { out.push(format!("coincident {}.p2 {}", self.lines[c.line].name, self.points[c.point].name)); }
        for c in &self.angle { out.push(format!("angle {} {} = {:.1}deg", self.lines[c.a].name, self.lines[c.b].name, c.angle.to_degrees())); }
        for c in &self.tangent_la { out.push(format!("tangent {} {}", self.lines[c.line].name, self.arcs[c.arc].name)); }
        for c in &self.tangent_aa { out.push(format!("tangent {} {}", self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.concentric { out.push(format!("concentric {} {}", self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.equal_radius { out.push(format!("equal_radius {} {}", self.arcs[c.a].name, self.arcs[c.b].name)); }
        for c in &self.symmetry_ll { out.push(format!("symmetry {} {} {}", self.lines[c.a].name, self.lines[c.b].name, self.lines[c.c].name)); }
        for c in &self.symmetry_pp { out.push(format!("symmetry {} {} {}", self.points[c.a].name, self.lines[c.line].name, self.points[c.c].name)); }
        for c in &self.point_on_line { out.push(format!("point_on {} {}", self.points[c.point].name, self.lines[c.line].name)); }
        for c in &self.point_on_arc { out.push(format!("point_on {} {}", self.points[c.point].name, self.arcs[c.arc].name)); }
        for c in &self.midpoint { out.push(format!("midpoint {} {}", self.points[c.point].name, self.lines[c.line].name)); }
        for c in &self.distance_pp { out.push(format!("distance {} {} = {}", self.points[c.a].name, self.points[c.b].name, c.distance)); }
        for c in &self.distance_pl { out.push(format!("distance_pl {} = {}", self.lines[c.line].name, c.distance)); }
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
        };
        let measured = dim.measured_symbol(self);
        self.dimensions.push(dim);

        // Residual: measured - expr = 0
        let residual = measured - parsed;
        self.add_expr_constraint(residual, format!("{} = {}", name, expr_str));
        Ok(())
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

        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        self.serialize64(&mut params64);
        let n = params64.len();

        if n == 0 {
            return arael::simple_lm::LmResult {
                x: params64, start_cost: 0.0, end_cost: 0.0, iterations: 0,
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
        let mut result = arael::simple_lm::LmResult {
            x: params64.clone(),
            start_cost,
            end_cost: start_cost,
            iterations: 0,
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
            result.end_cost = stage_result.end_cost;
            result.x = stage_result.x;
        }

        self.constraint_isigma = full_isigma;
        self.update_expr_dim_values();
        result.iterations = total_iters;
        result
    }

    /// Evaluate expression/derived dimensions and user params, cache their computed values.
    pub fn update_expr_dim_values(&mut self) {
        let has_work = self.dimensions.iter().any(|d| d.expr_str.is_some() || d.derived)
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
            if let Some(ref expr_str) = dim.expr_str {
                if let Ok(parsed) = arael_sym::parse(expr_str) {
                    let expanded = expr_constraint::expand_derived(&parsed, &bag);
                    match expanded.eval(&vars) {
                        Ok(val) => dim.value = val,
                        Err(_) => dim.broken = true,
                    }
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
    }
}
