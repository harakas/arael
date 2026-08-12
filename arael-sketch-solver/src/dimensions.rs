use arael::refs::Ref;
use arael::vect::vect2d;

use crate::{Point, Line, Arc};

// ---------------------------------------------------------------------------
// Dimension annotations (constraint + visual)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum DimensionEndpoint {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum DimensionKind {
    LineLength(Ref<Line>),
    PointPointDistance(DimensionEndpoint, DimensionEndpoint),
    PointLineDistance(DimensionEndpoint, Ref<Line>),
    ArcRadius(Ref<Arc>),
    /// Ellipse semi-minor axis radius.
    ArcRadiusB(Ref<Arc>),
    /// Arc sweep angle (end_angle - start_angle), stored in degrees.
    ArcSweep(Ref<Arc>),
    /// Ellipse/earc rotation: angle of the semi-major axis from the
    /// world x-axis, stored in degrees. Only meaningful when the arc
    /// is an ellipse (`is_ellipse`); for circular arcs `rotation` is
    /// `Param::fixed(0.0)` and this dimension is a no-op.
    ArcRotation(Ref<Arc>),
    /// Angle between two lines. The bool is `supplement`: when true,
    /// constrains the supplementary angle (pi - angle) instead.
    Angle(Ref<Line>, Ref<Line>, bool),
    /// Horizontal (x-axis) distance between two endpoints. Displayed unsigned, applied signed.
    HDistance(DimensionEndpoint, DimensionEndpoint),
    /// Vertical (y-axis) distance between two endpoints. Displayed unsigned, applied signed.
    VDistance(DimensionEndpoint, DimensionEndpoint),
    /// Line angle from x-axis, in degrees.
    LineAngle(Ref<Line>),
    /// Radial distance between two concentric arcs/circles. Stored as
    /// the magnitude `|b.radius - a.radius|`; the sign is captured in
    /// the underlying constraint at creation so the solver can't flip
    /// which arc is outer under value updates. Requires a
    /// `Concentric` constraint coupling the same arcs to be
    /// meaningful; the dimension is auto-removed when that constraint
    /// is deleted.
    ConcentricDistance(Ref<Arc>, Ref<Arc>),
    /// Perpendicular distance between two lines. Meaningful only when
    /// the lines are parallel -- the creation sites (CLI / GUI) apply
    /// a `Parallel` constraint alongside this dimension. The residual
    /// reuses the point-to-line perpendicular math with line B's `p1`
    /// as the anchor point (any point on B works once parallel is
    /// enforced). The dimension is auto-removed when the backing
    /// `Parallel` constraint is deleted.
    LineLineDistance(Ref<Line>, Ref<Line>),
}

impl DimensionEndpoint {
    pub fn references_point(&self, r: Ref<Point>) -> bool {
        matches!(self, DimensionEndpoint::Point(p) if *p == r)
    }
    pub fn references_line(&self, r: Ref<Line>) -> bool {
        matches!(self, DimensionEndpoint::LineP1(l) | DimensionEndpoint::LineP2(l) if *l == r)
    }
    pub fn references_arc(&self, r: Ref<Arc>) -> bool {
        matches!(self, DimensionEndpoint::ArcCenter(a) | DimensionEndpoint::ArcStart(a) | DimensionEndpoint::ArcEnd(a) if *a == r)
    }
}

impl DimensionKind {
    pub fn references_point(&self, r: Ref<Point>) -> bool {
        match self {
            DimensionKind::PointPointDistance(a, b) => a.references_point(r) || b.references_point(r),
            DimensionKind::PointLineDistance(a, _) => a.references_point(r),
            DimensionKind::HDistance(a, b) | DimensionKind::VDistance(a, b) => a.references_point(r) || b.references_point(r),
            DimensionKind::LineAngle(_) => false,
            _ => false,
        }
    }
    pub fn references_line(&self, r: Ref<Line>) -> bool {
        match self {
            DimensionKind::LineLength(l) => *l == r,
            DimensionKind::PointPointDistance(a, b) => a.references_line(r) || b.references_line(r),
            DimensionKind::PointLineDistance(a, l) => a.references_line(r) || *l == r,
            DimensionKind::Angle(a, b, _) => *a == r || *b == r,
            DimensionKind::HDistance(a, b) | DimensionKind::VDistance(a, b) => a.references_line(r) || b.references_line(r),
            DimensionKind::LineAngle(l) => *l == r,
            DimensionKind::LineLineDistance(a, b) => *a == r || *b == r,
            _ => false,
        }
    }
    pub fn references_arc(&self, r: Ref<Arc>) -> bool {
        match self {
            DimensionKind::ArcRadius(a) | DimensionKind::ArcRadiusB(a) | DimensionKind::ArcSweep(a) | DimensionKind::ArcRotation(a) => *a == r,
            DimensionKind::PointPointDistance(a, b) => a.references_arc(r) || b.references_arc(r),
            DimensionKind::PointLineDistance(a, _) => a.references_arc(r),
            DimensionKind::HDistance(a, b) | DimensionKind::VDistance(a, b) => a.references_arc(r) || b.references_arc(r),
            DimensionKind::ConcentricDistance(a, b) => *a == r || *b == r,
            DimensionKind::LineAngle(_) => false,
            _ => false,
        }
    }

    /// True when this is a `ConcentricDistance` referencing the
    /// (unordered) arc pair `(a, b)`. Used by the cascade that
    /// removes such dimensions when their backing `Concentric`
    /// constraint is deleted.
    pub fn references_concentric_pair(&self, a: Ref<Arc>, b: Ref<Arc>) -> bool {
        matches!(self,
            DimensionKind::ConcentricDistance(x, y)
            if (*x == a && *y == b) || (*x == b && *y == a)
        )
    }

    /// True when this is a `LineLineDistance` referencing the
    /// (unordered) line pair `(a, b)`. Used by the cascade that
    /// removes such dimensions when their backing `Parallel`
    /// constraint is deleted.
    pub fn references_parallel_pair(&self, a: Ref<Line>, b: Ref<Line>) -> bool {
        matches!(self,
            DimensionKind::LineLineDistance(x, y)
            if (*x == a && *y == b) || (*x == b && *y == a)
        )
    }
}

/// A single bound value: either a resolved literal (captured at
/// command time, including the result of an evaluate-once
/// expression) or a live arael-sym source string that the solver
/// re-evaluates every iteration.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RangeValue {
    Literal(f64),
    Live(String),
}

impl RangeValue {
    /// Best-effort numeric read for display / diagnostics. Live
    /// values return the stashed f64 from the dimension's cached
    /// `value` field via the caller; we just expose the literal.
    pub fn as_literal(&self) -> Option<f64> {
        match self {
            RangeValue::Literal(v) => Some(*v),
            RangeValue::Live(_) => None,
        }
    }
}

impl std::fmt::Display for RangeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RangeValue::Literal(v) => write!(f, "{}", v),
            RangeValue::Live(src) => write!(f, "{}", src),
        }
    }
}

/// One-sided or two-sided bound on a dimension's measured value.
/// Residual is piecewise-zero inside the feasible region, linear
/// outside (penalty method). Inactive bounds contribute zero rows
/// to J and zero curvature to J^T J; the solver sees them only
/// when they're violated. Each slot is a `RangeValue`, so bounds
/// can be literals or live expressions that track user params.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RangeBound {
    /// measured >= v
    Min(RangeValue),
    /// measured <= v
    Max(RangeValue),
    /// lo <= measured <= hi
    Between(RangeValue, RangeValue),
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Dimension {
    /// Permanent unique id: the stable handle for selection and the
    /// dimension actions. Vec indices shift on every removal; the did
    /// never changes. Minted by Sketch::assign_dimension_ids.
    #[serde(default)]
    pub did: u32,
    pub kind: DimensionKind,
    pub value: f64,
    pub offset: vect2d,      // visual offset (y = perpendicular distance)
    pub text_along: f64,     // text position along the line: 0=center, -0.5..0.5=within arrows, outside=extend
    pub name: String,
    /// Expression string for parametric dimensions (e.g. "d0 * 2").
    /// None = constant numeric value.
    #[serde(default)]
    pub expr_str: Option<String>,
    /// True when an expression dimension references a symbol that no
    /// longer exists. The dimension falls back to its last computed value.
    #[serde(default)]
    pub broken: bool,
    /// Derived (reference) dimension: displays measured value but does
    /// not constrain the solver. Shown with parentheses: "(3.40)".
    #[serde(default)]
    pub derived: bool,
    /// Inequality bound on the measured value. When Some, the dimension
    /// contributes a barrier residual (zero inside the feasible region,
    /// linear penalty outside). `value` tracks the current measured
    /// reading for display. Mutually exclusive with `expr_str` and
    /// `derived`.
    #[serde(default)]
    pub range: Option<RangeBound>,
}

// ---------------------------------------------------------------------------
// User-defined parameters
// ---------------------------------------------------------------------------

/// A named parameter defined by the user, usable in dimension expressions.
/// The value can be a numeric literal or an expression referencing other
/// parameters, dimensions, and entity properties.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct UserParam {
    pub name: String,
    pub expr_str: String,
    pub value: f64,
    #[serde(default)]
    pub broken: bool,
}

/// Check whether a name matches a system naming pattern (d0, L0, P0, A0, etc.)
/// that could conflict with auto-generated entity/dimension names.
pub fn is_system_name(name: &str) -> bool {
    if name.is_empty() { return false; }
    let bytes = name.as_bytes();
    // Single-letter prefix + digits: d0, L5, P0, A3, etc.
    let prefix = bytes[0];
    if matches!(prefix, b'd' | b'L' | b'P' | b'A') && bytes.len() > 1 {
        let rest = &name[1..];
        // Could be "L0" or "L0.p1.x" -- check if first part after prefix is digits
        let num_part = rest.split('.').next().unwrap_or("");
        if !num_part.is_empty() && num_part.bytes().all(|b| b.is_ascii_digit()) {
            return true;
        }
    }
    false
}

/// Opaque "which side of the bound" marker used when the caller
/// already has per-slot bound `E` expressions in hand.
pub enum ResolvedBound {
    Min(arael_sym::E),
    Max(arael_sym::E),
    Between(arael_sym::E, arael_sym::E),
}

impl Dimension {
    /// Build a barrier residual E tree from a `measured` expression
    /// and pre-resolved bound(s). The residual is zero inside the
    /// feasible region and linear on the violating side (heaviside
    /// convention `H(x)=1` for `x >= 0`). Gauss-Newton auto-diff
    /// treats the heaviside gradient as zero, matching the line-
    /// length positivity guards. The caller resolves each
    /// `RangeValue` into an `E` first: `Literal(v)` -> `constant(v)`,
    /// `Live(src)` -> `parse_with_functions + expand_derived` against
    /// the current `SymbolBag`.
    pub fn range_residual(rb: &ResolvedBound, measured: arael_sym::E) -> arael_sym::E {
        use arael_sym::heaviside;
        match rb {
            ResolvedBound::Min(v) => {
                let d = v.clone() - measured;
                heaviside(d.clone()) * d
            }
            ResolvedBound::Max(v) => {
                let d = measured - v.clone();
                heaviside(d.clone()) * d
            }
            ResolvedBound::Between(lo, hi) => {
                let d_lo = lo.clone() - measured.clone();
                let d_hi = measured - hi.clone();
                heaviside(d_lo.clone()) * d_lo + heaviside(d_hi.clone()) * d_hi
            }
        }
    }

    /// Build a symbolic expression for the measured property of this
    /// dimension kind, using entity names from the sketch.
    pub fn measured_symbol(&self, sketch: &super::Sketch) -> arael_sym::E {
        use arael_sym::symbol;
        match &self.kind {
            DimensionKind::LineLength(r) => {
                let name = &sketch.lines[*r].name;
                symbol(&format!("{}.length", name))
            }
            DimensionKind::ArcRadius(r) => {
                let name = &sketch.arcs[*r].name;
                symbol(&format!("{}.radius", name))
            }
            DimensionKind::ArcRadiusB(r) => {
                let name = &sketch.arcs[*r].name;
                symbol(&format!("{}.radius_b", name))
            }
            DimensionKind::ArcSweep(r) => {
                let name = &sketch.arcs[*r].name;
                let start = symbol(&format!("{}.start_angle", name));
                let end = symbol(&format!("{}.end_angle", name));
                arael_sym::abs(end - start) * arael_sym::constant(180.0 / std::f64::consts::PI)
            }
            DimensionKind::ArcRotation(r) => {
                let name = &sketch.arcs[*r].name;
                symbol(&format!("{}.rotation", name))
                    * arael_sym::constant(180.0 / std::f64::consts::PI)
            }
            DimensionKind::PointPointDistance(a, b) => {
                // When one endpoint anchors to a line endpoint and the other
                // is on the same line, both are collinear with the line.
                // Use a signed along-line projection so the residual has a
                // preferred direction -- prevents the solver from flipping
                // the point to the mirror solution under large scale changes.
                if let Some(expr) = try_along_line_symbol(a, b, sketch) {
                    return expr;
                }
                let pa = dim_endpoint_symbol(a, sketch);
                let pb = dim_endpoint_symbol(b, sketch);
                let dx = pa.0 - pb.0;
                let dy = pa.1 - pb.1;
                arael_sym::sqrt(dx.clone() * dx + dy.clone() * dy)
            }
            DimensionKind::PointLineDistance(pt, line) => {
                let (px, py) = dim_endpoint_symbol(pt, sketch);
                let ln = &sketch.lines[*line].name;
                let p1x = symbol(&format!("{}.p1.x", ln));
                let p1y = symbol(&format!("{}.p1.y", ln));
                let p2x = symbol(&format!("{}.p2.x", ln));
                let p2y = symbol(&format!("{}.p2.y", ln));
                let dx = p2x.clone() - p1x.clone();
                let dy = p2y.clone() - p1y.clone();
                let len = arael_sym::sqrt(dx.clone() * dx.clone() + dy.clone() * dy.clone());
                let signed = ((px - p1x) * dy.clone() - (py - p1y) * dx.clone()) / len;
                // Determine sign from current geometry so expression matches
                // the positive value the user sees. Negate if point is on
                // the negative side of the line.
                let pt_pos = dim_endpoint_pos(pt, sketch);
                let l = &sketch.lines[*line];
                let ldx = l.p2.value.x - l.p1.value.x;
                let ldy = l.p2.value.y - l.p1.value.y;
                let cross = (pt_pos.x - l.p1.value.x) * ldy - (pt_pos.y - l.p1.value.y) * ldx;
                if cross >= 0.0 { signed } else { -signed }
            }
            DimensionKind::HDistance(a, b) => {
                let pa = dim_endpoint_symbol(a, sketch);
                let pb = dim_endpoint_symbol(b, sketch);
                arael_sym::abs(pa.0 - pb.0)
            }
            DimensionKind::VDistance(a, b) => {
                let pa = dim_endpoint_symbol(a, sketch);
                let pb = dim_endpoint_symbol(b, sketch);
                arael_sym::abs(pa.1 - pb.1)
            }
            DimensionKind::LineAngle(r) => {
                let name = &sketch.lines[*r].name;
                let p1x = symbol(&format!("{}.p1.x", name));
                let p1y = symbol(&format!("{}.p1.y", name));
                let p2x = symbol(&format!("{}.p2.x", name));
                let p2y = symbol(&format!("{}.p2.y", name));
                let dx = p2x - p1x;
                let dy = p2y - p1y;
                arael_sym::atan2(dy, dx) * arael_sym::constant(180.0 / std::f64::consts::PI)
            }
            DimensionKind::Angle(a, b, supplement) => {
                let la = &sketch.lines[*a].name;
                let lb = &sketch.lines[*b].name;
                let dx1 = symbol(&format!("{}.p2.x", la)) - symbol(&format!("{}.p1.x", la));
                let dy1 = symbol(&format!("{}.p2.y", la)) - symbol(&format!("{}.p1.y", la));
                let dx2 = symbol(&format!("{}.p2.x", lb)) - symbol(&format!("{}.p1.x", lb));
                let dy2 = symbol(&format!("{}.p2.y", lb)) - symbol(&format!("{}.p1.y", lb));
                let cross = dx1.clone() * dy2.clone() - dy1.clone() * dx2.clone();
                let dot = dx1 * dx2 + dy1 * dy2;
                let angle = arael_sym::atan2(cross, dot);
                // Determine sign from current geometry
                let la_line = &sketch.lines[*a];
                let lb_line = &sketch.lines[*b];
                let cur_dx1 = la_line.p2.value.x - la_line.p1.value.x;
                let cur_dy1 = la_line.p2.value.y - la_line.p1.value.y;
                let cur_dx2 = lb_line.p2.value.x - lb_line.p1.value.x;
                let cur_dy2 = lb_line.p2.value.y - lb_line.p1.value.y;
                let cur_cross = cur_dx1 * cur_dy2 - cur_dy1 * cur_dx2;
                let cur_dot = cur_dx1 * cur_dx2 + cur_dy1 * cur_dy2;
                let cur_angle = cur_cross.atan2(cur_dot);
                // Convert to degrees, matching sign of current angle
                let deg_factor = arael_sym::constant(180.0 / std::f64::consts::PI);
                let signed_deg = angle * deg_factor;
                if *supplement {
                    // 180 - |angle|: the reading is a magnitude, like
                    // the non-supplement branch below.
                    if cur_angle >= 0.0 {
                        arael_sym::constant(180.0) - signed_deg
                    } else {
                        arael_sym::constant(180.0) + signed_deg
                    }
                } else {
                    if cur_angle >= 0.0 { signed_deg } else { -signed_deg }
                }
            }
            DimensionKind::ConcentricDistance(a, b) => {
                // Signed along-radius difference, with sign captured at
                // build time so the residual stays sign-stable under big
                // value changes (no flip on which arc is outer).
                let na = &sketch.arcs[*a].name;
                let nb = &sketch.arcs[*b].name;
                let signed = symbol(&format!("{}.radius", nb))
                           - symbol(&format!("{}.radius", na));
                let init_diff = sketch.arcs[*b].radius.value - sketch.arcs[*a].radius.value;
                if init_diff >= 0.0 { signed } else { -signed }
            }
            DimensionKind::LineLineDistance(a, b) => {
                // Reuse the point-to-line perpendicular expression with
                // line B's p1 as the anchor point. Once the paired
                // Parallel constraint is active, the perpendicular
                // distance from any point on B to A equals the inter-
                // line gap by definition. Sign captured from current
                // geometry so the residual stays stable across solver
                // iterations.
                let la = &sketch.lines[*a].name;
                let lb = &sketch.lines[*b].name;
                let px = symbol(&format!("{}.p1.x", lb));
                let py = symbol(&format!("{}.p1.y", lb));
                let p1x = symbol(&format!("{}.p1.x", la));
                let p1y = symbol(&format!("{}.p1.y", la));
                let p2x = symbol(&format!("{}.p2.x", la));
                let p2y = symbol(&format!("{}.p2.y", la));
                let dx = p2x.clone() - p1x.clone();
                let dy = p2y.clone() - p1y.clone();
                let len = arael_sym::sqrt(dx.clone() * dx.clone() + dy.clone() * dy.clone());
                let signed = ((px - p1x) * dy.clone() - (py - p1y) * dx.clone()) / len;
                let la_line = &sketch.lines[*a];
                let lb_line = &sketch.lines[*b];
                let ldx = la_line.p2.value.x - la_line.p1.value.x;
                let ldy = la_line.p2.value.y - la_line.p1.value.y;
                let cross = (lb_line.p1.value.x - la_line.p1.value.x) * ldy
                          - (lb_line.p1.value.y - la_line.p1.value.y) * ldx;
                if cross >= 0.0 { signed } else { -signed }
            }
        }
    }
}

/// Get the current numeric position of a DimensionEndpoint.
fn dim_endpoint_pos(ep: &DimensionEndpoint, sketch: &super::Sketch) -> vect2d {
    match ep {
        DimensionEndpoint::Point(r) => sketch.points[*r].pos.value,
        DimensionEndpoint::LineP1(r) => sketch.lines[*r].p1.value,
        DimensionEndpoint::LineP2(r) => sketch.lines[*r].p2.value,
        DimensionEndpoint::ArcCenter(r) => sketch.arcs[*r].center.value,
        DimensionEndpoint::ArcStart(r) => sketch.arcs[*r].start_pos(),
        DimensionEndpoint::ArcEnd(r) => sketch.arcs[*r].end_pos(),
    }
}

/// Build symbolic (x, y) expressions for a DimensionEndpoint.
fn dim_endpoint_symbol(ep: &DimensionEndpoint, sketch: &super::Sketch) -> (arael_sym::E, arael_sym::E) {
    use arael_sym::symbol;
    match ep {
        DimensionEndpoint::Point(r) => {
            let n = &sketch.points[*r].name;
            (symbol(&format!("{}.pos.x", n)), symbol(&format!("{}.pos.y", n)))
        }
        DimensionEndpoint::LineP1(r) => {
            let n = &sketch.lines[*r].name;
            (symbol(&format!("{}.p1.x", n)), symbol(&format!("{}.p1.y", n)))
        }
        DimensionEndpoint::LineP2(r) => {
            let n = &sketch.lines[*r].name;
            (symbol(&format!("{}.p2.x", n)), symbol(&format!("{}.p2.y", n)))
        }
        DimensionEndpoint::ArcCenter(r) => {
            let n = &sketch.arcs[*r].name;
            (symbol(&format!("{}.center.x", n)), symbol(&format!("{}.center.y", n)))
        }
        DimensionEndpoint::ArcStart(r) => {
            let n = &sketch.arcs[*r].name;
            let sa = symbol(&format!("{}.start_angle", n));
            crate::arc_endpoint_symbols(n, sa)
        }
        DimensionEndpoint::ArcEnd(r) => {
            let n = &sketch.arcs[*r].name;
            let ea = symbol(&format!("{}.end_angle", n));
            crate::arc_endpoint_symbols(n, ea)
        }
    }
}

// ---------------------------------------------------------------------------
// Along-line detection for signed PointPointDistance
//
// When a PointPointDistance dimension has one endpoint anchored to a line
// endpoint (via CoincidentLP1/LP2 or the arc-specific variants) and the
// other endpoint is on that same line (either itself a line endpoint, or
// via PointOnLine / helper-point bindings), both points are collinear with
// the line. In that case the Euclidean sqrt residual admits a mirror
// solution: the point can satisfy |distance| on either side of the anchor.
// Under dynamic scale changes this lets the solver drift into the mirrored
// geometry.
//
// The helpers below detect the pattern and let measured_symbol emit a
// signed along-line projection instead of sqrt, with the sign captured
// from the current geometry. Matches the idiom already used by
// DimensionKind::PointLineDistance above.
// ---------------------------------------------------------------------------

/// If `ep` is bound to a specific line endpoint, return that line and
/// whether the endpoint is p1 (true) or p2 (false).
fn endpoint_is_line_endpoint(
    sketch: &super::Sketch,
    ep: &DimensionEndpoint,
) -> Option<(Ref<Line>, bool)> {
    match ep {
        DimensionEndpoint::LineP1(l) => Some((*l, true)),
        DimensionEndpoint::LineP2(l) => Some((*l, false)),
        DimensionEndpoint::Point(p) => {
            if let Some(c) = sketch.coincident_lp1.iter().find(|c| c.point == *p) {
                return Some((c.line, true));
            }
            if let Some(c) = sketch.coincident_lp2.iter().find(|c| c.point == *p) {
                return Some((c.line, false));
            }
            None
        }
        DimensionEndpoint::ArcCenter(a) => {
            if let Some(c) = sketch.coincident_lp1_arc_center.iter().find(|c| c.arc == *a) {
                return Some((c.line, true));
            }
            if let Some(c) = sketch.coincident_lp2_arc_center.iter().find(|c| c.arc == *a) {
                return Some((c.line, false));
            }
            None
        }
        DimensionEndpoint::ArcStart(a) => {
            if let Some(c) = sketch.coincident_lp1_arc_start.iter().find(|c| c.arc == *a) {
                return Some((c.line, true));
            }
            if let Some(c) = sketch.coincident_lp2_arc_start.iter().find(|c| c.arc == *a) {
                return Some((c.line, false));
            }
            None
        }
        DimensionEndpoint::ArcEnd(a) => {
            if let Some(c) = sketch.coincident_lp1_arc_end.iter().find(|c| c.arc == *a) {
                return Some((c.line, true));
            }
            if let Some(c) = sketch.coincident_lp2_arc_end.iter().find(|c| c.arc == *a) {
                return Some((c.line, false));
            }
            None
        }
    }
}

/// Is `ep` constrained to lie on `line` (anywhere along it)? Considers both
/// direct line-endpoint bindings and helper-point bindings (for arc
/// endpoints that have a helper Point bound via coincident_arc_center /
/// coincident_arc_start / coincident_arc_end, with that helper on the line).
fn endpoint_is_on_line(
    sketch: &super::Sketch,
    ep: &DimensionEndpoint,
    line: Ref<Line>,
) -> bool {
    if let Some((l, _)) = endpoint_is_line_endpoint(sketch, ep)
        && l == line { return true; }
    match ep {
        DimensionEndpoint::Point(p) => {
            sketch.point_on_line.iter().any(|c| c.point == *p && c.line == line)
        }
        DimensionEndpoint::ArcCenter(a) => {
            sketch.coincident_arc_center.iter()
                .filter(|c| c.arc == *a)
                .any(|c| sketch.point_on_line.iter().any(|p| p.point == c.point && p.line == line))
        }
        DimensionEndpoint::ArcStart(a) => {
            sketch.coincident_arc_start.iter()
                .filter(|c| c.arc == *a)
                .any(|c| sketch.point_on_line.iter().any(|p| p.point == c.point && p.line == line))
        }
        DimensionEndpoint::ArcEnd(a) => {
            sketch.coincident_arc_end.iter()
                .filter(|c| c.arc == *a)
                .any(|c| sketch.point_on_line.iter().any(|p| p.point == c.point && p.line == line))
        }
        DimensionEndpoint::LineP1(_) | DimensionEndpoint::LineP2(_) => false,
    }
}

/// If the PointPointDistance between `a` and `b` matches the
/// along-line pattern (one is a line endpoint, other is on the same line,
/// and they aren't both endpoints of that line), return a signed along-line
/// symbolic expression. Otherwise return None and the caller falls back to
/// the unsigned sqrt form.
fn try_along_line_symbol(
    a: &DimensionEndpoint,
    b: &DimensionEndpoint,
    sketch: &super::Sketch,
) -> Option<arael_sym::E> {
    use arael_sym::symbol;

    let ba = endpoint_is_line_endpoint(sketch, a);
    let bb = endpoint_is_line_endpoint(sketch, b);

    // Decide which endpoint is the anchor. Reject if both are line endpoints
    // of the same line (that case is just the line length and the sqrt form
    // is fine).
    let (anchor_ep, point_ep, line, anchor_is_p1) =
        if let Some((l, is_p1)) = ba {
            // Reject both-endpoints-of-same-line
            let both_same_line = matches!(bb, Some((lb, _)) if lb == l);
            if !both_same_line && endpoint_is_on_line(sketch, b, l) {
                (a, b, l, is_p1)
            } else if let Some((lb, is_p1_b)) = bb {
                if !matches!(ba, Some((la, _)) if la == lb) && endpoint_is_on_line(sketch, a, lb) {
                    (b, a, lb, is_p1_b)
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else if let Some((l, is_p1)) = bb {
            if endpoint_is_on_line(sketch, a, l) {
                (b, a, l, is_p1)
            } else {
                return None;
            }
        } else {
            return None;
        };

    // Capture sign from current geometry.
    let anchor_pos = dim_endpoint_pos(anchor_ep, sketch);
    let point_pos = dim_endpoint_pos(point_ep, sketch);
    let ln_data = &sketch.lines[line];
    let ldx = ln_data.p2.value.x - ln_data.p1.value.x;
    let ldy = ln_data.p2.value.y - ln_data.p1.value.y;
    let (fx0, fy0) = if anchor_is_p1 { (ldx, ldy) } else { (-ldx, -ldy) };
    let proj = (point_pos.x - anchor_pos.x) * fx0 + (point_pos.y - anchor_pos.y) * fy0;
    let sign_positive = proj >= 0.0;

    // Build signed along-line expression.
    let name = &sketch.lines[line].name;
    let p1x = symbol(&format!("{}.p1.x", name));
    let p1y = symbol(&format!("{}.p1.y", name));
    let p2x = symbol(&format!("{}.p2.x", name));
    let p2y = symbol(&format!("{}.p2.y", name));
    let dx = p2x.clone() - p1x.clone();
    let dy = p2y.clone() - p1y.clone();
    let len = arael_sym::sqrt(dx.clone() * dx.clone() + dy.clone() * dy.clone());
    let (ax, ay) = if anchor_is_p1 {
        (p1x, p1y)
    } else {
        (p2x.clone(), p2y.clone())
    };
    let (fx, fy) = if anchor_is_p1 { (dx.clone(), dy.clone()) } else { (-dx.clone(), -dy.clone()) };
    let (ox, oy) = dim_endpoint_symbol(point_ep, sketch);
    let signed = ((ox - ax) * fx + (oy - ay) * fy) / len;
    Some(if sign_positive { signed } else { -signed })
}
