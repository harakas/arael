use arael::utils::Float as _;

// ---------------------------------------------------------------------------
// Line/arc visual style
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[arael::model]
pub enum LineStyle {
    #[default]
    Solid,
    Dashed,
    DashDot,
}

impl LineStyle {
    pub fn next(self) -> Self {
        match self {
            LineStyle::Solid => LineStyle::Dashed,
            LineStyle::Dashed => LineStyle::DashDot,
            LineStyle::DashDot => LineStyle::Solid,
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "solid" => Some(Self::Solid),
            "dashed" => Some(Self::Dashed),
            "dashdot" | "dash_dot" | "dash-dot" => Some(Self::DashDot),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Solid => "solid",
            Self::Dashed => "dashed",
            Self::DashDot => "dashdot",
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint data stored on entities (for guarded self-constraints)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct PointConstraints {
    pub has_fix_x: bool,
    pub fix_x: f64,
    pub has_fix_y: bool,
    pub fix_y: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct LineConstraints {
    pub horizontal: bool,
    pub vertical: bool,
    pub has_length: bool,
    pub length: f64,
    #[serde(default)]
    pub has_angle: bool,
    #[serde(default)]
    pub target_angle: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct ArcConstraints {
    pub has_target_radius: bool,
    pub target_radius: f64,
    #[serde(default)]
    pub has_target_radius_b: bool,
    #[serde(default)]
    pub target_radius_b: f64,
    #[serde(default)]
    pub has_target_sweep: bool,
    #[serde(default)]
    pub target_sweep: f64,
    #[serde(default = "default_sweep_sign")]
    pub sweep_sign: f64,
}

fn default_sweep_sign() -> f64 { 1.0 }
fn default_ccw() -> bool { true }
fn default_param_zero() -> arael::model::Param<f64> { arael::model::Param::fixed(0.0) }

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer
#[arael(constraint(hb, name = "drift", {
    let d = point.pos - point.pos_value;
    [d.x * sketch.drift_isigma, d.y * sketch.drift_isigma]
}))]
// Fix X coordinate
#[arael(constraint(hb, guard = self.constraints.has_fix_x, name = "fix_x", {
    [(point.pos.x - point.constraints.fix_x) * sketch.constraint_isigma]
}))]
// Fix Y coordinate
#[arael(constraint(hb, guard = self.constraints.has_fix_y, name = "fix_y", {
    [(point.pos.y - point.constraints.fix_y) * sketch.constraint_isigma]
}))]
pub struct Point {
    pub pos: Param<vect2d>,
    pub constraints: PointConstraints,
    pub helper: bool,
    #[serde(default)]
    pub quiet: bool,
    pub name: String,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: SelfBlock<Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer on both endpoints
#[arael(constraint(hb, name = "drift", {
    let d1 = line.p1 - line.p1_value;
    let d2 = line.p2 - line.p2_value;
    [d1.x * sketch.drift_isigma, d1.y * sketch.drift_isigma,
     d2.x * sketch.drift_isigma, d2.y * sketch.drift_isigma]
}))]
// Drift: weak regularizer on length (epsilon avoids sqrt singularity at zero length)
#[arael(constraint(hb, name = "drift_length", {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let dx0 = line.p2_value.x - line.p1_value.x;
    let dy0 = line.p2_value.y - line.p1_value.y;
    [(sqrt(dx * dx + dy * dy + 1e-6) - sqrt(dx0 * dx0 + dy0 * dy0 + 1e-6)) * sketch.drift_isigma]
}))]
// Drift: weak regularizer on angle (safe_atan2 avoids NaN at zero length)
#[arael(constraint(hb, name = "drift_angle", {
    let angle = safe_atan2(line.p2.y - line.p1.y, line.p2.x - line.p1.x);
    let angle0 = safe_atan2(line.p2_value.y - line.p1_value.y, line.p2_value.x - line.p1_value.x);
    [rad_diff(angle, angle0) * sketch.drift_isigma]
}))]
// Horizontal: p1.y == p2.y
#[arael(constraint(hb, guard = self.constraints.horizontal, name = "horizontal", {
    [(line.p1.y - line.p2.y) * sketch.constraint_isigma]
}))]
// Vertical: p1.x == p2.x
#[arael(constraint(hb, guard = self.constraints.vertical, name = "vertical", {
    [(line.p1.x - line.p2.x) * sketch.constraint_isigma]
}))]
// Length
#[arael(constraint(hb, guard = self.constraints.has_length, name = "length_target", {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - line.constraints.length) * sketch.constraint_isigma]
}))]
// Angle from x-axis
#[arael(constraint(hb, guard = self.constraints.has_angle, name = "angle_target", {
    [(atan2(line.p2.y - line.p1.y, line.p2.x - line.p1.x) - line.constraints.target_angle) * sketch.constraint_isigma]
}))]
// Soft minimum length via squared heaviside penalty.
// Prevents line from collapsing to zero length (which makes direction undefined
// and breaks tangent/angle constraints). Same pattern as arc minimum radius.
// Uses length^2 directly to avoid sqrt singularity at zero.
#[arael(constraint(hb, name = "min_length", {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let d = sketch.min_length * sketch.min_length - (dx * dx + dy * dy);
    [heaviside(d) * d * sketch.constraint_isigma * sketch.constraint_isigma]
}))]
pub struct Line {
    pub p1: Param<vect2d>,
    pub p2: Param<vect2d>,
    pub constraints: LineConstraints,
    pub style: LineStyle,
    #[serde(default)]
    pub construction: bool,
    #[serde(default)]
    pub quiet: bool,
    pub name: String,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: SelfBlock<Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer on center, radii, rotation, angles
#[arael(constraint(hb, name = "drift", {
    let dc = arc.center - arc.center_value;
    let dr = arc.radius - arc.radius_value;
    let drb = arc.radius_b - arc.radius_b_value;
    let drot = arc.rotation - arc.rotation_value;
    let dsa = arc.start_angle - arc.start_angle_value;
    let dea = arc.end_angle - arc.end_angle_value;
    [dc.x * sketch.drift_isigma, dc.y * sketch.drift_isigma,
     dr * sketch.drift_isigma, drb * sketch.drift_isigma,
     drot * sketch.drift_isigma,
     dsa * sketch.drift_isigma, dea * sketch.drift_isigma]
}))]
// Target radius (semi-major axis)
#[arael(constraint(hb, guard = self.constraints.has_target_radius, name = "radius_target", {
    [(arc.radius - arc.constraints.target_radius) * sketch.constraint_isigma]
}))]
// Target radius_b (semi-minor axis, for ellipses)
#[arael(constraint(hb, guard = self.constraints.has_target_radius_b, name = "radius_b_target", {
    [(arc.radius_b - arc.constraints.target_radius_b) * sketch.constraint_isigma]
}))]
// For non-ellipse arcs: radius_b must equal radius (rotation is Param::fixed so no constraint needed)
#[arael(constraint(hb, guard = !self.is_ellipse, name = "radius_b_eq_radius", {
    [(arc.radius_b - arc.radius) * sketch.constraint_isigma]
}))]
// EXPERIMENTAL: soft minimum radius via squared heaviside penalty.
// Prevents radius from going below 0.001. The squared penalty is smooth
// at the transition (value and gradient both zero at threshold).
// A proper solution would be bound-constrained optimization in the framework.
#[arael(constraint(hb, name = "min_radius", {
    let d = sketch.min_length - arc.radius;
    [heaviside(d) * d * d * sketch.constraint_isigma * sketch.constraint_isigma]
}))]
// EXPERIMENTAL: same for radius_b on ellipses.
#[arael(constraint(hb, guard = self.is_ellipse, name = "min_radius_b", {
    let d = sketch.min_length - arc.radius_b;
    [heaviside(d) * d * d * sketch.constraint_isigma * sketch.constraint_isigma]
}))]
// Target sweep angle (multiplied by radius for position-equivalent scaling)
#[arael(constraint(hb, guard = self.constraints.has_target_sweep, name = "sweep", {
    [(arc.end_angle - arc.start_angle - arc.constraints.sweep_sign * arc.constraints.target_sweep) * arc.radius * sketch.constraint_isigma]
}))]
pub struct Arc {
    pub center: Param<vect2d>,
    pub radius: Param<f64>,
    #[serde(default = "default_param_zero")]
    pub radius_b: Param<f64>,
    #[serde(default = "default_param_zero")]
    pub rotation: Param<f64>,
    pub start_angle: Param<f64>,
    pub end_angle: Param<f64>,
    /// Full circle/ellipse (true) vs partial arc (false). When true, start/end
    /// angles are fixed and excluded from optimization.
    pub closed: bool,
    /// True for elliptic arcs/ellipses (radius_b and rotation are free params).
    /// False for circular arcs/circles (radius_b fixed to radius, rotation fixed to 0).
    #[serde(default)]
    pub is_ellipse: bool,
    /// Arc direction: true = counter-clockwise from start to end,
    /// false = clockwise. Determined at creation from the midpoint.
    #[serde(default = "default_ccw")]
    pub ccw: bool,
    pub style: LineStyle,
    #[serde(default)]
    pub construction: bool,
    #[serde(default)]
    pub quiet: bool,
    pub name: String,
    pub constraints: ArcConstraints,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: SelfBlock<Arc>,
}
