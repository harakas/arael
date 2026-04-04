// ---------------------------------------------------------------------------
// Line/arc visual style
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
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
    #[arael(skip)]
    pub has_fix_x: bool,
    pub fix_x: f64,
    #[arael(skip)]
    pub has_fix_y: bool,
    pub fix_y: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct LineConstraints {
    #[arael(skip)]
    pub horizontal: bool,
    #[arael(skip)]
    pub vertical: bool,
    #[arael(skip)]
    pub has_length: bool,
    pub length: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct ArcConstraints {
    #[arael(skip)]
    pub has_target_radius: bool,
    pub target_radius: f64,
    #[arael(skip)]
    pub has_target_sweep: bool,
    pub target_sweep: f64,
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * sketch.drift_isigma, d.y * sketch.drift_isigma]
}))]
// Fix X coordinate
#[arael(constraint(hb, guard = self.constraints.has_fix_x, {
    [(point.pos.x - point.constraints.fix_x) * sketch.constraint_isigma]
}))]
// Fix Y coordinate
#[arael(constraint(hb, guard = self.constraints.has_fix_y, {
    [(point.pos.y - point.constraints.fix_y) * sketch.constraint_isigma]
}))]
pub struct Point {
    pub pos: Param<vect2d>,
    pub constraints: PointConstraints,
    #[arael(skip)]
    pub helper: bool,
    #[arael(skip)]
    pub name: String,
    #[serde(skip)]
    pub hb: SelfBlock<Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer on both endpoints
#[arael(constraint(hb, {
    let d1 = line.p1 - line.p1_value;
    let d2 = line.p2 - line.p2_value;
    [d1.x * sketch.drift_isigma, d1.y * sketch.drift_isigma,
     d2.x * sketch.drift_isigma, d2.y * sketch.drift_isigma]
}))]
// Drift: weak regularizer on length
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let dx0 = line.p2_value.x - line.p1_value.x;
    let dy0 = line.p2_value.y - line.p1_value.y;
    [(sqrt(dx * dx + dy * dy) - sqrt(dx0 * dx0 + dy0 * dy0)) * sketch.drift_isigma]
}))]
// Horizontal: p1.y == p2.y
#[arael(constraint(hb, guard = self.constraints.horizontal, {
    [(line.p1.y - line.p2.y) * sketch.constraint_isigma]
}))]
// Vertical: p1.x == p2.x
#[arael(constraint(hb, guard = self.constraints.vertical, {
    [(line.p1.x - line.p2.x) * sketch.constraint_isigma]
}))]
// Length
#[arael(constraint(hb, guard = self.constraints.has_length, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - line.constraints.length) * sketch.constraint_isigma]
}))]
pub struct Line {
    pub p1: Param<vect2d>,
    pub p2: Param<vect2d>,
    pub constraints: LineConstraints,
    #[arael(skip)]
    pub style: LineStyle,
    #[arael(skip)]
    pub name: String,
    #[serde(skip)]
    pub hb: SelfBlock<Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer on center, radius, angles
#[arael(constraint(hb, {
    let dc = arc.center - arc.center_value;
    let dr = arc.radius - arc.radius_value;
    let dsa = arc.start_angle - arc.start_angle_value;
    let dea = arc.end_angle - arc.end_angle_value;
    [dc.x * sketch.drift_isigma, dc.y * sketch.drift_isigma,
     dr * sketch.drift_isigma,
     dsa * sketch.drift_isigma, dea * sketch.drift_isigma]
}))]
// Target radius
#[arael(constraint(hb, guard = self.constraints.has_target_radius, {
    [(arc.radius - arc.constraints.target_radius) * sketch.constraint_isigma]
}))]
// Target sweep angle (end_angle - start_angle)
#[arael(constraint(hb, guard = self.constraints.has_target_sweep, {
    [(arc.end_angle - arc.start_angle - arc.constraints.target_sweep) * sketch.constraint_isigma]
}))]
pub struct Arc {
    pub center: Param<vect2d>,
    pub radius: Param<f64>,
    pub start_angle: Param<f64>,
    pub end_angle: Param<f64>,
    /// Full circle (true) vs partial arc (false). When true, start/end
    /// angles are fixed and excluded from optimization.
    #[arael(skip)]
    pub closed: bool,
    #[arael(skip)]
    pub style: LineStyle,
    #[arael(skip)]
    pub name: String,
    pub constraints: ArcConstraints,
    #[serde(skip)]
    pub hb: SelfBlock<Arc>,
}
