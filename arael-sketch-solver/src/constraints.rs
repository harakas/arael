// ---------------------------------------------------------------------------
// Cross-constraints (stored in root collections)
// ---------------------------------------------------------------------------

// -- Point-Point --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [a.pos.x - b.pos.x, a.pos.y - b.pos.y]
}))]
pub struct CoincidentPP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.pos.x - b.pos.x;
    let dy = a.pos.y - b.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancepp.distance) * sketch.constraint_isigma]
}))]
pub struct DistancePP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    pub distance: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

// -- Line-Line endpoint distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p1.x - b.p1.x; let dy = a.p1.y - b.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancell11.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL11 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p1.x - b.p2.x; let dy = a.p1.y - b.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancell12.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL12 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - b.p1.x; let dy = a.p2.y - b.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancell21.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL21 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - b.p2.x; let dy = a.p2.y - b.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancell22.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL22 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// -- Line endpoint to Point distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p1.x - point.pos.x; let dy = line.p1.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancelp1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP1 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - point.pos.x; let dy = line.p2.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancelp2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP2 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.pos.x - b.pos.x - horizontaldistancepp.distance) * sketch.constraint_isigma]
}))]
pub struct HorizontalDistancePP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    pub distance: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.pos.y - b.pos.y - verticaldistancepp.distance) * sketch.constraint_isigma]
}))]
pub struct VerticalDistancePP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    pub distance: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

// -- Point-Line --

// Point lies on infinite line through p1-p2 (cross product = 0)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    [((point.pos.x - line.p1.x) * dy - (point.pos.y - line.p1.y) * dx) / len
     * sketch.constraint_isigma]
}))]
pub struct PointOnLine {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// Point at midpoint of line segment
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mx = (line.p1.x + line.p2.x) * 0.5;
    let my = (line.p1.y + line.p2.y) * 0.5;
    [(point.pos.x - mx) * sketch.constraint_isigma,
     (point.pos.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointConstraint {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// Line P1 at midpoint of another line
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mx = (target.p1.x + target.p2.x) * 0.5;
    let my = (target.p1.y + target.p2.y) * 0.5;
    [(line.p1.x - mx) * sketch.constraint_isigma,
     (line.p1.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointLP1 {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.lines)]
    pub target: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Line P2 at midpoint of another line
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mx = (target.p1.x + target.p2.x) * 0.5;
    let my = (target.p1.y + target.p2.y) * 0.5;
    [(line.p2.x - mx) * sketch.constraint_isigma,
     (line.p2.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointLP2 {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.lines)]
    pub target: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Arc start point at midpoint of line
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    let mx = (line.p1.x + line.p2.x) * 0.5;
    let my = (line.p1.y + line.p2.y) * 0.5;
    [(sx - mx) * sketch.constraint_isigma,
     (sy - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcStart {
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Line>,
}

// Arc end point at midpoint of line
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    let mx = (line.p1.x + line.p2.x) * 0.5;
    let my = (line.p1.y + line.p2.y) * 0.5;
    [(ex - mx) * sketch.constraint_isigma,
     (ey - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcEnd {
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Line>,
}

// -- Point-Arc --

// Point lies on circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = point.pos.x - arc.center.x;
    let dy = point.pos.y - arc.center.y;
    [(sqrt(dx * dx + dy * dy) - arc.radius) * sketch.constraint_isigma]
}))]
pub struct PointOnArc {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc center
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(point.pos.x - arc.center.x) * sketch.constraint_isigma,
     (point.pos.y - arc.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenter {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc start endpoint (center + radius * [cos(sa), sin(sa)])
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    [(point.pos.x - sx) * sketch.constraint_isigma,
     (point.pos.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStart {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc end endpoint (center + radius * [cos(ea), sin(ea)])
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    [(point.pos.x - ex) * sketch.constraint_isigma,
     (point.pos.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEnd {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// -- Line-Line --

// Parallel: cross product of direction vectors = 0
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    let len1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let len2 = sqrt(dx2 * dx2 + dy2 * dy2);
    [(dx1 * dy2 - dy1 * dx2) / (len1 * len2) * sketch.constraint_isigma]
}))]
pub struct Parallel {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Perpendicular: dot product of direction vectors = 0
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    let len1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let len2 = sqrt(dx2 * dx2 + dy2 * dy2);
    [(dx1 * dx2 + dy1 * dy2) / (len1 * len2) * sketch.constraint_isigma]
}))]
pub struct Perpendicular {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Collinear: line2 endpoints both lie on infinite line of line1
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - a.p1.x;
    let dy = a.p2.y - a.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let cross1 = ((b.p1.x - a.p1.x) * dy - (b.p1.y - a.p1.y) * dx) / len;
    let cross2 = ((b.p2.x - a.p1.x) * dy - (b.p2.y - a.p1.y) * dx) / len;
    [cross1 * sketch.constraint_isigma, cross2 * sketch.constraint_isigma]
}))]
pub struct Collinear {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Equal length
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    [(sqrt(dx1*dx1 + dy1*dy1) - sqrt(dx2*dx2 + dy2*dy2)) * sketch.constraint_isigma]
}))]
pub struct EqualLength {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Angle between lines
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    [(atan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2) - angleconstraint.angle)
     * sketch.constraint_isigma]
}))]
pub struct AngleConstraint {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    pub angle: f64,  // target angle in radians
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line-Point (endpoint coincidence) --

// Line p1 coincides with standalone point
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p1.x - point.pos.x) * sketch.constraint_isigma,
     (line.p1.y - point.pos.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1 {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Point>,
}

// Line p2 coincides with standalone point
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p2.x - point.pos.x) * sketch.constraint_isigma,
     (line.p2.y - point.pos.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2 {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Point>,
}

// -- Line-Line endpoint coincidence (4 variants for endpoint combos) --

// a.p1 == b.p1
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p1.x - b.p1.x) * sketch.constraint_isigma,
     (a.p1.y - b.p1.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL11 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p1 == b.p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p1.x - b.p2.x) * sketch.constraint_isigma,
     (a.p1.y - b.p2.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL12 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p2 == b.p1  (most common: end of line a -> start of line b)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p2.x - b.p1.x) * sketch.constraint_isigma,
     (a.p2.y - b.p1.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL21 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p2 == b.p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p2.x - b.p2.x) * sketch.constraint_isigma,
     (a.p2.y - b.p2.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL22 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line-Arc --

// Line tangent to arc (distance from center to infinite line = radius)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((arc.center.x - line.p1.x) * dy - (arc.center.y - line.p1.y) * dx) / len;
    [(dist * dist - arc.radius * arc.radius) * sketch.constraint_isigma]
}))]
pub struct TangentLA {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Arc-Arc --

// Concentric: centers coincide
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.center.x - b.center.x) * sketch.constraint_isigma,
     (a.center.y - b.center.y) * sketch.constraint_isigma]
}))]
pub struct Concentric {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Equal radius
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.radius - b.radius) * sketch.constraint_isigma]
}))]
pub struct EqualRadius {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Tangent arc-arc (external: dist between centers = r1 + r2)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.center.x - b.center.x;
    let dy = a.center.y - b.center.y;
    let dist = sqrt(dx * dx + dy * dy);
    let target = a.radius + b.radius;
    [(dist - target) * sketch.constraint_isigma]
}))]
pub struct TangentAA {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Line endpoint <-> Arc point coincidence --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p1.x - arc.center.x) * sketch.constraint_isigma,
     (line.p1.y - arc.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcCenter {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p2.x - arc.center.x) * sketch.constraint_isigma,
     (line.p2.y - arc.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcCenter {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    [(line.p1.x - sx) * sketch.constraint_isigma,
     (line.p1.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcStart {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    [(line.p2.x - sx) * sketch.constraint_isigma,
     (line.p2.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcStart {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    [(line.p1.x - ex) * sketch.constraint_isigma,
     (line.p1.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcEnd {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    [(line.p2.x - ex) * sketch.constraint_isigma,
     (line.p2.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcEnd {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Arc-Arc endpoint coincidence --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bsx = b.center.x + b.radius * cos(b.start_angle);
    let bsy = b.center.y + b.radius * sin(b.start_angle);
    [(a.center.x - bsx) * sketch.constraint_isigma,
     (a.center.y - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenterStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bex = b.center.x + b.radius * cos(b.end_angle);
    let bey = b.center.y + b.radius * sin(b.end_angle);
    [(a.center.x - bex) * sketch.constraint_isigma,
     (a.center.y - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenterEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = a.center.x + a.radius * cos(a.start_angle);
    let asy = a.center.y + a.radius * sin(a.start_angle);
    [(asx - b.center.x) * sketch.constraint_isigma,
     (asy - b.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartCenter {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = a.center.x + a.radius * cos(a.end_angle);
    let aey = a.center.y + a.radius * sin(a.end_angle);
    [(aex - b.center.x) * sketch.constraint_isigma,
     (aey - b.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndCenter {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = a.center.x + a.radius * cos(a.start_angle);
    let asy = a.center.y + a.radius * sin(a.start_angle);
    let bsx = b.center.x + b.radius * cos(b.start_angle);
    let bsy = b.center.y + b.radius * sin(b.start_angle);
    [(asx - bsx) * sketch.constraint_isigma,
     (asy - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = a.center.x + a.radius * cos(a.start_angle);
    let asy = a.center.y + a.radius * sin(a.start_angle);
    let bex = b.center.x + b.radius * cos(b.end_angle);
    let bey = b.center.y + b.radius * sin(b.end_angle);
    [(asx - bex) * sketch.constraint_isigma,
     (asy - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = a.center.x + a.radius * cos(a.end_angle);
    let aey = a.center.y + a.radius * sin(a.end_angle);
    let bsx = b.center.x + b.radius * cos(b.start_angle);
    let bsy = b.center.y + b.radius * sin(b.start_angle);
    [(aex - bsx) * sketch.constraint_isigma,
     (aey - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = a.center.x + a.radius * cos(a.end_angle);
    let aey = a.center.y + a.radius * sin(a.end_angle);
    let bex = b.center.x + b.radius * cos(b.end_angle);
    let bey = b.center.y + b.radius * sin(b.end_angle);
    [(aex - bex) * sketch.constraint_isigma,
     (aey - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Line endpoint on line --

// Line a's p1 lies on infinite line through b's p1-p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.p2.x - b.p1.x;
    let dy = b.p2.y - b.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    [((a.p1.x - b.p1.x) * dy - (a.p1.y - b.p1.y) * dx) / len
     * sketch.constraint_isigma]
}))]
pub struct LineP1OnLine {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Line a's p2 lies on infinite line through b's p1-p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.p2.x - b.p1.x;
    let dy = b.p2.y - b.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    [((a.p2.x - b.p1.x) * dy - (a.p2.y - b.p1.y) * dx) / len
     * sketch.constraint_isigma]
}))]
pub struct LineP2OnLine {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line endpoint on arc --

// Line p1 lies on circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p1.x - arc.center.x;
    let dy = line.p1.y - arc.center.y;
    [(sqrt(dx * dx + dy * dy) - arc.radius) * sketch.constraint_isigma]
}))]
pub struct LineP1OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Line p2 lies on circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - arc.center.x;
    let dy = line.p2.y - arc.center.y;
    [(sqrt(dx * dx + dy * dy) - arc.radius) * sketch.constraint_isigma]
}))]
pub struct LineP2OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Symmetry (3-entity) --

// Symmetry of 3 lines: from each B endpoint, cast a ray and intersect
// with A and C. Signed ray parameters must sum to zero.
// `use_normal_ray` selects ray direction: true = B's normal (default),
// false = B's direction (for when A/C are nearly perpendicular to B).
// Set at constraint creation based on initial geometry.
// Uses TripletBlock for 3-entity Hessian accumulation.
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bn = (b.p2 - b.p1).across();
    let bd = b.p2 - b.p1;
    let ad = a.p2 - a.p1;
    let cd = c.p2 - c.p1;
    // Normalize by direction vector lengths for well-conditioned residuals
    let alen = ad.norm();
    let clen = cd.norm();
    let blen = bd.norm();
    // Ray along B's normal: equidistant perpendicular to B
    let bna = bn.cross(ad) / (blen * alen);
    let bnc = bn.cross(cd) / (blen * clen);
    let rn1 = (a.p1 - b.p1).cross(ad) / alen * bnc + (c.p1 - b.p1).cross(cd) / clen * bna;
    let rn2 = (a.p1 - b.p2).cross(ad) / alen * bnc + (c.p1 - b.p2).cross(cd) / clen * bna;
    // Intersection of A and C lies on line B (normalized)
    let adc = ad.cross(cd) / (alen * clen);
    let r3 = (a.p1 - b.p1).cross(bd) / blen * adc + ad.cross(bd) / (alen * blen) * (c.p1 - a.p1).cross(cd) / clen;
    [rn1 * sketch.constraint_isigma,
     rn2 * sketch.constraint_isigma,
     r3 * sketch.constraint_isigma]
}))]
pub struct SymmetryLL {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[arael(ref = root.lines)]
    pub c: Ref<Line>,
    #[serde(skip)]
    pub hb: TripletBlock<f64>,
}

// -- Distance Point-Line --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((point.pos.x - line.p1.x) * dy - (point.pos.y - line.p1.y) * dx) / len;
    [(dist - distancepl.distance) * sketch.constraint_isigma]
}))]
pub struct DistancePL {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    pub distance: f64,  // signed distance (positive = left of line direction)
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}
