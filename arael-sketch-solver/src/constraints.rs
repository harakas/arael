// ---------------------------------------------------------------------------
// Cross-constraints (stored in root collections)
// ---------------------------------------------------------------------------

/// Which endpoints are shared between two arcs for tangent constraints.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SharedEndpoint {
    #[default]
    None,
    StartStart,
    StartEnd,
    EndStart,
    EndEnd,
}

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

// -- Arc-Point distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = arc.center.x - point.pos.x; let dy = arc.center.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancearccenterp.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dx = sx - point.pos.x; let dy = sy - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancearcstartp.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dx = ex - point.pos.x; let dy = ey - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancearcendp.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

// -- Arc-Line distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = arc.center.x - line.p1.x; let dy = arc.center.y - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancearccenterl1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = arc.center.x - line.p2.x; let dy = arc.center.y - line.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancearccenterl2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dx = sx - line.p1.x; let dy = sy - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancearcstartl1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dx = sx - line.p2.x; let dy = sy - line.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancearcstartl2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dx = ex - line.p1.x; let dy = ey - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancearcendl1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dx = ex - line.p2.x; let dy = ey - line.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancearcendl2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// -- Arc-Arc distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.center.x - b.center.x; let dy = a.center.y - b.center.y;
    [(sqrt(dx * dx + dy * dy) - distanceaacece.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAACeCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    let dx = a.center.x - bsx; let dy = a.center.y - bsy;
    [(sqrt(dx * dx + dy * dy) - distanceaaces.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAACeS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    let dx = a.center.x - bex; let dy = a.center.y - bey;
    [(sqrt(dx * dx + dy * dy) - distanceaacee.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAACeE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let dx = asx - b.center.x; let dy = asy - b.center.y;
    [(sqrt(dx * dx + dy * dy) - distanceaasce.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAASCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    let dx = asx - bsx; let dy = asy - bsy;
    [(sqrt(dx * dx + dy * dy) - distanceaass.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAASS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    let dx = asx - bex; let dy = asy - bey;
    [(sqrt(dx * dx + dy * dy) - distanceaase.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAASE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let dx = aex - b.center.x; let dy = aey - b.center.y;
    [(sqrt(dx * dx + dy * dy) - distanceaaece.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAAECe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    let dx = aex - bsx; let dy = aey - bsy;
    [(sqrt(dx * dx + dy * dy) - distanceaaes.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAAES {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    let dx = aex - bex; let dy = aey - bey;
    [(sqrt(dx * dx + dy * dy) - distanceaaee.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAAEE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
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
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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

// -- Midpoint on Arc (angular midpoint) --

// Point at angular midpoint of arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mid_angle = (arc.start_angle + arc.end_angle) * 0.5;
    let ctm = cos(mid_angle); let stm = sin(mid_angle);
    let crm = cos(arc.rotation); let srm = sin(arc.rotation);
    let mx = arc.center.x + arc.radius * ctm * crm - arc.radius_b * stm * srm;
    let my = arc.center.y + arc.radius * ctm * srm + arc.radius_b * stm * crm;
    [(point.pos.x - mx) * sketch.constraint_isigma,
     (point.pos.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcPoint {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Line P1 at angular midpoint of arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mid_angle = (arc.start_angle + arc.end_angle) * 0.5;
    let ctm = cos(mid_angle); let stm = sin(mid_angle);
    let crm = cos(arc.rotation); let srm = sin(arc.rotation);
    let mx = arc.center.x + arc.radius * ctm * crm - arc.radius_b * stm * srm;
    let my = arc.center.y + arc.radius * ctm * srm + arc.radius_b * stm * crm;
    [(line.p1.x - mx) * sketch.constraint_isigma,
     (line.p1.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointLP1Arc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Line P2 at angular midpoint of arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mid_angle = (arc.start_angle + arc.end_angle) * 0.5;
    let ctm = cos(mid_angle); let stm = sin(mid_angle);
    let crm = cos(arc.rotation); let srm = sin(arc.rotation);
    let mx = arc.center.x + arc.radius * ctm * crm - arc.radius_b * stm * srm;
    let my = arc.center.y + arc.radius * ctm * srm + arc.radius_b * stm * crm;
    [(line.p2.x - mx) * sketch.constraint_isigma,
     (line.p2.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointLP2Arc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Arc start at angular midpoint of another arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let sx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let sy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let mid_angle = (b.start_angle + b.end_angle) * 0.5;
    let ctm_b = cos(mid_angle); let stm_b = sin(mid_angle);
    let crm_b = cos(b.rotation); let srm_b = sin(b.rotation);
    let mx = b.center.x + b.radius * ctm_b * crm_b - b.radius_b * stm_b * srm_b;
    let my = b.center.y + b.radius * ctm_b * srm_b + b.radius_b * stm_b * crm_b;
    [(sx - mx) * sketch.constraint_isigma,
     (sy - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcStartArc {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Arc end at angular midpoint of another arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let ex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let ey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let mid_angle = (b.start_angle + b.end_angle) * 0.5;
    let ctm_b = cos(mid_angle); let stm_b = sin(mid_angle);
    let crm_b = cos(b.rotation); let srm_b = sin(b.rotation);
    let mx = b.center.x + b.radius * ctm_b * crm_b - b.radius_b * stm_b * srm_b;
    let my = b.center.y + b.radius * ctm_b * srm_b + b.radius_b * stm_b * crm_b;
    [(ex - mx) * sketch.constraint_isigma,
     (ey - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcEndArc {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Point-Arc --

// Point lies on ellipse/circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = point.pos.x - arc.center.x;
    let dy = point.pos.y - arc.center.y;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let u = dx * cr + dy * sr;
    let v = 0.0 - dx * sr + dy * cr;
    [(u * u / (arc.radius * arc.radius) + v * v / (arc.radius_b * arc.radius_b) - 1.0) * sketch.constraint_isigma]
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
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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

// Line tangent to arc/ellipse. Uses perpendicular-distance residual (always active)
// plus a gradient-based residual when the tangent point is a shared endpoint.
// For ellipses, the effective radius along the line normal direction is used.
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Perpendicular distance from center to line = sign * effective_radius (no shared endpoint)
#[arael(constraint(hb, guard = !self.at_p1 && !self.at_p2, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((arc.center.x - line.p1.x) * dy - (arc.center.y - line.p1.y) * dx) / len;
    let nx = 0.0 - dy / len;
    let ny = dx / len;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let nlx = nx * cr + ny * sr;
    let nly = 0.0 - nx * sr + ny * cr;
    let r_eff = sqrt(nlx * nlx * arc.radius * arc.radius + nly * nly * arc.radius_b * arc.radius_b);
    [(dist - tangentla.sign * r_eff) * sketch.constraint_isigma]
}))]
// Directed tangent at shared endpoint p1: constrain the angle between line_dir
// and ellipse gradient to be exactly dir_sign*PI/2.  Unlike the undirected
// dot-product formulation this has a unique zero, preventing line direction reversal.
#[arael(constraint(hb, guard = self.at_p1, {
    let px = line.p1.x - arc.center.x;
    let py = line.p1.y - arc.center.y;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let u = px * cr + py * sr;
    let v = 0.0 - px * sr + py * cr;
    let gx = u / (arc.radius * arc.radius) * cr - v / (arc.radius_b * arc.radius_b) * sr;
    let gy = u / (arc.radius * arc.radius) * sr + v / (arc.radius_b * arc.radius_b) * cr;
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let cross = dx * gy - dy * gx;
    let dot = dx * gx + dy * gy;
    [rad_diff(safe_atan2(cross, dot), tangentla.dir_sign * pi / 2.0) * sketch.constraint_isigma]
}))]
// Directed tangent at shared endpoint p2 (line direction reversed).
#[arael(constraint(hb, guard = self.at_p2, {
    let px = line.p2.x - arc.center.x;
    let py = line.p2.y - arc.center.y;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let u = px * cr + py * sr;
    let v = 0.0 - px * sr + py * cr;
    let gx = u / (arc.radius * arc.radius) * cr - v / (arc.radius_b * arc.radius_b) * sr;
    let gy = u / (arc.radius * arc.radius) * sr + v / (arc.radius_b * arc.radius_b) * cr;
    let dx = line.p1.x - line.p2.x;
    let dy = line.p1.y - line.p2.y;
    let cross = dx * gy - dy * gx;
    let dot = dx * gx + dy * gy;
    [rad_diff(safe_atan2(cross, dot), tangentla.dir_sign * pi / 2.0) * sketch.constraint_isigma]
}))]
pub struct TangentLA {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default = "default_tangent_sign")]
    pub sign: f64,
    #[serde(skip)]
    pub at_p1: bool,
    #[serde(skip)]
    pub at_p2: bool,
    #[serde(skip)]
    pub dir_sign: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

fn default_tangent_sign() -> f64 { 1.0 }

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

// Tangent arc-arc (external tangency).
// Uses effective radii along center-to-center direction.
// Generalizes circles: when rx=ry=r, r_eff = r.
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// No shared endpoint: center-distance = sum of effective radii
#[arael(constraint(hb, guard = self.shared == SharedEndpoint::None, {
    let dx = a.center.x - b.center.x;
    let dy = a.center.y - b.center.y;
    let dist = sqrt(dx * dx + dy * dy);
    let nx = dx / dist;
    let ny = dy / dist;
    let cra = cos(a.rotation);
    let sra = sin(a.rotation);
    let nxa = nx * cra + ny * sra;
    let nya = 0.0 - nx * sra + ny * cra;
    let r_eff_a = sqrt(nxa * nxa * a.radius * a.radius + nya * nya * a.radius_b * a.radius_b);
    let crb = cos(b.rotation);
    let srb = sin(b.rotation);
    let nxb = 0.0 - nx * crb - ny * srb;
    let nyb = nx * srb - ny * crb;
    let r_eff_b = sqrt(nxb * nxb * b.radius * b.radius + nyb * nyb * b.radius_b * b.radius_b);
    [(dist - r_eff_a - r_eff_b) * sketch.constraint_isigma]
}))]
// Shared endpoint a.start = b.start: cross(tangent_a, tangent_b) = 0
#[arael(constraint(hb, guard = self.shared == SharedEndpoint::StartStart, {
    let tax = 0.0 - a.radius * sin(a.start_angle) * cos(a.rotation) - a.radius_b * cos(a.start_angle) * sin(a.rotation);
    let tay = 0.0 - a.radius * sin(a.start_angle) * sin(a.rotation) + a.radius_b * cos(a.start_angle) * cos(a.rotation);
    let tbx = 0.0 - b.radius * sin(b.start_angle) * cos(b.rotation) - b.radius_b * cos(b.start_angle) * sin(b.rotation);
    let tby = 0.0 - b.radius * sin(b.start_angle) * sin(b.rotation) + b.radius_b * cos(b.start_angle) * cos(b.rotation);
    [(tax * tby - tay * tbx) * sketch.constraint_isigma]
}))]
// Shared endpoint a.start = b.end
#[arael(constraint(hb, guard = self.shared == SharedEndpoint::StartEnd, {
    let tax = 0.0 - a.radius * sin(a.start_angle) * cos(a.rotation) - a.radius_b * cos(a.start_angle) * sin(a.rotation);
    let tay = 0.0 - a.radius * sin(a.start_angle) * sin(a.rotation) + a.radius_b * cos(a.start_angle) * cos(a.rotation);
    let tbx = 0.0 - b.radius * sin(b.end_angle) * cos(b.rotation) - b.radius_b * cos(b.end_angle) * sin(b.rotation);
    let tby = 0.0 - b.radius * sin(b.end_angle) * sin(b.rotation) + b.radius_b * cos(b.end_angle) * cos(b.rotation);
    [(tax * tby - tay * tbx) * sketch.constraint_isigma]
}))]
// Shared endpoint a.end = b.start
#[arael(constraint(hb, guard = self.shared == SharedEndpoint::EndStart, {
    let tax = 0.0 - a.radius * sin(a.end_angle) * cos(a.rotation) - a.radius_b * cos(a.end_angle) * sin(a.rotation);
    let tay = 0.0 - a.radius * sin(a.end_angle) * sin(a.rotation) + a.radius_b * cos(a.end_angle) * cos(a.rotation);
    let tbx = 0.0 - b.radius * sin(b.start_angle) * cos(b.rotation) - b.radius_b * cos(b.start_angle) * sin(b.rotation);
    let tby = 0.0 - b.radius * sin(b.start_angle) * sin(b.rotation) + b.radius_b * cos(b.start_angle) * cos(b.rotation);
    [(tax * tby - tay * tbx) * sketch.constraint_isigma]
}))]
// Shared endpoint a.end = b.end
#[arael(constraint(hb, guard = self.shared == SharedEndpoint::EndEnd, {
    let tax = 0.0 - a.radius * sin(a.end_angle) * cos(a.rotation) - a.radius_b * cos(a.end_angle) * sin(a.rotation);
    let tay = 0.0 - a.radius * sin(a.end_angle) * sin(a.rotation) + a.radius_b * cos(a.end_angle) * cos(a.rotation);
    let tbx = 0.0 - b.radius * sin(b.end_angle) * cos(b.rotation) - b.radius_b * cos(b.end_angle) * sin(b.rotation);
    let tby = 0.0 - b.radius * sin(b.end_angle) * sin(b.rotation) + b.radius_b * cos(b.end_angle) * cos(b.rotation);
    [(tax * tby - tay * tbx) * sketch.constraint_isigma]
}))]
pub struct TangentAA {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[arael(skip)]
    #[serde(default)]
    pub shared: SharedEndpoint,
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
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
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
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
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
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
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
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
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
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
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
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
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
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
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
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
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
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
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

// Line p1 lies on ellipse/circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p1.x - arc.center.x;
    let dy = line.p1.y - arc.center.y;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let u = dx * cr + dy * sr;
    let v = 0.0 - dx * sr + dy * cr;
    [(u * u / (arc.radius * arc.radius) + v * v / (arc.radius_b * arc.radius_b) - 1.0) * sketch.constraint_isigma]
}))]
pub struct LineP1OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Line p2 lies on ellipse/circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - arc.center.x;
    let dy = line.p2.y - arc.center.y;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let u = dx * cr + dy * sr;
    let v = 0.0 - dx * sr + dy * cr;
    [(u * u / (arc.radius * arc.radius) + v * v / (arc.radius_b * arc.radius_b) - 1.0) * sketch.constraint_isigma]
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

// Line endpoint p1 to line (signed perpendicular distance)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.p2.x - b.p1.x;
    let dy = b.p2.y - b.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((a.p1.x - b.p1.x) * dy - (a.p1.y - b.p1.y) * dx) / len;
    [(dist - distancelp1l.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP1L {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// Line endpoint p2 to line (signed perpendicular distance)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.p2.x - b.p1.x;
    let dy = b.p2.y - b.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((a.p2.x - b.p1.x) * dy - (a.p2.y - b.p1.y) * dx) / len;
    [(dist - distancelp2l.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP2L {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// Arc center to line (signed perpendicular distance)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((arc.center.x - line.p1.x) * dy - (arc.center.y - line.p1.y) * dx) / len;
    [(dist - distancearccenterl.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterL {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// Arc start to line (signed perpendicular distance)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dist = ((sx - line.p1.x) * dy - (sy - line.p1.y) * dx) / len;
    [(dist - distancearcstartl.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartL {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// Arc end to line (signed perpendicular distance)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    let dist = ((ex - line.p1.x) * dy - (ey - line.p1.y) * dx) / len;
    [(dist - distancearcendl.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndL {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// ---------------------------------------------------------------------------
// Symmetry: two points about a mirror line
// ---------------------------------------------------------------------------

/// Two points forced symmetric about a mirror line.
/// Residual: reflect a across line, compare to c.
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len2 = dx * dx + dy * dy;
    // Signed perpendicular distance of a from line (unnormalized)
    let da = (a.pos.x - line.p1.x) * dy - (a.pos.y - line.p1.y) * dx;
    // Reflect a across line
    let rx = a.pos.x - 2.0 * da * dy / len2;
    let ry = a.pos.y + 2.0 * da * dx / len2;
    [(rx - c.pos.x) * sketch.constraint_isigma,
     (ry - c.pos.y) * sketch.constraint_isigma]
}))]
pub struct SymmetryPP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub c: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: TripletBlock<f64>,
}

// Symmetry of two arcs/ellipses about a mirror line.
// Guarded: circle path uses 3 residuals (center + radius), ellipse path
// adds radius_b equality + rotation reflection (5 residuals).
// Cannot unify because radius_b is a real parameter for circles (equality
// constraint), so the extra residuals affect DOF counting.
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = !a.is_ellipse && !c.is_ellipse, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len2 = dx * dx + dy * dy;
    let da = (a.center.x - line.p1.x) * dy - (a.center.y - line.p1.y) * dx;
    let rx = a.center.x - 2.0 * da * dy / len2;
    let ry = a.center.y + 2.0 * da * dx / len2;
    [(rx - c.center.x) * sketch.constraint_isigma,
     (ry - c.center.y) * sketch.constraint_isigma,
     (a.radius - c.radius) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = a.is_ellipse || c.is_ellipse, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len2 = dx * dx + dy * dy;
    let da = (a.center.x - line.p1.x) * dy - (a.center.y - line.p1.y) * dx;
    let rx = a.center.x - 2.0 * da * dy / len2;
    let ry = a.center.y + 2.0 * da * dx / len2;
    let alpha = atan2(dy, dx);
    let reflected_rot = 2.0 * alpha - a.rotation;
    [(rx - c.center.x) * sketch.constraint_isigma,
     (ry - c.center.y) * sketch.constraint_isigma,
     (a.radius - c.radius) * sketch.constraint_isigma,
     (a.radius_b - c.radius_b) * sketch.constraint_isigma,
     sin(reflected_rot - c.rotation) * sketch.constraint_isigma]
}))]
pub struct SymmetryAA {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub c: Ref<Arc>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: TripletBlock<f64>,
}

// ---------------------------------------------------------------------------
// Axis distance (horizontal/vertical unified via guard flag)
// ---------------------------------------------------------------------------

// -- Line-Line --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(a.p1.x - b.p1.x - axisdistancell11.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(a.p1.y - b.p1.y - axisdistancell11.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceLL11 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(a.p1.x - b.p2.x - axisdistancell12.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(a.p1.y - b.p2.y - axisdistancell12.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceLL12 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(a.p2.x - b.p1.x - axisdistancell21.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(a.p2.y - b.p1.y - axisdistancell21.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceLL21 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(a.p2.x - b.p2.x - axisdistancell22.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(a.p2.y - b.p2.y - axisdistancell22.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceLL22 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// -- Line-Point --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(line.p1.x - point.pos.x - axisdistancelp1.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(line.p1.y - point.pos.y - axisdistancelp1.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceLP1 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(line.p2.x - point.pos.x - axisdistancelp2.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(line.p2.y - point.pos.y - axisdistancelp2.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceLP2 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

// -- Arc-Point --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(arc.center.x - point.pos.x - axisdistancearccenterp.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(arc.center.y - point.pos.y - axisdistancearccenterp.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcCenterP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    [(sx - point.pos.x - axisdistancearcstartp.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    [(sy - point.pos.y - axisdistancearcstartp.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcStartP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    [(ex - point.pos.x - axisdistancearcendp.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    [(ey - point.pos.y - axisdistancearcendp.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcEndP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

// -- Arc-Line --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(arc.center.x - line.p1.x - axisdistancearccenterl1.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(arc.center.y - line.p1.y - axisdistancearccenterl1.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcCenterL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(arc.center.x - line.p2.x - axisdistancearccenterl2.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(arc.center.y - line.p2.y - axisdistancearccenterl2.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcCenterL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    [(sx - line.p1.x - axisdistancearcstartl1.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    [(sy - line.p1.y - axisdistancearcstartl1.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcStartL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sx = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    [(sx - line.p2.x - axisdistancearcstartl2.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct = cos(arc.start_angle); let st = sin(arc.start_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let sy = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    [(sy - line.p2.y - axisdistancearcstartl2.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcStartL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    [(ex - line.p1.x - axisdistancearcendl1.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    [(ey - line.p1.y - axisdistancearcendl1.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcEndL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ex = arc.center.x + arc.radius * ct * cr - arc.radius_b * st * sr;
    [(ex - line.p2.x - axisdistancearcendl2.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct = cos(arc.end_angle); let st = sin(arc.end_angle);
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let ey = arc.center.y + arc.radius * ct * sr + arc.radius_b * st * cr;
    [(ey - line.p2.y - axisdistancearcendl2.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcEndL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// -- Arc-Arc --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    [(a.center.x - b.center.x - axisdistanceaacece.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    [(a.center.y - b.center.y - axisdistanceaacece.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAACeCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    [(a.center.x - bsx - axisdistanceaaces.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    [(a.center.y - bsy - axisdistanceaaces.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAACeS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    [(a.center.x - bex - axisdistanceaacee.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    [(a.center.y - bey - axisdistanceaacee.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAACeE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    [(asx - b.center.x - axisdistanceaasce.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    [(asy - b.center.y - axisdistanceaasce.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAASCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    [(asx - bsx - axisdistanceaass.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    [(asy - bsy - axisdistanceaass.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAASS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asx = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    [(asx - bex - axisdistanceaase.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_a = cos(a.start_angle); let st_a = sin(a.start_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let asy = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    [(asy - bey - axisdistanceaase.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAASE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    [(aex - b.center.x - axisdistanceaaece.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    [(aey - b.center.y - axisdistanceaaece.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAAECe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsx = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    [(aex - bsx - axisdistanceaaes.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.start_angle); let st_b = sin(b.start_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bsy = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    [(aey - bsy - axisdistanceaaes.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAAES {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aex = a.center.x + a.radius * ct_a * cr_a - a.radius_b * st_a * sr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bex = b.center.x + b.radius * ct_b * cr_b - b.radius_b * st_b * sr_b;
    [(aex - bex - axisdistanceaaee.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ct_a = cos(a.end_angle); let st_a = sin(a.end_angle);
    let cr_a = cos(a.rotation); let sr_a = sin(a.rotation);
    let aey = a.center.y + a.radius * ct_a * sr_a + a.radius_b * st_a * cr_a;
    let ct_b = cos(b.end_angle); let st_b = sin(b.end_angle);
    let cr_b = cos(b.rotation); let sr_b = sin(b.rotation);
    let bey = b.center.y + b.radius * ct_b * sr_b + b.radius_b * st_b * cr_b;
    [(aey - bey - axisdistanceaaee.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAAEE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

