// ---------------------------------------------------------------------------
// Cross-constraints (stored in root collections)
// ---------------------------------------------------------------------------

/// Which endpoints are shared between two arcs for tangent constraints.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[arael::model]
pub enum SharedEndpoint {
    #[default]
    None,
    StartStart,
    StartEnd,
    EndStart,
    EndEnd,
}

// -- Point-Point --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [a.pos.x - b.pos.x, a.pos.y - b.pos.y]
}))]
pub struct CoincidentPP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

// -- Line-Line endpoint distance --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p1.x - b.p1.x; let dy = a.p1.y - b.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancell11.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL11 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p1.x - b.p2.x; let dy = a.p1.y - b.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancell12.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL12 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - b.p1.x; let dy = a.p2.y - b.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancell21.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL21 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - b.p2.x; let dy = a.p2.y - b.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancell22.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL22 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// -- Line endpoint to Point distance --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p1.x - point.pos.x; let dy = line.p1.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancelp1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP1 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - point.pos.x; let dy = line.p2.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancelp2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP2 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

// -- Arc-Point distance --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = arc.center.x - point.pos.x; let dy = arc.center.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancearccenterp.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let dx = sx - point.pos.x; let dy = sy - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancearcstartp.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let dx = ex - point.pos.x; let dy = ey - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancearcendp.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

// -- Arc-Line distance --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = arc.center.x - line.p1.x; let dy = arc.center.y - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancearccenterl1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = arc.center.x - line.p2.x; let dy = arc.center.y - line.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancearccenterl2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcCenterL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let dx = sx - line.p1.x; let dy = sy - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancearcstartl1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let dx = sx - line.p2.x; let dy = sy - line.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancearcstartl2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let dx = ex - line.p1.x; let dy = ey - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancearcendl1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let dx = ex - line.p2.x; let dy = ey - line.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancearcendl2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// -- Arc-Arc distance --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.center.x - b.center.x; let dy = a.center.y - b.center.y;
    [(sqrt(dx * dx + dy * dy) - distanceaacece.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAACeCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    let dx = a.center.x - bsx; let dy = a.center.y - bsy;
    [(sqrt(dx * dx + dy * dy) - distanceaaces.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAACeS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    let dx = a.center.x - bex; let dy = a.center.y - bey;
    [(sqrt(dx * dx + dy * dy) - distanceaacee.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAACeE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let dx = asx - b.center.x; let dy = asy - b.center.y;
    [(sqrt(dx * dx + dy * dy) - distanceaasce.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAASCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    let dx = asx - bsx; let dy = asy - bsy;
    [(sqrt(dx * dx + dy * dy) - distanceaass.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAASS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    let dx = asx - bex; let dy = asy - bey;
    [(sqrt(dx * dx + dy * dy) - distanceaase.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAASE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let dx = aex - b.center.x; let dy = aey - b.center.y;
    [(sqrt(dx * dx + dy * dy) - distanceaaece.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAAECe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    let dx = aex - bsx; let dy = aey - bsy;
    [(sqrt(dx * dx + dy * dy) - distanceaaes.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAAES {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    let dx = aex - bex; let dy = aey - bey;
    [(sqrt(dx * dx + dy * dy) - distanceaaee.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceAAEE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

// -- Point-Line --

// Point lies on infinite line through p1-p2 (cross product = 0)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// Point at midpoint of line segment
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// Line P1 at midpoint of another line
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Line P2 at midpoint of another line
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Arc start point at midpoint of line
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Line>,
}

// Arc end point at midpoint of line
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Line>,
}

// -- Midpoint on Arc (angular midpoint) --

// Point at angular midpoint of arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mid_angle = (arc.start_angle + arc.end_angle) * 0.5;
    let mx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, mid_angle);
    let my = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, mid_angle);
    [(point.pos.x - mx) * sketch.constraint_isigma,
     (point.pos.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcPoint {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Line P1 at angular midpoint of arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mid_angle = (arc.start_angle + arc.end_angle) * 0.5;
    let mx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, mid_angle);
    let my = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, mid_angle);
    [(line.p1.x - mx) * sketch.constraint_isigma,
     (line.p1.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointLP1Arc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Line P2 at angular midpoint of arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mid_angle = (arc.start_angle + arc.end_angle) * 0.5;
    let mx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, mid_angle);
    let my = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, mid_angle);
    [(line.p2.x - mx) * sketch.constraint_isigma,
     (line.p2.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointLP2Arc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Arc start at angular midpoint of another arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let sy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let mid_angle = (b.start_angle + b.end_angle) * 0.5;
    let mx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, mid_angle);
    let my = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, mid_angle);
    [(sx - mx) * sketch.constraint_isigma,
     (sy - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcStartArc {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Arc end at angular midpoint of another arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let ey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let mid_angle = (b.start_angle + b.end_angle) * 0.5;
    let mx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, mid_angle);
    let my = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, mid_angle);
    [(ex - mx) * sketch.constraint_isigma,
     (ey - my) * sketch.constraint_isigma]
}))]
pub struct MidpointArcEndArc {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Point-Arc --

// Point lies on ellipse/circle defined by arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [ellipse_implicit(point.pos.x, point.pos.y, arc.center.x, arc.center.y,
        arc.radius, arc.radius_b, arc.rotation) * sketch.constraint_isigma]
}))]
pub struct PointOnArc {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc center
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc start endpoint (center + radius * [cos(sa), sin(sa)])
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(point.pos.x - sx) * sketch.constraint_isigma,
     (point.pos.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStart {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc end endpoint (center + radius * [cos(ea), sin(ea)])
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(point.pos.x - ex) * sketch.constraint_isigma,
     (point.pos.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEnd {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// -- Line-Line --

// Parallel: cross product of direction vectors = 0
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    let len1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let len2 = sqrt(dx2 * dx2 + dy2 * dy2);
    let mlen = (len1 + len2) / 2.0;
    [(dx1 * dy2 - dy1 * dx2) / mlen * sketch.constraint_isigma]
}))]
pub struct Parallel {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Perpendicular: dot product of direction vectors = 0, with direction enforcement.
// The Heaviside on the unnormalized cross product prevents direction reversal.
// Its gradient at zero line length is the other line's direction -- always well-defined.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    let len1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let len2 = sqrt(dx2 * dx2 + dy2 * dy2);
    let mlen = (len1 + len2) / 2.0;
    let cross = dx1 * dy2 - dy1 * dx2;
    let d = sketch.min_length - perpendicular.dir_sign * cross;
    [
        (dx1 * dx2 + dy1 * dy2) / mlen * sketch.constraint_isigma,
        heaviside(d) * d * sketch.constraint_isigma
    ]
}))]
pub struct Perpendicular {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(default = "default_dir_sign")]
    pub dir_sign: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Arc-Line parallel: ellipse's major-axis direction parallel to a
// line's direction. Residual is the unnormalised 2D cross product of
// the line direction `(dx, dy)` with the ellipse's major-axis unit
// vector `(cos(rotation), sin(rotation))`, which equals
// `|line| * sin(angle_between)`. Zero iff axes are parallel or
// antiparallel -- natural pi-periodicity means the solver does not
// fight the ellipse's inherent two-fold rotational symmetry.
//
// We deliberately do NOT divide by the line length. The old form
// `... / len * isigma` normalised the residual to `sin(angle)`, but
// that made the Jacobian wrt positions scale as `isigma / len`, so
// on a long line the SVD's singular value from this row shrank like
// `1 / len` -- producing tiny sigmas at large sketch scales and
// breaking the rank algorithm's gap detection. The unnormalised
// form gives Jacobian wrt positions ~= `isigma` (scale-invariant)
// and Jacobian wrt rotation ~= `len * isigma` (scale-linear, same
// family as coincidence-to-arc-endpoint derivatives wrt angles).
// The SV from this row is now in the same order as the other
// angle-mode sigmas, not collapsing to zero at large scales.
//
// Guarded on arc.is_ellipse; circular arcs have rotation fixed at 0
// and the constraint would be meaningless.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = arc.is_ellipse, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    [(cos(arc.rotation) * dy - sin(arc.rotation) * dx) * sketch.constraint_isigma]
}))]
pub struct ArcLineParallel {
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Line>,
}

// Arc-Arc parallel: two ellipses share the same major-axis direction.
// Residual is sin(a.rotation - b.rotation) -- zero when rotations
// match modulo pi, which matches the ellipse's inherent two-fold
// rotational symmetry. Guarded on both is_ellipse; inert for
// circular-arc operands (whose rotations are fixed at 0 anyway).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = a.is_ellipse && b.is_ellipse, {
    [sin(a.rotation - b.rotation) * sketch.constraint_isigma]
}))]
pub struct ArcArcParallel {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Collinear: line2 endpoints both lie on infinite line of line1
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Equal length
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Angle between lines
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line-Point (endpoint coincidence) --

// Line p1 coincides with standalone point
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Point>,
}

// Line p2 coincides with standalone point
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Point>,
}

// -- Line-Line endpoint coincidence (4 variants for endpoint combos) --

// a.p1 == b.p1
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p1 == b.p2
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p2 == b.p1  (most common: end of line a -> start of line b)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p2 == b.p2
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line-Arc --

// Line tangent to arc/ellipse. Uses perpendicular-distance residual (always active)
// plus a gradient-based residual when the tangent point is a shared endpoint.
// For ellipses, the effective radius along the line normal direction is used.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
// Perpendicular distance from center to line = sign * effective_radius (no shared endpoint)
#[arael(constraint(hb, guard = !self.p1_arc_start && !self.p1_arc_end && !self.p2_arc_start && !self.p2_arc_end, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((arc.center.x - line.p1.x) * dy - (arc.center.y - line.p1.y) * dx) / len;
    let nx = 0.0 - dy / len;
    let ny = dx / len;
    let cr = cos(arc.rotation); let sr = sin(arc.rotation);
    let nlx = nx * cr + ny * sr;
    let nly = 0.0 - nx * sr + ny * cr;
    let r_eff = ellipse_effective_radius(nlx, nly, arc.radius, arc.radius_b);
    [(dist - tangentla.sign * r_eff) * sketch.constraint_isigma]
}))]
// Directed tangent at shared endpoint: uses arc parametric endpoint position
// instead of the line endpoint for the direction vector, so the constraint
// stays well-defined even for zero-length lines.  Four variants for which
// line endpoint (p1/p2) meets which arc endpoint (start/end).
#[arael(constraint(hb, guard = self.p1_arc_start, {
    let ax = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let ay = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let tx = arc_tangent_x(arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let ty = arc_tangent_y(arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let tlen = sqrt(tx * tx + ty * ty);
    let dx = line.p2.x - ax;
    let dy = line.p2.y - ay;
    let projx = tangentla.dir_sign * (dx * tx + dy * ty) / tlen;
    let projy = (0.0 - dx * ty + dy * tx) / tlen;
    let d = sketch.min_length - projx;
    [
        heaviside(d) * d * sketch.constraint_isigma,
        projy * sketch.constraint_isigma
    ]
}))]
#[arael(constraint(hb, guard = self.p1_arc_end, {
    let ax = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ay = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let tx = arc_tangent_x(arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ty = arc_tangent_y(arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let tlen = sqrt(tx * tx + ty * ty);
    let dx = line.p2.x - ax;
    let dy = line.p2.y - ay;
    let projx = tangentla.dir_sign * (dx * tx + dy * ty) / tlen;
    let projy = (0.0 - dx * ty + dy * tx) / tlen;
    let d = sketch.min_length - projx;
    [
        heaviside(d) * d * sketch.constraint_isigma,
        projy * sketch.constraint_isigma
    ]
}))]
#[arael(constraint(hb, guard = self.p2_arc_start, {
    let ax = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let ay = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let tx = arc_tangent_x(arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let ty = arc_tangent_y(arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let tlen = sqrt(tx * tx + ty * ty);
    let dx = line.p1.x - ax;
    let dy = line.p1.y - ay;
    let projx = tangentla.dir_sign * (dx * tx + dy * ty) / tlen;
    let projy = (0.0 - dx * ty + dy * tx) / tlen;
    let d = sketch.min_length - projx;
    [
        heaviside(d) * d * sketch.constraint_isigma,
        projy * sketch.constraint_isigma
    ]
}))]
#[arael(constraint(hb, guard = self.p2_arc_end, {
    let ax = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ay = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let tx = arc_tangent_x(arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ty = arc_tangent_y(arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let tlen = sqrt(tx * tx + ty * ty);
    let dx = line.p1.x - ax;
    let dy = line.p1.y - ay;
    let projx = tangentla.dir_sign * (dx * tx + dy * ty) / tlen;
    let projy = (0.0 - dx * ty + dy * tx) / tlen;
    let d = sketch.min_length - projx;
    [
        heaviside(d) * d * sketch.constraint_isigma,
        projy * sketch.constraint_isigma
    ]
}))]
pub struct TangentLA {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default = "default_tangent_sign")]
    pub sign: f64,
    #[serde(skip)]
    pub p1_arc_start: bool,
    #[serde(skip)]
    pub p1_arc_end: bool,
    #[serde(skip)]
    pub p2_arc_start: bool,
    #[serde(skip)]
    pub p2_arc_end: bool,
    #[serde(skip, default = "default_dir_sign")]
    pub dir_sign: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

fn default_dir_sign() -> f64 { f64::NAN }

fn default_tangent_sign() -> f64 { 1.0 }

// -- Arc-Arc --

// Concentric: centers coincide
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Radial distance between two concentric arcs/circles. Self-contained:
// the residual enforces both center-coincidence (`a.center == b.center`)
// and the signed radial gap (`b.radius - a.radius = sign * distance`).
// `sign` is captured at dimension creation time so the gap stays
// sign-stable under big value updates (no mirror flip on which arc is
// outer). Self-containment means the dim survives manual deletion of
// the paired `Concentric` constraint -- the circles stay concentric
// because the dim is enforcing it directly.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, name = "concentric_distance", {
    [(a.center.x - b.center.x) * sketch.constraint_isigma,
     (a.center.y - b.center.y) * sketch.constraint_isigma,
     (b.radius - a.radius - distanceconcentric.distance * distanceconcentric.sign)
     * sketch.constraint_isigma]
}))]
pub struct DistanceConcentric {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default = "default_tangent_sign")]
    pub sign: f64,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Equal radius
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.radius - b.radius) * sketch.constraint_isigma]
}))]
pub struct EqualRadius {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Tangent arc-arc (external tangency).
// Uses effective radii along center-to-center direction.
// Generalizes circles: when rx=ry=r, r_eff = r.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    let r_eff_a = ellipse_effective_radius(nxa, nya, a.radius, a.radius_b);
    let crb = cos(b.rotation);
    let srb = sin(b.rotation);
    let nxb = 0.0 - nx * crb - ny * srb;
    let nyb = nx * srb - ny * crb;
    let r_eff_b = ellipse_effective_radius(nxb, nyb, b.radius, b.radius_b);
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
    #[serde(default)]
    pub shared: SharedEndpoint,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Line endpoint <-> Arc point coincidence --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(line.p1.x - sx) * sketch.constraint_isigma,
     (line.p1.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcStart {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(line.p2.x - sx) * sketch.constraint_isigma,
     (line.p2.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcStart {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(line.p1.x - ex) * sketch.constraint_isigma,
     (line.p1.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcEnd {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(line.p2.x - ex) * sketch.constraint_isigma,
     (line.p2.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcEnd {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Arc-Arc endpoint coincidence --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(a.center.x - bsx) * sketch.constraint_isigma,
     (a.center.y - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenterStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(a.center.x - bex) * sketch.constraint_isigma,
     (a.center.y - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenterEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    [(asx - b.center.x) * sketch.constraint_isigma,
     (asy - b.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartCenter {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    [(aex - b.center.x) * sketch.constraint_isigma,
     (aey - b.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndCenter {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(asx - bsx) * sketch.constraint_isigma,
     (asy - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(asx - bex) * sketch.constraint_isigma,
     (asy - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(aex - bsx) * sketch.constraint_isigma,
     (aey - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(aex - bex) * sketch.constraint_isigma,
     (aey - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Line endpoint on line --

// Line a's p1 lies on infinite line through b's p1-p2
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Line a's p2 lies on infinite line through b's p1-p2
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line endpoint on arc --

// Line p1 lies on ellipse/circle defined by arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [ellipse_implicit(line.p1.x, line.p1.y, arc.center.x, arc.center.y,
        arc.radius, arc.radius_b, arc.rotation) * sketch.constraint_isigma]
}))]
pub struct LineP1OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Line p2 lies on ellipse/circle defined by arc
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [ellipse_implicit(line.p2.x, line.p2.y, arc.center.x, arc.center.y,
        arc.radius, arc.radius_b, arc.rotation) * sketch.constraint_isigma]
}))]
pub struct LineP2OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Symmetry (3-entity) --

// Symmetry of 3 lines: from each B endpoint, cast a ray and intersect
// with A and C. Signed ray parameters must sum to zero.
// `use_normal_ray` selects ray direction: true = B's normal (default),
// false = B's direction (for when A/C are nearly perpendicular to B).
// Set at constraint creation based on initial geometry.
// Dense 3-entity coupling: one named CrossBlock<Line, Line> per unordered
// ref pair. Packed NA*NB storage is faster than TripletBlock's COO push.
// `cross = (refA, refB)` is mandatory here since all three CrossBlocks
// share the same type signature.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], {
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[arael(cross = (a, b))]
    #[serde(skip)]
    pub hb_ab: CrossBlock<Line, Line>,
    #[arael(cross = (a, c))]
    #[serde(skip)]
    pub hb_ac: CrossBlock<Line, Line>,
    #[arael(cross = (b, c))]
    #[serde(skip)]
    pub hb_bc: CrossBlock<Line, Line>,
}

// -- Distance Point-Line --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// Line endpoint p1 to line (signed perpendicular distance)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// Line endpoint p2 to line (signed perpendicular distance)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// Arc center to line (signed perpendicular distance)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// Arc start to line (signed perpendicular distance)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    let dist = ((sx - line.p1.x) * dy - (sy - line.p1.y) * dx) / len;
    [(dist - distancearcstartl.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcStartL {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// Arc end to line (signed perpendicular distance)
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    let dist = ((ex - line.p1.x) * dy - (ey - line.p1.y) * dx) / len;
    [(dist - distancearcendl.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceArcEndL {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// ---------------------------------------------------------------------------
// Symmetry: two points about a mirror line
// ---------------------------------------------------------------------------

/// Two points forced symmetric about a mirror line.
/// Residual: reflect a across line, compare to c.
/// Dense 3-entity coupling: packed CrossBlocks for each pair beat
/// TripletBlock's COO push. hb_ac is unambiguous by type (only
/// Point-Point pair); the two Point-Line blocks need explicit
/// `cross = (..)` to pick their ref.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint([hb_ac, hb_al, hb_cl], {
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb_ac: CrossBlock<Point, Point>,
    #[arael(cross = (a, line))]
    #[serde(skip)]
    pub hb_al: CrossBlock<Point, Line>,
    #[arael(cross = (c, line))]
    #[serde(skip)]
    pub hb_cl: CrossBlock<Point, Line>,
}

// Symmetry of two arcs/ellipses about a mirror line.
// Guarded: circle path uses 3 residuals (center + radius), ellipse path
// adds radius_b equality + rotation reflection (5 residuals).
// Cannot unify because radius_b is a real parameter for circles (equality
// constraint), so the extra residuals affect DOF counting.
// Dense 3-entity coupling; decomposed into packed CrossBlocks per pair.
// Both guarded constraint bodies share the same three CrossBlock fields.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint([hb_ac, hb_al, hb_cl], guard = !a.is_ellipse && !c.is_ellipse, {
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
#[arael(constraint([hb_ac, hb_al, hb_cl], guard = a.is_ellipse || c.is_ellipse, {
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)]
    pub hb_ac: CrossBlock<Arc, Arc>,
    #[arael(cross = (a, line))]
    #[serde(skip)]
    pub hb_al: CrossBlock<Arc, Line>,
    #[arael(cross = (c, line))]
    #[serde(skip)]
    pub hb_cl: CrossBlock<Arc, Line>,
}

// ---------------------------------------------------------------------------
// Axis distance (horizontal/vertical unified via guard flag)
// ---------------------------------------------------------------------------

// -- Line-Line --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// -- Line-Point --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

// -- Arc-Point --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(sx - point.pos.x - axisdistancearcstartp.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(sy - point.pos.y - axisdistancearcstartp.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcStartP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(ex - point.pos.x - axisdistancearcendp.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(ey - point.pos.y - axisdistancearcendp.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcEndP {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Point>,
}

// -- Arc-Line --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(sx - line.p1.x - axisdistancearcstartl1.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(sy - line.p1.y - axisdistancearcstartl1.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcStartL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let sx = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(sx - line.p2.x - axisdistancearcstartl2.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let sy = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.start_angle);
    [(sy - line.p2.y - axisdistancearcstartl2.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcStartL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(ex - line.p1.x - axisdistancearcendl1.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(ey - line.p1.y - axisdistancearcendl1.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcEndL1 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let ex = arc_point_x(arc.center.x, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(ex - line.p2.x - axisdistancearcendl2.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let ey = arc_point_y(arc.center.y, arc.radius, arc.radius_b, arc.rotation, arc.end_angle);
    [(ey - line.p2.y - axisdistancearcendl2.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceArcEndL2 {
    #[arael(ref = root.arcs)] pub arc: Ref<Arc>,
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Line>,
}

// -- Arc-Arc --

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(a.center.x - bsx - axisdistanceaaces.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(a.center.y - bsy - axisdistanceaaces.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAACeS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(a.center.x - bex - axisdistanceaacee.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(a.center.y - bey - axisdistanceaacee.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAACeE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    [(asx - b.center.x - axisdistanceaasce.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    [(asy - b.center.y - axisdistanceaasce.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAASCe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(asx - bsx - axisdistanceaass.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(asy - bsy - axisdistanceaass.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAASS {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let asx = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(asx - bex - axisdistanceaase.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let asy = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.start_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(asy - bey - axisdistanceaase.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAASE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    [(aex - b.center.x - axisdistanceaaece.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    [(aey - b.center.y - axisdistanceaaece.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAAECe {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bsx = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(aex - bsx - axisdistanceaaes.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bsy = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.start_angle);
    [(aey - bsy - axisdistanceaaes.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAAES {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, guard = self.horizontal, {
    let aex = arc_point_x(a.center.x, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bex = arc_point_x(b.center.x, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(aex - bex - axisdistanceaaee.distance) * sketch.constraint_isigma]
}))]
#[arael(constraint(hb, guard = !self.horizontal, {
    let aey = arc_point_y(a.center.y, a.radius, a.radius_b, a.rotation, a.end_angle);
    let bey = arc_point_y(b.center.y, b.radius, b.radius_b, b.rotation, b.end_angle);
    [(aey - bey - axisdistanceaaee.distance) * sketch.constraint_isigma]
}))]
pub struct AxisDistanceAAEE {
    #[arael(ref = root.arcs)] pub a: Ref<Arc>,
    #[arael(ref = root.arcs)] pub b: Ref<Arc>,
    pub distance: f64,
    pub horizontal: bool,
    #[serde(default)]
    pub nid: u32,
    #[arael(constraint_index)]
    #[serde(skip)]
    pub cid: u32,
    #[serde(skip)] pub hb: CrossBlock<Arc, Arc>,
}

