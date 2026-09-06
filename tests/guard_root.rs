// A constraint guard may read a root field: `guard = root.<field>`
// switches the constraint off for the whole model, on every path.
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::LmConfig;
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;

#[arael::model]
#[arael(constraint(hb, guard = root.anchors_on, {
    [point.pos.x * chain.anchor, point.pos.y * chain.anchor]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * chain.drift, d.y * chain.drift]
}))]
struct Point {
    pos: Param<vect2d>,
    hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, guard = root.links_on && self.rest > 0.0, {
    let d = b.pos - a.pos;
    [(d.norm() - link.rest) * chain.spring]
}))]
struct Link {
    #[arael(ref = root.points)]
    a: Ref<Point>,
    #[arael(ref = root.points)]
    b: Ref<Point>,
    rest: f64,
    hb: CrossBlock<Point, Point>,
}

#[arael::model]
#[arael(root)]
struct Chain {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    anchors_on: bool,
    links_on: bool,
    anchor: f64,
    drift: f64,
    spring: f64,
}

fn build(anchors_on: bool, links_on: bool) -> Chain {
    let mut c = Chain {
        points: refs::Vec::new(),
        links: std::vec::Vec::new(),
        anchors_on,
        links_on,
        anchor: 1.0,
        drift: 1.0,
        spring: 1.0,
    };
    let a = c.points.push(Point { pos: Param::new(vect2d::new(1.0, 1.0)), hb: SelfBlock::new() });
    let b = c.points.push(Point { pos: Param::new(vect2d::new(3.0, 1.0)), hb: SelfBlock::new() });
    c.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    c
}

fn cfg() -> LmConfig<f64> {
    LmConfig { abs_precision: 1e-14, rel_precision: 1e-12, max_iters: 200, ..Default::default() }
}

#[test]
fn root_guard_switches_a_self_block_constraint() {
    // Anchors off: only the drift term, so the points stay put.
    let x = build(false, false).solve_sparse(&cfg()).unwrap().x;
    assert!((x[0] - 1.0).abs() < 1e-6, "x = {:?}", x);
    assert!((x[2] - 3.0).abs() < 1e-6, "x = {:?}", x);
    // Anchors on: the origin pull and the drift balance halfway.
    let x = build(true, false).solve_sparse(&cfg()).unwrap().x;
    assert!((x[0] - 0.5).abs() < 1e-6, "x = {:?}", x);
    assert!((x[2] - 1.5).abs() < 1e-6, "x = {:?}", x);
}

#[test]
fn root_guard_switches_a_cross_block_constraint() {
    // Links off: the drift alone, distance stays 2.
    let x = build(false, false).solve_sparse(&cfg()).unwrap().x;
    assert!((x[2] - x[0] - 2.0).abs() < 1e-6, "x = {:?}", x);
    // Links on: the spring (rest 1) and the drifts pull the distance below 2.
    let x = build(false, true).solve_sparse(&cfg()).unwrap().x;
    assert!(x[2] - x[0] < 1.9, "x = {:?}", x);
}
