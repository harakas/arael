//! Two-entity cross constraints held two hops down under different
//! parents, in collections of the same field name: `a_images[].obs` and
//! `b_images[].obs`. Each observation couples its point and its camera
//! and reads the image's plain data through `parent.`. The two sweeps
//! must stay apart: the cost is the sum over both collections, and the
//! model validates.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmProblem, RootProblem};
use arael::vect::vect2d;

#[arael::model]
struct Point {
    p: Param<vect2d>,
    hb: SelfBlock<Point>,
}

#[arael::model]
struct CamA {
    f: Param<f64>,
    hb: SelfBlock<CamA>,
}

#[arael::model]
struct CamB {
    g: Param<f64>,
    hb: SelfBlock<CamB>,
}

#[arael::model]
struct AImage {
    t: f64,
    obs: std::vec::Vec<AObs>,
}

#[arael::model]
#[arael(constraint([hb], {
    [cam.f * (point.p.x + parent.t) - aobs.m]
}))]
struct AObs {
    #[arael(ref = root.points)] point: Ref<Point>,
    #[arael(ref = root.cams_a)] cam: Ref<CamA>,
    m: f64,
    hb: CrossBlock<Point, CamA>,
}

#[arael::model]
struct BImage {
    t: f64,
    obs: std::vec::Vec<BObs>,
}

#[arael::model]
#[arael(constraint([hb], {
    [cam.g * (point.p.y - parent.t) - bobs.m]
}))]
struct BObs {
    #[arael(ref = root.points)] point: Ref<Point>,
    #[arael(ref = root.cams_b)] cam: Ref<CamB>,
    m: f64,
    hb: CrossBlock<Point, CamB>,
}

#[arael::model]
#[arael(root)]
struct World {
    points: refs::Vec<Point>,
    cams_a: refs::Vec<CamA>,
    cams_b: refs::Vec<CamB>,
    a_images: refs::Vec<AImage>,
    b_images: refs::Vec<BImage>,
}

const PX: [f64; 2] = [1.0, -0.5];
const PY: [f64; 2] = [0.25, 2.0];
const F: f64 = 1.5;
const G: f64 = 0.8;
const TA: f64 = 0.3;
const TB: f64 = -0.2;
const MA: [f64; 2] = [1.7, -0.4];
const MB: [f64; 2] = [0.5, 1.9];

fn build() -> World {
    let mut points = refs::Vec::new();
    for k in 0..2 {
        points.push(Point { p: Param::new(vect2d::new(PX[k], PY[k])), hb: SelfBlock::new() });
    }
    let mut cams_a = refs::Vec::new();
    cams_a.push(CamA { f: Param::new(F), hb: SelfBlock::new() });
    let mut cams_b = refs::Vec::new();
    cams_b.push(CamB { g: Param::new(G), hb: SelfBlock::new() });
    let krefs: Vec<Ref<Point>> = points.iter_refs().map(|(r, _)| r).collect();
    let ca = cams_a.iter_refs().next().unwrap().0;
    let cb = cams_b.iter_refs().next().unwrap().0;

    let mut a_images = refs::Vec::new();
    a_images.push(AImage {
        t: TA,
        obs: (0..2).map(|k| AObs { point: krefs[k], cam: ca, m: MA[k], hb: CrossBlock::new() }).collect(),
    });
    let mut b_images = refs::Vec::new();
    b_images.push(BImage {
        t: TB,
        obs: (0..2).map(|k| BObs { point: krefs[k], cam: cb, m: MB[k], hb: CrossBlock::new() }).collect(),
    });
    World { points, cams_a, cams_b, a_images, b_images }
}

fn manual_cost() -> f64 {
    let mut c = 0.0;
    for k in 0..2 {
        let ra = F * (PX[k] + TA) - MA[k];
        let rb = G * (PY[k] - TB) - MB[k];
        c += ra * ra + rb * rb;
    }
    c
}

#[test]
fn same_named_collections_under_different_parents_stay_apart() {
    let mut m = build();
    let mut x = Vec::new();
    RootProblem::serialize(&mut m, &mut x);
    let cost = m.calc_cost(&x);
    let manual = manual_cost();
    assert!((cost - manual).abs() <= 1e-12 * (1.0 + manual),
        "calc_cost {cost} != manual {manual}");
    let d = m.validate();
    assert!(d.is_clean(), "validate:\n{}", d);
}
