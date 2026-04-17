// Multi-CrossBlock constraint tests.
//
// Verifies that a 3-entity constraint declared with three named CrossBlocks
// (one per ref pair) produces the same gradient and Hessian as the
// equivalent constraint declared with a single TripletBlock. This is the
// "never drop cross-Hessian" invariant under the multi-cross refactor.

#[allow(unused_imports)]
use arael::model::{Param, SelfBlock, CrossBlock, TripletBlock, Model};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;

// Minimal 2D point entity with its own SelfBlock + drift constraint.
#[arael::model]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * testmodel.drift_isigma, d.y * testmodel.drift_isigma]
}))]
struct Point {
    pos: Param<vect2d>,
    pos_value: vect2d,
    hb: SelfBlock<Point>,
}

// 3-entity constraint declared with a single TripletBlock. Residual
// couples all three points (a simple dense coupling).
#[arael::model]
#[arael(constraint(hb, {
    let dx_ab = a.pos.x - b.pos.x;
    let dy_ab = a.pos.y - b.pos.y;
    let dx_bc = b.pos.x - c.pos.x;
    let dy_bc = b.pos.y - c.pos.y;
    let dx_ac = a.pos.x - c.pos.x;
    // Couples all three: contains derivatives with respect to
    // a.pos, b.pos, c.pos.
    [(dx_ab + dx_bc - dx_ac) * testmodel.constraint_isigma,
     (dy_ab + dy_bc) * testmodel.constraint_isigma,
     (dx_ab * dy_bc) * testmodel.constraint_isigma]
}))]
struct TripletCoupling {
    #[arael(ref = root.points)]
    a: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    b: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    c: arael::refs::Ref<Point>,
    hb: TripletBlock<f64>,
}

// Same residual, but expressed via three CrossBlocks (one per unordered
// ref pair) instead of a TripletBlock. Each CrossBlock carries the
// Hessian contributions for its pair; per-entity SelfBlocks hold
// gradient + within-entity diagonal (as usual).
#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], {
    let dx_ab = a.pos.x - b.pos.x;
    let dy_ab = a.pos.y - b.pos.y;
    let dx_bc = b.pos.x - c.pos.x;
    let dy_bc = b.pos.y - c.pos.y;
    let dx_ac = a.pos.x - c.pos.x;
    [(dx_ab + dx_bc - dx_ac) * testmodel.constraint_isigma,
     (dy_ab + dy_bc) * testmodel.constraint_isigma,
     (dx_ab * dy_bc) * testmodel.constraint_isigma]
}))]
struct MultiCrossCoupling {
    #[arael(ref = root.points)]
    a: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    b: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    c: arael::refs::Ref<Point>,
    #[arael(cross = (a, b))] hb_ab: CrossBlock<Point, Point>,
    #[arael(cross = (a, c))] hb_ac: CrossBlock<Point, Point>,
    #[arael(cross = (b, c))] hb_bc: CrossBlock<Point, Point>,
}

#[arael::model]
#[arael(root)]
struct TestModel {
    points: arael::refs::Vec<Point>,
    triplet_coupling: arael::refs::Vec<TripletCoupling>,
    multi_cross_coupling: arael::refs::Vec<MultiCrossCoupling>,
    drift_isigma: f64,
    constraint_isigma: f64,
}

fn make_points() -> arael::refs::Vec<Point> {
    let mut pts = arael::refs::Vec::new();
    pts.push(Point {
        pos: Param::new(vect2d::new(1.3, -0.7)),
        pos_value: vect2d::new(1.0, 0.0),
        hb: SelfBlock::new(),
    });
    pts.push(Point {
        pos: Param::new(vect2d::new(3.1, 0.4)),
        pos_value: vect2d::new(3.0, 0.0),
        hb: SelfBlock::new(),
    });
    pts.push(Point {
        pos: Param::new(vect2d::new(5.2, -0.3)),
        pos_value: vect2d::new(5.0, 0.0),
        hb: SelfBlock::new(),
    });
    pts
}

fn make_triplet_model() -> (TestModel, Vec<f64>) {
    let mut model = TestModel {
        points: make_points(),
        triplet_coupling: arael::refs::Vec::new(),
        multi_cross_coupling: arael::refs::Vec::new(),
        drift_isigma: 1.0,
        constraint_isigma: 10.0,
    };
    model.triplet_coupling.push(TripletCoupling {
        a: arael::refs::Ref::new(0),
        b: arael::refs::Ref::new(1),
        c: arael::refs::Ref::new(2),
        hb: TripletBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize64(&mut params);
    (model, params)
}

fn make_multi_cross_model() -> (TestModel, Vec<f64>) {
    let mut model = TestModel {
        points: make_points(),
        triplet_coupling: arael::refs::Vec::new(),
        multi_cross_coupling: arael::refs::Vec::new(),
        drift_isigma: 1.0,
        constraint_isigma: 10.0,
    };
    model.multi_cross_coupling.push(MultiCrossCoupling {
        a: arael::refs::Ref::new(0),
        b: arael::refs::Ref::new(1),
        c: arael::refs::Ref::new(2),
        hb_ab: CrossBlock::new(),
        hb_ac: CrossBlock::new(),
        hb_bc: CrossBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize64(&mut params);
    (model, params)
}

#[test]
fn test_multi_cross_cost_matches_triplet() {
    let (mut tm, tp) = make_triplet_model();
    let (mut mm, mp) = make_multi_cross_model();
    let tc = tm.calc_cost(&tp);
    let mc = mm.calc_cost(&mp);
    assert!((tc - mc).abs() < 1e-10,
        "cost mismatch: triplet={}, multi-cross={}", tc, mc);
}

#[test]
fn test_multi_cross_grad_hessian_matches_triplet() {
    let (mut tm, tp) = make_triplet_model();
    let (mut mm, mp) = make_multi_cross_model();
    let n = tp.len();
    assert_eq!(mp.len(), n);

    let mut tg = vec![0.0; n];
    let mut th = vec![0.0; n * n];
    tm.calc_grad_hessian_dense(&tp, &mut tg, &mut th);

    let mut mg = vec![0.0; n];
    let mut mh = vec![0.0; n * n];
    mm.calc_grad_hessian_dense(&mp, &mut mg, &mut mh);

    for i in 0..n {
        assert!((tg[i] - mg[i]).abs() < 1e-8,
            "grad[{}] differs: triplet={}, multi-cross={}", i, tg[i], mg[i]);
    }
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            assert!((th[idx] - mh[idx]).abs() < 1e-6,
                "hess[{},{}] differs: triplet={}, multi-cross={}", i, j, th[idx], mh[idx]);
        }
    }
}
