// Integration tests: geometry ops in constraint bodies, end to end through
// the macro. Residual values are checked against the runtime vector math,
// and analytic gradients against central finite differences of calc_cost.

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;
use arael::vect::{vect2d, vect3d};
use arael::refs::{self, Ref};

#[arael::model]
struct P3 {
    pos: Param<vect3d>,
    hb: SelfBlock<P3>,
}

// Vec3 ops: cross via `%` and via `.cross()`, norm, square, unit,
// division by scalar.
#[arael::model]
#[arael(constraint(hb, {
    let c = a.pos % b.pos;
    let u = a.pos.unit();
    let d = (a.pos - b.pos) / 2.0;
    [c.x * space.w, c.y * space.w, c.z * space.w,
     b.pos.cross(a.pos).y * space.w,
     (a.pos - b.pos).norm() * space.w,
     b.pos.square() * space.w,
     u.x * space.w, u.z * space.w,
     d.x * space.w, d.y * space.w, d.z * space.w]
}))]
struct V3Link {
    #[arael(ref = root.points)]
    a: Ref<P3>,
    #[arael(ref = root.points)]
    b: Ref<P3>,
    hb: CrossBlock<P3, P3>,
}

// Vec2 unary negation.
#[arael::model]
#[arael(constraint(hb, {
    let n = -planar.pos;
    [n.x * space.w, n.y * space.w]
}))]
struct Planar {
    pos: Param<vect2d>,
    hb: SelfBlock<Planar>,
}

#[arael::model]
#[arael(root)]
struct Space {
    points: refs::Vec<P3>,
    links: std::vec::Vec<V3Link>,
    planars: refs::Vec<Planar>,
    w: f64,
}

const A: (f64, f64, f64) = (0.3, -1.2, 2.0);
const B: (f64, f64, f64) = (1.7, 0.4, -0.6);
const P: (f64, f64) = (0.7, -0.4);
const W: f64 = 1.3;

fn build() -> (Space, Vec<f64>) {
    let mut space = Space {
        points: refs::Vec::new(),
        links: std::vec::Vec::new(),
        planars: refs::Vec::new(),
        w: W,
    };
    space.points.push(P3 {
        pos: Param::new(vect3d::new(A.0, A.1, A.2)),
        hb: SelfBlock::new(),
    });
    space.points.push(P3 {
        pos: Param::new(vect3d::new(B.0, B.1, B.2)),
        hb: SelfBlock::new(),
    });
    space.links.push(V3Link {
        a: Ref::new(0),
        b: Ref::new(1),
        hb: CrossBlock::new(),
    });
    space.planars.push(Planar {
        pos: Param::new(vect2d::new(P.0, P.1)),
        hb: SelfBlock::new(),
    });
    let mut params = Vec::new();
    space.serialize64(&mut params);
    (space, params)
}

#[test]
fn cost_matches_runtime_vector_math() {
    let (mut space, params) = build();

    let a = vect3d::new(A.0, A.1, A.2);
    let b = vect3d::new(B.0, B.1, B.2);
    let c = a % b;
    let u = a.unit();
    let d = (a - b) * 0.5;
    let residuals = [
        c.x * W, c.y * W, c.z * W,
        (b % a).y * W,
        (a - b).norm() * W,
        b.square() * W,
        u.x * W, u.z * W,
        d.x * W, d.y * W, d.z * W,
        -P.0 * W, -P.1 * W,
    ];
    let expected: f64 = residuals.iter().map(|r| r * r).sum();

    let cost = space.calc_cost(&params);
    assert!((cost - expected).abs() < 1e-10 * expected,
        "cost={} expected={}", cost, expected);
}

#[test]
fn gradient_matches_finite_differences() {
    let (mut space, params) = build();
    let n = params.len();
    assert_eq!(n, 8, "2 x vect3 + 1 x vect2 params");

    let mut grad = vec![0.0_f64; n];
    let mut hessian = vec![0.0_f64; n * n];
    space.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    let eps = 1e-6;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        let mut p = params.clone();
        p[i] += eps;
        let cp = space.calc_cost(&p);
        p[i] -= 2.0 * eps;
        let cm = space.calc_cost(&p);
        let fd = (cp - cm) / (2.0 * eps);
        assert!((fd - grad[i]).abs() < 1e-4 * (1.0 + fd.abs()),
            "grad[{}]: analytic={} fd={}", i, grad[i], fd);
        max_abs = max_abs.max(grad[i].abs());
    }
    // Guard against a silently-zero gradient passing the comparison.
    assert!(max_abs > 1.0, "gradient suspiciously small: {}", max_abs);
}
