// Integration tests: geometry ops in constraint bodies, end to end through
// the macro. Residual values are checked against the runtime vector math,
// and analytic gradients against central finite differences of calc_cost.

use arael::model::{Model, Param, SelfBlock, CrossBlock, SimpleEulerAngleParam};
use arael::simple_lm::LmProblem;
use arael::vect::{vect2d, vect3d};
use arael::matrix::{matrix2d, matrix3d};
use arael::quatern::quaternd;
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

// Vec2 unary negation; plus binding-shadows-constant semantics: `let e`
// must refer to the binding, not to Euler's number, while the unbound
// `pi` still resolves to the named constant.
#[arael::model]
#[arael(constraint(hb, {
    let n = -planar.pos;
    let e = n.x + n.y;
    [n.x * space.w, n.y * space.w, e * space.w, (pi - 3.0) * space.w]
}))]
struct Planar {
    pos: Param<vect2d>,
    hb: SelfBlock<Planar>,
}

// 2x2 matrix surface: matrix2d data field, constructors, det,
// get_rotation_angle, add/sub/neg/scalar mul, v * M, m[i][j].
#[arael::model]
#[arael(constraint(hb, {
    let r = matrix2sym::rotation(n2.ang);
    let s = r * n2.m - n2.m.transpose();
    let t = -s + s * 2.0;
    let vm = n2.pos * n2.m;
    let f = matrix2sym::from_elements(n2.ang, 1.0, 0.0, n2.ang);
    let g = matrix2sym::from_rows(n2.pos, n2.pos.across());
    let h = matrix2sym::from_cols(n2.pos, n2.pos.across());
    let rs = matrix2sym::rotation_from_sincos(sin(n2.ang), cos(n2.ang));
    [t[0][0] * space.w, t[1][0] * space.w,
     vm.x * space.w, vm.y * space.w,
     (r * n2.m).det() * space.w,
     rs.get_rotation_angle() * space.w,
     (f * n2.pos).x * space.w,
     (g * n2.pos).y * space.w,
     (h * n2.pos).x * space.w,
     (matrix2sym::identity() * n2.pos).y * space.w]
}))]
struct N2 {
    pos: Param<vect2d>,
    ang: Param<f64>,
    m: matrix2d,
    hb: SelfBlock<N2>,
}

// 3x3 matrix surface: constructors (identity, from_rows/cols/elements,
// rotation_from_euler_angles, rotation_from_axis_angle), det,
// add/sub/neg/scalar mul, v * M, m[i][j].
#[arael::model]
#[arael(constraint(hb, {
    let r = matrix3sym::rotation_from_axis_angle(n3.axis, n3.ang);
    let re = matrix3sym::rotation_from_euler_angles(n3.p);
    let q = (r * 2.0 - matrix3sym::identity()) * n3.p;
    let vm = n3.p * r;
    let fr = matrix3sym::from_rows(n3.p, n3.axis, n3.p);
    let fc = matrix3sym::from_cols(n3.p, n3.axis, n3.p);
    let fe = matrix3sym::from_elements(n3.ang, 0.0, 0.0, 0.0, n3.ang, 0.0, 0.0, 0.0, 1.0);
    let s = -fe + fe * 2.0;
    [q.x * space.w, q.y * space.w, q.z * space.w,
     vm.x * space.w, vm.z * space.w,
     r.det() * space.w,
     re[2][0] * space.w,
     fr[1][2] * space.w,
     fc[1][2] * space.w,
     (s * n3.p).x * space.w]
}))]
struct N3 {
    p: Param<vect3d>,
    ang: Param<f64>,
    axis: vect3d,
    hb: SelfBlock<N3>,
}

// Euler angle param coercion: the composed angle vector must work in
// Mul (scaling and dot), Div, and unary Neg exactly as in Add/Sub.
#[arael::model]
#[arael(constraint(hb, {
    let s = ean.ea * 2.0;
    let t = 0.5 * ean.ea;
    let n = -ean.ea;
    let d = ean.ea / 4.0;
    [s.x * space.w, t.y * space.w, n.z * space.w, d.x * space.w,
     (ean.ea * ean.v) * space.w,
     (ean.ea - ean.v).x * space.w]
}))]
struct Ean {
    ea: SimpleEulerAngleParam<f64>,
    v: vect3d,
    hb: SelfBlock<Ean>,
}

// Quaternion surface: quaternd data field, Hamilton algebra, rotate,
// conj/dot/norm/unit, rotation_matrix, get_euler_angles, constructors.
#[arael::model]
#[arael(constraint(hb, {
    let r = quaternsym::from_axis_angle(qn.axis, qn.ang);
    let fe = quaternsym::from_euler_angles(qn.p);
    let h = (qn.q * r).conj();
    let rv = qn.q.rotate(qn.p);
    let u = (qn.q * 2.0 - -qn.q + quaternsym::identity()).unit();
    [rv.x * space.w, rv.y * space.w, rv.z * space.w,
     h.t * space.w, h.v.x * space.w,
     qn.q.dot(r) * space.w,
     fe.v.y * space.w, fe.t * space.w,
     u.t * space.w, u.v.z * space.w,
     qn.q.rotation_matrix()[0][1] * space.w,
     r.get_euler_angles().z * space.w,
     (0.5 * qn.q).norm() * space.w]
}))]
struct Qn {
    p: Param<vect3d>,
    ang: Param<f64>,
    q: quaternd,
    axis: vect3d,
    hb: SelfBlock<Qn>,
}

// A2/A1 semantics: a `let` named like the entity variable shadows the
// pre-registered entity paths (shnode.x below is the let's Vec2
// component, NOT the entity's decoy `x` param field, which the old
// dotted-first lookup would have picked), and #[arael(skip)] fields
// pass through to generated code as verbatim field access.
#[arael::model]
#[arael(constraint(hb, {
    let b = shnode.bias;
    let shnode = shnode.pos * 2.0;
    [(shnode.x - 1.0) * space.w,
     (shnode.y + b) * space.w]
}))]
struct ShNode {
    pos: Param<vect2d>,
    x: Param<f64>,
    #[arael(skip)]
    bias: f64,
    hb: SelfBlock<ShNode>,
}

#[arael::model]
#[arael(root)]
struct Space {
    points: refs::Vec<P3>,
    links: std::vec::Vec<V3Link>,
    planars: refs::Vec<Planar>,
    nodes2: refs::Vec<N2>,
    nodes3: refs::Vec<N3>,
    eans: refs::Vec<Ean>,
    quats: refs::Vec<Qn>,
    shs: refs::Vec<ShNode>,
    w: f64,
}

const A: (f64, f64, f64) = (0.3, -1.2, 2.0);
const B: (f64, f64, f64) = (1.7, 0.4, -0.6);
const P: (f64, f64) = (0.7, -0.4);
const W: f64 = 1.3;
const ANG2: f64 = 0.9;
const ANG3: f64 = 0.6;
const EA: (f64, f64, f64) = (0.2, -0.3, 0.4);
const EV: (f64, f64, f64) = (1.1, 0.5, -0.7);
const QANG: f64 = 0.7;

fn qdata() -> quaternd {
    quaternd::from_axis_angle(vect3d::new(2.0, -1.0, 0.5).unit(), 1.1)
}

fn m2() -> matrix2d {
    matrix2d::from_elements(0.8, -0.3, 0.5, 1.1)
}
fn axis3() -> vect3d {
    vect3d::new(1.0, 2.0, 2.0).unit()
}

fn build() -> (Space, Vec<f64>) {
    let mut space = Space {
        points: refs::Vec::new(),
        links: std::vec::Vec::new(),
        planars: refs::Vec::new(),
        nodes2: refs::Vec::new(),
        nodes3: refs::Vec::new(),
        eans: refs::Vec::new(),
        quats: refs::Vec::new(),
        shs: refs::Vec::new(),
        w: W,
    };
    space.shs.push(ShNode {
        pos: Param::new(vect2d::new(0.6, -0.8)),
        x: Param::new(9.0), // decoy: must NOT appear in any residual
        bias: 0.35,
        hb: SelfBlock::new(),
    });
    space.quats.push(Qn {
        p: Param::new(vect3d::new(A.0, A.1, A.2)),
        ang: Param::new(QANG),
        q: qdata(),
        axis: axis3(),
        hb: SelfBlock::new(),
    });
    space.eans.push(Ean {
        ea: SimpleEulerAngleParam::new(vect3d::new(EA.0, EA.1, EA.2)),
        v: vect3d::new(EV.0, EV.1, EV.2),
        hb: SelfBlock::new(),
    });
    space.nodes2.push(N2 {
        pos: Param::new(vect2d::new(P.0, P.1)),
        ang: Param::new(ANG2),
        m: m2(),
        hb: SelfBlock::new(),
    });
    space.nodes3.push(N3 {
        p: Param::new(vect3d::new(A.0, A.1, A.2)),
        ang: Param::new(ANG3),
        axis: axis3(),
        hb: SelfBlock::new(),
    });
    space.points.push(P3 {
        pos: Param::new(vect3d::new(A.0, A.1, A.2)),
        hb: SelfBlock::new(),
    });
    space.points.push(P3 {
        pos: Param::new(vect3d::new(B.0, B.1, B.2)),
        hb: SelfBlock::new(),
    });
    space.links.push(V3Link {
        a: space.points.ref_at(0),
        b: space.points.ref_at(1),
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

// The constraint bodies above, replayed with the runtime types.
fn expected_residuals() -> Vec<f64> {
    let mut rs = Vec::new();

    let a = vect3d::new(A.0, A.1, A.2);
    let b = vect3d::new(B.0, B.1, B.2);
    let c = a % b;
    let u = a.unit();
    let d = (a - b) * 0.5;
    rs.extend([
        c.x * W, c.y * W, c.z * W,
        (b % a).y * W,
        (a - b).norm() * W,
        b.square() * W,
        u.x * W, u.z * W,
        d.x * W, d.y * W, d.z * W,
    ]);

    rs.extend([
        -P.0 * W, -P.1 * W,
        (-P.0 + -P.1) * W,
        (std::f64::consts::PI - 3.0) * W,
    ]);

    let pos = vect2d::new(P.0, P.1);
    let r = matrix2d::rotation(ANG2);
    let s = r * m2() - m2().transpose();
    let t = -s + s * 2.0;
    let vm = pos * m2();
    let f = matrix2d::from_elements(ANG2, 1.0, 0.0, ANG2);
    let g = matrix2d::from_rows(pos, pos.across());
    let h = matrix2d::from_cols(pos, pos.across());
    let rs2 = matrix2d::rotation_from_sincos(ANG2.sin(), ANG2.cos());
    rs.extend([
        t[0][0] * W, t[1][0] * W,
        vm.x * W, vm.y * W,
        (r * m2()).det() * W,
        rs2.get_rotation_angle() * W,
        (f * pos).x * W,
        (g * pos).y * W,
        (h * pos).x * W,
        (matrix2d::identity() * pos).y * W,
    ]);

    let p = vect3d::new(A.0, A.1, A.2);
    let r3 = matrix3d::rotation_from_axis_angle(axis3(), ANG3);
    let e = matrix3d::rotation_from_euler_angles(p);
    let q = (2.0 * r3 - matrix3d::identity()) * p;
    let vm3 = p * r3;
    let fr = matrix3d::from_rows(p, axis3(), p);
    let fc = matrix3d::from_cols(p, axis3(), p);
    let fe = matrix3d::from_elements(ANG3, 0.0, 0.0, 0.0, ANG3, 0.0, 0.0, 0.0, 1.0);
    let s3 = -fe + 2.0 * fe;
    rs.extend([
        q.x * W, q.y * W, q.z * W,
        vm3.x * W, vm3.z * W,
        r3.det() * W,
        e[2][0] * W,
        fr[1][2] * W,
        fc[1][2] * W,
        (s3 * p).x * W,
    ]);

    let ea = vect3d::new(EA.0, EA.1, EA.2);
    let v = vect3d::new(EV.0, EV.1, EV.2);
    rs.extend([
        (ea * 2.0).x * W,
        (ea * 0.5).y * W,
        (-ea).z * W,
        (ea * 0.25).x * W,
        (ea * v) * W,
        (ea - v).x * W,
    ]);

    let q = qdata();
    let p = vect3d::new(A.0, A.1, A.2);
    let r = quaternd::from_axis_angle(axis3(), QANG);
    let fe = quaternd::from_euler_angles(p);
    let h = (q * r).conj();
    let rv = q.rotate(p);
    let u = (q * 2.0 - -q + quaternd::identity()).unit();
    rs.extend([
        rv.x * W, rv.y * W, rv.z * W,
        h.t * W, h.v.x * W,
        q.dot(r) * W,
        fe.v.y * W, fe.t * W,
        u.t * W, u.v.z * W,
        q.rotation_matrix()[0][1] * W,
        r.get_euler_angles().z * W,
        (q * 0.5).norm() * W,
    ]);

    // ShNode: let-shadowed entity var + skip-field passthrough.
    let sv = vect2d::new(0.6, -0.8) * 2.0;
    rs.extend([
        (sv.x - 1.0) * W,
        (sv.y + 0.35) * W,
    ]);

    rs
}

#[test]
fn cost_matches_runtime_vector_math() {
    let (mut space, params) = build();
    let expected: f64 = expected_residuals().iter().map(|r| r * r).sum();
    let cost = space.calc_cost(&params);
    assert!((cost - expected).abs() < 1e-10 * expected,
        "cost={} expected={}", cost, expected);
}

#[test]
fn gradient_matches_finite_differences() {
    let (mut space, params) = build();
    let n = params.len();
    assert_eq!(n, 25, "vect3/vect2 params + scalars + euler + quaternion + shadow node");

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
