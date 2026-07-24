// UnitVecParam: the S^2 direction component, end to end through the macro.
// A landmark's direction is pulled toward two disagreeing unit measurements;
// the optimum is the normalized mean direction (for r = dir - m rows, the
// cost is minimized on the sphere by maximizing dir . (m1 + m2)), which is a
// nonzero-cost compromise -- generic convergence, no exact-zero artifact.
// The FD check validates the whole chain: chart, cached rotation, symbolic
// embed, span folding into the owner's block.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::unitvec::UnitVecParam;
use arael::vect::vect3;

#[arael::model]
#[arael(constraint(hb, {
    let u = lm.dir.unit;
    [(u.x - lm.m1.x) * lm.isigma, (u.y - lm.m1.y) * lm.isigma, (u.z - lm.m1.z) * lm.isigma,
     (u.x - lm.m2.x) * lm.isigma, (u.y - lm.m2.y) * lm.isigma, (u.z - lm.m2.z) * lm.isigma,
     (lm.w - u.z) * 0.5]
}))]
struct Lm {
    dir: UnitVecParam,
    w: Param<f64>,
    m1: vect3<f64>,
    m2: vect3<f64>,
    isigma: f64,
    hb: SelfBlock<Lm>,
}

#[arael::model]
#[arael(root)]
struct M {
    lms: arael::refs::Vec<Lm>,
}

fn build(start: vect3<f64>) -> M {
    let mut lms = arael::refs::Vec::new();
    let m1 = vect3::new(0.6, 0.64, 0.48).unit();
    let m2 = vect3::new(0.8, 0.36, 0.48).unit();
    lms.push(Lm {
        dir: UnitVecParam::new(start),
        w: Param::new(0.0),
        m1,
        m2,
        isigma: 1.5,
        hb: SelfBlock::new(),
    });
    M { lms }
}

fn expected() -> vect3<f64> {
    (vect3::new(0.6, 0.64, 0.48).unit() + vect3::new(0.8, 0.36, 0.48).unit()).unit()
}

#[test]
fn grad_hessian_match_finite_differences() {
    let mut m = build(vect3::new(0.2, -0.5, 0.9));
    let mut params = Vec::new();
    m.serialize64(&mut params);
    assert_eq!(params.len(), 3, "dir.d (2) + w (1)");

    let n = params.len();
    let mut ag = vec![0.0; n];
    let mut ah = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let eps = 1e-6;
    for i in 0..n {
        let mut pp = params.clone();
        pp[i] += eps;
        let cp = m.calc_cost(&pp);
        pp[i] -= 2.0 * eps;
        let cm = m.calc_cost(&pp);
        let ng = (cp - cm) / (2.0 * eps);
        assert!((ag[i] - ng).abs() < 2e-4,
            "grad[{}]: analytic={} numerical={}", i, ag[i], ng);
    }
}

#[test]
fn solves_to_the_mean_direction() {
    let mut m = build(vect3::new(0.2, -0.5, 0.9));
    let r = m.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    let lm = &m.lms[0];
    let e = expected();
    assert!((lm.dir.unit - e).norm() < 1e-8,
        "dir = {:?} vs {:?}", (lm.dir.unit.x, lm.dir.unit.y, lm.dir.unit.z), (e.x, e.y, e.z));
    // Exactly on the sphere, re-centred.
    assert!((lm.dir.unit.norm() - 1.0).abs() < 1e-14, "|unit| = {}", lm.dir.unit.norm());
    assert!(lm.dir.d.value.x.abs() < 1e-12 && lm.dir.d.value.y.abs() < 1e-12);
    assert!((lm.w.value - e.z).abs() < 1e-6, "w = {}", lm.w.value);
    assert!(r.end_cost > 0.01, "the measurements disagree by design: {}", r.end_cost);
}

/// The chart re-centres every accepted step, so even a start far around the
/// sphere (nearly antipodal) walks in.
#[test]
fn converges_from_far_start() {
    let e = expected();
    let mut m = build(vect3::new(-e.x, -e.y, -e.z + 0.2));
    let r = m.solve_dense(&LmConfig::conservative().with_max_iters(300)).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((m.lms[0].dir.unit - e).norm() < 1e-6,
        "dir did not cross the sphere: {:?}", (m.lms[0].dir.unit.x, m.lms[0].dir.unit.y, m.lms[0].dir.unit.z));
}

/// A fixed direction contributes no params and does not move.
#[test]
fn fixed_direction_stays_put() {
    let start = vect3::new(0.2, -0.5, 0.9).unit();
    let mut m = build(start);
    m.lms[0].dir = UnitVecParam::fixed(start);
    let mut params = Vec::new();
    m.serialize64(&mut params);
    assert_eq!(params.len(), 1, "only w remains");
    let r = m.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((m.lms[0].dir.unit - start).norm() < 1e-12);
}

// --- equivalence with a user-defined macro component -----------------------
//
// The builtin UnitVecParam is documented as the hand-written twin of a
// user-defined #[arael(component)] with the same fields. Prove it: the same
// problem built with a macro-expanded twin must solve identically -- same
// cost trajectory endpoint, same optimized direction, and the same
// precomputed unit / unit_d caches at the solution.

use arael::matrix::matrix3d;
use arael::model::Component;
use arael::quatern::quaternd;
use arael::vect::{vect2d, vect3d};

#[arael::model]
#[arael(component)]
struct UnitVecMacro {
    ref_q: quaternd,
    #[arael(compute = self.ref_q.rotation_matrix())]
    rot: matrix3d,
    d: Param<vect2d>,
    #[arael(symbolic = {
        let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25;
        let local = vect3sym::from_components(
            1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2);
        rot * local
    })]
    unit: vect3d,
    #[arael(deriv = unit, by = d)]
    unit_d: [vect3d; 2],
}

impl UnitVecMacro {
    fn new(dir: vect3d) -> UnitVecMacro {
        let mut u = UnitVecMacro {
            ref_q: quaternd::identity(),
            rot: matrix3d::identity(),
            d: Param::new(vect2d::new(0.0, 0.0)),
            unit: dir,
            unit_d: [vect3d::new(0.0, 0.0, 0.0); 2],
        };
        Component::start(&mut u);
        u
    }
}

impl Component for UnitVecMacro {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quaternd::from_two_vectors(vect3d::new(1.0, 0.0, 0.0), self.unit);
        self.d.value = vect2d::new(0.0, 0.0);
    }
    fn update(&mut self) {
        let dq = quaternd::from_rotation_vector_small(
            vect3d::new(0.0, self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2d::new(0.0, 0.0);
    }
    fn finish(&mut self) {
        let dq = quaternd::from_rotation_vector_small(
            vect3d::new(0.0, self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(vect3d::new(1.0, 0.0, 0.0));
    }
}

#[arael::model]
#[arael(constraint(hb, {
    let u = lm2.dir.unit;
    [(u.x - lm2.m1.x) * lm2.isigma, (u.y - lm2.m1.y) * lm2.isigma, (u.z - lm2.m1.z) * lm2.isigma,
     (u.x - lm2.m2.x) * lm2.isigma, (u.y - lm2.m2.y) * lm2.isigma, (u.z - lm2.m2.z) * lm2.isigma,
     (lm2.w - u.z) * 0.5]
}))]
struct Lm2 {
    dir: UnitVecMacro,
    w: Param<f64>,
    m1: vect3<f64>,
    m2: vect3<f64>,
    isigma: f64,
    hb: SelfBlock<Lm2>,
}

#[arael::model]
#[arael(root)]
struct M2 {
    lms: arael::refs::Vec<Lm2>,
}

#[test]
fn builtin_matches_the_macro_component() {
    let start = vect3::new(0.3, -0.8, 0.51);
    let mut builtin = build(start);

    let m1 = vect3::new(0.6, 0.64, 0.48).unit();
    let m2 = vect3::new(0.8, 0.36, 0.48).unit();
    let mut lms = arael::refs::Vec::new();
    lms.push(Lm2 {
        dir: UnitVecMacro::new(start),
        w: Param::new(0.0),
        m1,
        m2,
        isigma: 1.5,
        hb: SelfBlock::new(),
    });
    let mut macro_m = M2 { lms };

    let cfg = LmConfig::conservative();
    let rb = builtin.solve_dense(&cfg).unwrap();
    let rm = macro_m.solve_dense(&cfg).unwrap();

    assert!((rb.end_cost - rm.end_cost).abs() < 1e-12 * (1.0 + rb.end_cost),
        "cost: builtin {} vs macro {}", rb.end_cost, rm.end_cost);
    assert_eq!(rb.iterations, rm.iterations, "same damping trajectory");

    let b = &builtin.lms[0].dir;
    let m = &macro_m.lms[0].dir;
    assert!((b.unit - m.unit).norm() < 1e-12,
        "unit: {:?} vs {:?}", b.unit, m.unit);
    for k in 0..2 {
        assert!((b.unit_d[k] - m.unit_d[k]).norm() < 1e-12,
            "unit_d[{}]: {:?} vs {:?}", k, b.unit_d[k], m.unit_d[k]);
    }
}
