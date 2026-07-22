// A user-defined #[arael(component)] generic over its scalar.
//
// One definition, `T: Float`, instantiated in an f64 model and an f32
// model. Covers the generic spellings the field classifier must size
// (quatern<T>, matrix3<T>, Param<vect2<T>>, vect3<T>, bare Param<T>), the
// literal wrapping in the generated symbolic precompute, and the shared
// non-generic sym companion.

use arael::model::{Component, Model, Param, SelfBlock};
use arael::matrix::matrix3;
use arael::quatern::quatern;
use arael::refs;
use arael::simple_lm::{LmConfig, LmProblem};
use arael::unitvec::UnitVecParam;
use arael::utils::Float;
use arael::vect::{vect2, vect3, vect3d, vect3f};

// ---------------------------------------------------------------- components

/// Unit direction on S^2, 2 DOF: the plane_slam_demo component, generic.
#[arael::model]
#[arael(component)]
#[derive(Clone)]
struct Dir<T: Float> {
    /// Rotation taking (1, 0, 0) into `unit`. Solver-internal chart.
    ref_q: quatern<T>,
    /// Chart matrix, cached per reference move.
    #[arael(compute = self.ref_q.rotation_matrix())]
    rot: matrix3<T>,
    /// The 2-DOF tangent delta.
    d: Param<vect2<T>>,
    /// The direction constraint bodies read.
    #[arael(symbolic = {
        let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25;
        let local = vect3sym::from_components(
            1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2);
        rot * local
    })]
    unit: vect3<T>,
    #[arael(deriv = unit, by = d)]
    unit_d: [vect3<T>; 2],
}

impl<T: Float> Dir<T> {
    fn ex() -> vect3<T> {
        vect3::new(T::one(), T::zero(), T::zero())
    }
    fn new(direction: vect3<T>) -> Dir<T> {
        let mut u = Dir {
            ref_q: quatern::identity(),
            rot: matrix3::identity(),
            d: Param::new(vect2::new(T::zero(), T::zero())),
            unit: direction,
            unit_d: [vect3::new(T::zero(), T::zero(), T::zero()); 2],
        };
        Component::start(&mut u);
        u
    }
}

impl<T: Float> Component for Dir<T> {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quatern::from_two_vectors(Self::ex(), self.unit);
        self.d.value = vect2::new(T::zero(), T::zero());
    }
    fn update(&mut self) {
        let dq = quatern::from_rotation_vector_small(
            vect3::new(T::zero(), self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2::new(T::zero(), T::zero());
    }
    fn finish(&mut self) {
        let dq = quatern::from_rotation_vector_small(
            vect3::new(T::zero(), self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

/// A 1-DOF component with a bare scalar param: `g2 = 1 + g^2`, always
/// at least one. Exercises `Param<T>` sizing and a scalar symbolic field.
#[arael::model]
#[arael(component)]
#[derive(Clone)]
struct Gain<T: Float> {
    g: Param<T>,
    #[arael(symbolic = g * g + 1.0)]
    g2: T,
}

impl<T: Float> Gain<T> {
    fn new(g: T) -> Gain<T> {
        Gain { g: Param::new(g), g2: g * g + T::one() }
    }
}

impl<T: Float> Component for Gain<T> {
    fn start(&mut self) {}
    fn update(&mut self) {}
    fn finish(&mut self) {
        let g = self.g.value;
        self.g2 = g * g + T::one();
    }
}

// ------------------------------------------------------------------- models
// Fit gain * direction to a measured vector: 2 + 1 params against a
// 3-component residual, exactly determined.

#[arael::model]
#[arael(constraint(hb, {
    [lm64.gain.g2 * lm64.dir.unit.x - lm64.measured.x,
     lm64.gain.g2 * lm64.dir.unit.y - lm64.measured.y,
     lm64.gain.g2 * lm64.dir.unit.z - lm64.measured.z]
}))]
struct Lm64 {
    dir: Dir<f64>,
    gain: Gain<f64>,
    measured: vect3d,
    hb: SelfBlock<Lm64>,
}

#[arael::model]
#[arael(root)]
struct World64 {
    lms: refs::Vec<Lm64>,
}

#[arael::model]
#[arael(constraint(hb, {
    [lm32.gain.g2 * lm32.dir.unit.x - lm32.measured.x,
     lm32.gain.g2 * lm32.dir.unit.y - lm32.measured.y,
     lm32.gain.g2 * lm32.dir.unit.z - lm32.measured.z]
}))]
struct Lm32 {
    dir: Dir<f32>,
    gain: Gain<f32>,
    measured: vect3f,
    hb: SelfBlock<Lm32, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct World32 {
    lms: refs::Vec<Lm32>,
}

// The built-in components spelled with explicit generic args (previously
// only the bare/alias spellings were recognized as component fields).
#[arael::model]
#[arael(constraint(hb, {
    [norm.v.unit.x - norm.measured.x,
     norm.v.unit.y - norm.measured.y,
     norm.v.unit.z - norm.measured.z]
}))]
struct Norm {
    v: UnitVecParam<f32>,
    measured: vect3f,
    hb: SelfBlock<Norm, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct NormWorld {
    lms: refs::Vec<Norm>,
}

// -------------------------------------------------------------------- tests

/// Both components fold their params into the owner's span.
#[test]
fn generic_component_param_count() {
    assert_eq!(<Dir<f64> as Model>::PARAM_COUNT, 2);
    assert_eq!(<Gain<f32> as Model>::PARAM_COUNT, 1);
    assert_eq!(<Lm64 as Model>::PARAM_COUNT, 3);
    assert_eq!(<Lm32 as Model>::PARAM_COUNT, 3);
}

#[test]
fn generic_component_solves_f64() {
    let mut w = World64 { lms: refs::Vec::new() };
    let target = vect3d::new(1.0, 2.0, -0.5);
    w.lms.push(Lm64 {
        dir: Dir::new(vect3d::new(1.0, 0.0, 0.3)),
        gain: Gain::new(0.5),
        measured: target,
        hb: SelfBlock::new(),
    });
    let r = w.solve_sparse(&LmConfig::default());
    assert!(r.end_cost < 1e-16, "cost {}", r.end_cost);
    let lm = w.lms.iter().next().unwrap();
    let fit = lm.dir.unit * lm.gain.g2;
    assert!((fit - target).norm() < 1e-8,
        "fit ({}, {}, {})", fit.x, fit.y, fit.z);
    assert!((lm.dir.unit.norm() - 1.0).abs() < 1e-12, "left the sphere");
}

#[test]
fn generic_component_solves_f32() {
    let mut w = World32 { lms: refs::Vec::new() };
    let target = vect3f::new(1.0, 2.0, -0.5);
    w.lms.push(Lm32 {
        dir: Dir::new(vect3f::new(1.0, 0.0, 0.3)),
        gain: Gain::new(0.5),
        measured: target,
        hb: SelfBlock::new(),
    });
    let r = w.solve_sparse(&LmConfig::default());
    assert!(r.end_cost < 1e-9, "cost {}", r.end_cost);
    let lm = w.lms.iter().next().unwrap();
    let fit = lm.dir.unit * lm.gain.g2;
    assert!((fit - target).norm() < 1e-3,
        "fit ({}, {}, {})", fit.x, fit.y, fit.z);
}

/// `UnitVecParam<f32>` written with its generic args is sized and solved
/// like the `UnitVecParamF` alias.
#[test]
fn builtin_component_with_generic_args() {
    assert_eq!(<Norm as Model>::PARAM_COUNT, 2);
    let mut w = NormWorld { lms: refs::Vec::new() };
    let target = vect3f::new(0.0, 3.0, 4.0).unit();
    w.lms.push(Norm {
        v: UnitVecParam::new(vect3f::new(1.0, 0.2, 0.0)),
        measured: target,
        hb: SelfBlock::new(),
    });
    let r = w.solve_sparse(&LmConfig::default());
    assert!(r.end_cost < 1e-9, "cost {}", r.end_cost);
    let lm = w.lms.iter().next().unwrap();
    assert!((lm.v.unit - target).norm() < 1e-3,
        "unit ({}, {}, {})", lm.v.unit.x, lm.v.unit.y, lm.v.unit.z);
}
