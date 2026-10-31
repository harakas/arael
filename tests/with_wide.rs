//! The `with_wide` precision scope in a constraint body: the wrapped value is
//! computed in f64 and narrowed back to the body's scalar, the derivatives
//! stay in the body's scalar, an f64 body is unchanged, and the scope
//! composes with `branch`.

use arael::matrix::matrix3d;
use arael::model::{EulerAngleParam, JacobianModel, MixedEulerAngleParam, Model, Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};
use arael::utils::Float;
use arael::vect::{vect3, vect3d, vect3f};

/// A residual that loses digits in f32: the parameter is scaled up and
/// added to a big offset before the measurement nearly cancels it, so an
/// f32 evaluation rounds the sum at 1e-3 while the residual itself is
/// about 1e-2. In f64 the sum is exact to 1e-12. The `identity` barrier
/// keeps the simplifier from reordering the sum so that the cancellation
/// happens first, as a projection chain's rounding cannot be reordered.
#[arael::model]
#[arael(constraint(hb, { [with_wide(identity(scope.v * 1000.0 + scope.big) - scope.d)] }))]
struct Scope<T: Float> {
    v: Param<T>,
    big: T,
    d: T,
    hb: SelfBlock<Scope<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, { [identity(flat.v * 1000.0 + flat.big) - flat.d] }))]
struct Flat<T: Float> {
    v: Param<T>,
    big: T,
    d: T,
    hb: SelfBlock<Flat<T>, T>,
}

/// The scope inside a branch arm, the shape of a guarded residual.
#[arael::model]
#[arael(constraint(hb, { [branch(guard.d, with_wide(identity(guard.v * 1000.0 + guard.big) - guard.d), 0.0)] }))]
struct Guard<T: Float> {
    v: Param<T>,
    big: T,
    d: T,
    hb: SelfBlock<Guard<T>, T>,
}

#[arael::model]
#[arael(root, f32, jacobian)]
struct ScopeF32 { rows: refs::Vec<Scope<f32>> }

#[arael::model]
#[arael(root, f32, jacobian)]
struct FlatF32 { rows: refs::Vec<Flat<f32>> }

#[arael::model]
#[arael(root, f32, jacobian)]
struct GuardF32 { rows: refs::Vec<Guard<f32>> }

#[arael::model]
#[arael(root, jacobian)]
struct ScopeF64 { rows: refs::Vec<Scope<f64>> }

#[arael::model]
#[arael(root, jacobian)]
struct FlatF64 { rows: refs::Vec<Flat<f64>> }

const N: usize = 16;

/// The rows: `v` near 0.1234, `big` 1e4, `d` so that the residual is
/// about 1e-2 and differs per row. Every value is an exact f32, so the
/// two precisions see the same data.
fn row_values(i: usize) -> (f32, f32, f32) {
    let v = 0.1234f32 + i as f32 * 1e-3;
    let big = 1e4f32;
    let d = (v as f64 * 1000.0 + big as f64 - 0.01 * (i as f64 + 1.0)) as f32;
    (v, big, d)
}

macro_rules! build {
    ($root:ident, $row:ident, $t:ty) => {{
        let mut rows = refs::Vec::new();
        for i in 0..N {
            let (v, big, d) = row_values(i);
            rows.push($row { v: Param::new(v as $t), big: big as $t, d: d as $t, hb: SelfBlock::new() });
        }
        $root { rows }
    }};
}

/// The residuals and the single partial of each row, and the gradient.
fn evaluate<P, T>(m: &mut P) -> (Vec<f64>, Vec<f64>, Vec<f64>)
where
    P: LmProblem<T> + RootProblem<T> + JacobianModel<T>,
    T: Float,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let jac = m.calc_jacobian(&x);
    let residuals: Vec<f64> = jac.rows.iter().map(|r| r.residual.to_f64().unwrap()).collect();
    let partials: Vec<f64> = jac.rows.iter()
        .map(|r| r.entries.iter().map(|(_, d)| d.to_f64().unwrap()).sum())
        .collect();
    let mut g = vec![T::zero(); x.len()];
    let mut coo = CooMatrix::new(x.len());
    m.calc_grad_hessian_sparse(&x, &mut g, &mut coo);
    (residuals, partials, g.iter().map(|v| v.to_f64().unwrap()).collect())
}

fn rel(a: f64, b: f64) -> f64 { (a - b).abs() / b.abs().max(1e-300) }

#[test]
fn the_f32_value_is_the_f64_value() {
    let (r64, _, g64) = evaluate(&mut build!(FlatF64, Flat, f64));
    let (r_scope, _, g_scope) = evaluate(&mut build!(ScopeF32, Scope, f32));
    let (r_flat, _, _) = evaluate(&mut build!(FlatF32, Flat, f32));
    assert_eq!(r64.len(), N);
    let mut worst_scope = 0.0f64;
    let mut worst_flat = 0.0f64;
    for i in 0..N {
        worst_scope = worst_scope.max(rel(r_scope[i], r64[i]));
        worst_flat = worst_flat.max(rel(r_flat[i], r64[i]));
    }
    // An f32 residual of 1e-2 narrowed from the exact value carries f32's
    // own rounding, 6e-8; the plain f32 chain rounds the 1e4 sum at 1e-3.
    assert!(worst_scope < 1e-6, "with_wide residuals off by {worst_scope:e} relative");
    assert!(worst_flat > 1e-3, "the plain f32 residuals were expected to lose digits, worst {worst_flat:e}");
    for (a, b) in g_scope.iter().zip(&g64) {
        assert!(rel(*a, *b) < 1e-5, "with_wide gradient {a} against f64 {b}");
    }
}

#[test]
fn the_derivatives_stay_in_f32() {
    let (_, d_scope, _) = evaluate(&mut build!(ScopeF32, Scope, f32));
    let (_, d_flat, _) = evaluate(&mut build!(FlatF32, Flat, f32));
    assert_eq!(d_scope, d_flat, "the scope must not change the Jacobian");
    assert!(d_scope.iter().all(|&d| d == 1000.0));
}

#[test]
fn an_f64_body_is_unchanged() {
    let (r_scope, d_scope, g_scope) = evaluate(&mut build!(ScopeF64, Scope, f64));
    let (r_flat, d_flat, g_flat) = evaluate(&mut build!(FlatF64, Flat, f64));
    assert_eq!(r_scope, r_flat);
    assert_eq!(d_scope, d_flat);
    assert_eq!(g_scope, g_flat);
}

#[test]
fn the_scope_composes_with_a_branch() {
    let (r64, _, _) = evaluate(&mut build!(FlatF64, Flat, f64));
    let (r_guard, d_guard, _) = evaluate(&mut build!(GuardF32, Guard, f32));
    // d is positive, so every row takes the wide arm.
    for i in 0..N {
        assert!(rel(r_guard[i], r64[i]) < 1e-6, "guarded with_wide residual {} against f64 {}", r_guard[i], r64[i]);
    }
    assert!(d_guard.iter().all(|&d| d == 1000.0));
}

// --- a MixedEulerAngleParam's rotation is read in f64 inside the scope ---

/// The camera-frame transform of a far point against a near-equal
/// measurement, the first stage of a projection. The rotation is a cached
/// per-pose matrix: a `MixedEulerAngleParam` keeps it in f64 next to the
/// rounded copy the derivatives read, and the scope reads the f64 one.
#[arael::model]
#[arael(constraint(hb, {
    let q = rotscopemixed.ea.rotation_matrix() * rotscopemixed.p + rotscopemixed.t;
    [with_wide(q.x - rotscopemixed.d)]
}))]
struct RotScopeMixed<T: Float> {
    ea: MixedEulerAngleParam<T>,
    p: vect3<T>,
    t: vect3<T>,
    d: T,
    hb: SelfBlock<RotScopeMixed<T>, T>,
}

/// The control: the same transform without the scope.
#[arael::model]
#[arael(constraint(hb, {
    let q = rotplainmixed.ea.rotation_matrix() * rotplainmixed.p + rotplainmixed.t;
    [q.x - rotplainmixed.d]
}))]
struct RotPlainMixed<T: Float> {
    ea: MixedEulerAngleParam<T>,
    p: vect3<T>,
    t: vect3<T>,
    d: T,
    hb: SelfBlock<RotPlainMixed<T>, T>,
}

/// A plain rotation param in the scope: its cached rotation is `T`, so the
/// scope is bounded by that rounding.
#[arael::model]
#[arael(constraint(hb, {
    let q = rotscopeeuler.ea.rotation_matrix() * rotscopeeuler.p + rotscopeeuler.t;
    [with_wide(q.x - rotscopeeuler.d)]
}))]
struct RotScopeEuler<T: Float> {
    ea: EulerAngleParam<T>,
    p: vect3<T>,
    t: vect3<T>,
    d: T,
    hb: SelfBlock<RotScopeEuler<T>, T>,
}

#[arael::model]
#[arael(root, f32, jacobian, wide = f64)]
struct RotScopeMixedF32 { rows: refs::Vec<RotScopeMixed<f32>> }

#[arael::model]
#[arael(root, f32, jacobian)]
struct RotPlainMixedF32 { rows: refs::Vec<RotPlainMixed<f32>> }

#[arael::model]
#[arael(root, f32, jacobian)]
struct RotScopeEulerF32 { rows: refs::Vec<RotScopeEuler<f32>> }

/// A camera at a general rotation looking at a point 2000 units out:
/// an f32 transform carries about 1e-4 of error into a residual of 1e-2.
/// Every value is an exact f32.
fn rotation_inputs(i: usize) -> (vect3f, vect3f, vect3f) {
    let ea = vect3f::new(0.3, -0.2, 0.7 + i as f32 * 1e-2);
    let p = vect3f::new(1000.0 + i as f32 * 1.5, 2000.0 - i as f32 * 2.25, -500.0 + i as f32 * 0.5);
    let t = vect3f::new(12.5, -3.25, 7.0);
    (ea, p, t)
}

/// The measurement, an f32, and the residual it leaves in f64.
fn rotation_reference(i: usize) -> (f32, f64) {
    let (ea, p, t) = rotation_inputs(i);
    let q = matrix3d::rotation_from_euler_angles(ea.cast()) * p.cast::<f64>() + t.cast::<f64>();
    let d = (q.x - 0.01 * (i as f64 + 1.0)) as f32;
    (d, q.x - d as f64)
}

macro_rules! build_rotation {
    ($root:ident, $row:ident, $ctor:expr) => {{
        let mut rows = refs::Vec::new();
        for i in 0..N {
            let (ea, p, t) = rotation_inputs(i);
            let (d, _) = rotation_reference(i);
            rows.push($row { ea: $ctor(ea), p, t, d, hb: SelfBlock::new() });
        }
        $root { rows }
    }};
}

/// The worst relative error of the residuals against the f64 reference.
fn worst_rotation_error(r: &[f64]) -> f64 {
    r.iter().enumerate()
        .map(|(i, v)| rel(*v, rotation_reference(i).1))
        .fold(0.0, f64::max)
}

#[test]
fn a_mixed_rotation_is_read_in_f64_inside_the_scope() {
    let mixed = |ea: vect3f| MixedEulerAngleParam::new(ea.cast::<f64>());
    let (r, d_scope, _) = evaluate(&mut build_rotation!(RotScopeMixedF32, RotScopeMixed, mixed));
    let worst = worst_rotation_error(&r);
    assert!(worst < 1e-6, "mixed rotation in the scope: residuals off by {worst:e} relative");
    // The control: the same chain in f32 loses the residual's digits. The
    // derivatives read the rounded rotation and are those of the control
    // to f32 rounding (the two bodies' CSE orders the sums differently).
    let (r_plain, d_plain, _) = evaluate(&mut build_rotation!(RotPlainMixedF32, RotPlainMixed, mixed));
    let worst = worst_rotation_error(&r_plain);
    assert!(worst > 1e-3, "the plain f32 transform was expected to lose digits, worst {worst:e}");
    for (a, b) in d_scope.iter().zip(&d_plain) {
        assert!(rel(*a, *b) < 1e-6, "scope derivative {a} against the control's {b}");
    }
    // A plain param's rotation enters the scope as its f32 rounding.
    let (r_euler, _, _) = evaluate(&mut build_rotation!(RotScopeEulerF32, RotScopeEuler, EulerAngleParam::new));
    let worst = worst_rotation_error(&r_euler);
    assert!(worst > 1e-4, "an f32 EulerAngleParam in the scope was expected to be bounded by its rounding, worst {worst:e}");
}

/// Re-centering the mixed param's reference a thousand times leaves it a
/// rotation to f64 rounding; the f32 param's reference drifts.
#[test]
fn the_mixed_reference_rotation_is_kept_in_f64() {
    fn drift<P: Model>(p: &mut P, rotation: fn(&mut P) -> matrix3d) -> f64 {
        let mut x: Vec<f32> = Vec::new();
        p.serialize_params(&mut x);
        for _ in 0..1000 {
            p.update_params(&[1e-3f32, -2e-3, 5e-4]);
            p.advance_params(&mut x);
        }
        let w = rotation(p);
        let e = w * w.transpose() - matrix3d::identity();
        (0..3).map(|r| e[r].x.abs().max(e[r].y.abs()).max(e[r].z.abs())).fold(0.0, f64::max)
    }
    let mut mixed = MixedEulerAngleParam::<f32>::new(vect3d::new(0.3, -0.2, 0.7));
    let d = drift(&mut mixed, |p| { p.__precompute(); p.rotation_matrix_wide });
    assert!(d < 1e-12, "MixedEulerAngleParam reference drift {d:e}");
    let rounded = mixed.rotation_matrix_wide.cast::<f32>();
    assert!((0..3).all(|i| mixed.rotation_matrix[i].x == rounded[i].x
        && mixed.rotation_matrix[i].y == rounded[i].y && mixed.rotation_matrix[i].z == rounded[i].z));
    let mut plain = EulerAngleParam::<f32>::new(vect3f::new(0.3, -0.2, 0.7));
    let d = drift(&mut plain, |p| { p.__precompute(); p.rotation_matrix.cast::<f64>() });
    assert!(d > 1e-7, "an f32 EulerAngleParam's reference was expected to drift, got {d:e}");
}
