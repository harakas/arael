// The REVIEW5 consistency batch: cross-type API symmetry. Every math
// type casts, compares approximately and reports finiteness; the refs
// collections share construction and ref-accessor surfaces; the rotation
// param has the same constructors its quaternion has; LmConfig prints.

use arael::matrix::{matrix2d, matrix3d};
use arael::model::QuaternionParam;
use arael::quatern::{quaternd, quaternf};
use arael::refs;
use arael::se3::{se3d, se3f};
use arael::simple_lm::{Dense, LmConfig};
use arael::vect::{vect3d, Similar};

#[test]
fn se3_cast_similar_finite() {
    let t = se3d::new(vect3d::new(0.1, 0.2, 0.3), vect3d::new(0.01, 0.02, 0.03));
    let f: se3f = t.cast();
    let back: se3d = f.cast();
    assert!(t.similar(back) || (t.d - back.d).norm() < 1e-6);
    assert!(t.is_finite());
    let mut bad = t;
    bad.w.x = f64::NAN;
    assert!(!bad.is_finite());
}

#[test]
fn quatern_similar_trait_and_finite() {
    let q = quaternd::from_euler_angles(vect3d::new(0.1, -0.2, 0.3));
    // The Similar TRAIT (generic contexts), same result as the inherent
    // method.
    fn approx<S: Similar>(a: S, b: S) -> bool { a.similar(b) }
    assert!(approx(q, q));
    assert!(q.is_finite());
    let bad = quaternd::new(f64::INFINITY, q.v);
    assert!(!bad.is_finite());
    let _f: quaternf = q.cast();
}

#[test]
fn matrix_is_finite() {
    assert!(matrix3d::identity().is_finite());
    assert!(matrix2d::identity().is_finite());
    let mut m = matrix3d::identity();
    m[1].y = f64::NAN;
    assert!(!m.is_finite());
}

#[test]
fn refs_construction_and_ref_accessors() {
    // Vec: first_ref/last_ref (Deque already had front_ref/back_ref).
    let v = refs::Vec::from_slice(&[10, 20, 30]);
    assert_eq!(v[v.first_ref().unwrap()], 10);
    assert_eq!(v[v.last_ref().unwrap()], 30);
    assert!(refs::Vec::<i32>::new().first_ref().is_none());

    // From<std::vec::Vec> on all three collections.
    let v: refs::Vec<i32> = vec![1, 2].into();
    assert_eq!(v.len(), 2);
    let d: refs::Deque<i32> = vec![1, 2, 3].into();
    assert_eq!(d.len(), 3);
    assert_eq!(d[d.front_ref().unwrap()], 1);
    let a: refs::Arena<i32> = vec![4, 5].into();
    assert_eq!(a.len(), 2);

    // from_slice on Arena (Vec and Deque already had it).
    let a = refs::Arena::from_slice(&[7, 8, 9]);
    assert_eq!(a.len(), 3);
    assert_eq!(a[a.refs().nth(2).unwrap()], 9);
}

#[test]
fn quaternion_param_constructors() {
    let axis = vect3d::new(0.0, 0.0, 1.0);
    let angle = 0.3;
    let p = QuaternionParam::from_axis_angle(axis, angle);
    let q = quaternd::from_axis_angle(axis, angle);
    assert!(p.value.similar(q));

    let m = q.rotation_matrix();
    let p = QuaternionParam::from_rotation_matrix(m);
    // Same rotation up to quaternion sign.
    assert!(p.value.similar(q) || p.value.similar(-q));
}

#[test]
fn config_debug_and_dense_default() {
    let cfg = LmConfig::<f64>::default().with_max_iters(7);
    let s = format!("{:?}", cfg);
    assert!(s.contains("max_iters: 7"), "{}", s);
    let _ = Dense::default();
}
