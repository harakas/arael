// Serde round-trip tests for parameter types.
//
// Param's index is #[serde(skip)]; a plain skip restores 0, which is a
// VALID parameter index -- a loaded model where update* runs before
// serialize_params* would silently read data[0..] for every param. The
// sentinel u32::MAX must survive deserialization. The euler angle param
// types already get this right via their custom Deserialize impls;
// Param must behave the same, including tolerating a missing "optimize"
// field (defaults to true).

use arael::model::{Model, Param, SimpleEulerAngleParam};
use arael::vect::vect3;

#[test]
fn param_index_restores_sentinel_not_zero() {
    let mut p = Param::new(1.5_f64);
    // Assign a real index by serializing the parameter vector.
    let mut data: Vec<f64> = Vec::new();
    p.serialize_params64(&mut data);
    assert_eq!(p.index(), 0, "first param gets index 0");

    let json = serde_json::to_string(&p).expect("serialize");
    let q: Param<f64> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(q.index(), u32::MAX,
        "deserialized index must be the inactive sentinel, not a valid index");
    assert_eq!(q.value, 1.5);
    assert!(q.optimize);
}

#[test]
fn param_fixed_round_trips() {
    let p = Param::fixed(-2.25_f64);
    let json = serde_json::to_string(&p).expect("serialize");
    let q: Param<f64> = serde_json::from_str(&json).expect("deserialize");
    assert!(!q.optimize);
    assert_eq!(q.value, -2.25);
    assert_eq!(q.index(), u32::MAX);
}

#[test]
fn param_missing_optimize_defaults_true() {
    // The euler angle params tolerate JSON without "optimize"; Param
    // must accept the same shape instead of erroring.
    let q: Param<f64> = serde_json::from_str(r#"{"value": 2.0}"#)
        .expect("deserialize without optimize field");
    assert!(q.optimize);
    assert_eq!(q.value, 2.0);
    assert_eq!(q.index(), u32::MAX);
}

#[test]
fn vect3_f64_param_type() {
    // f64 storage: values beyond f32 mantissa range must round-trip
    // exactly through the parameter vector and serde.
    let mut p = Param::new(vect3::new(1.0e6_f64, 2.0e6, 3.0e6));
    let mut data: Vec<f64> = Vec::new();
    p.serialize_params64(&mut data);
    assert_eq!(data, vec![1.0e6, 2.0e6, 3.0e6]);
    assert_eq!(p.index(), 0);

    // Full-precision round trip through the parameter vector: values
    // like 1e6 + 0.25 must survive exactly (they would not in f32).
    let moved = vec![1.0e6 + 0.25, 2.0e6 + 0.25, 3.0e6 + 0.25];
    p.update64(&moved);
    p.deserialize_params64(&moved);
    assert_eq!(p.value.x, 1.0e6 + 0.25);
    assert_eq!(p.work().z, 3.0e6 + 0.25);

    let json = serde_json::to_string(&p).expect("serialize");
    let q: Param<vect3<f64>> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(q.value.y, 2.0e6 + 0.25);
    assert_eq!(q.index(), u32::MAX);

    // f64 euler angle params (Model requires vect3<T>: ParamType).
    let mut ea = SimpleEulerAngleParam::<f64>::new(vect3::new(0.1, 0.2, 0.3));
    let mut data: Vec<f64> = Vec::new();
    ea.serialize_params64(&mut data);
    assert_eq!(data.len(), 3);
}

#[test]
fn simple_euler_angle_param_round_trips_with_sentinel() {
    // Parity check: the EA type's custom impl already restores the
    // sentinel and defaults optimize -- keep it that way.
    let mut p = SimpleEulerAngleParam::<f32>::new(vect3::new(0.1, 0.2, 0.3));
    let mut data: Vec<f32> = Vec::new();
    p.serialize_params32(&mut data);
    assert_eq!(p.index(), 0);

    let json = serde_json::to_string(&p).expect("serialize");
    let q: SimpleEulerAngleParam<f32> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(q.index(), u32::MAX);
    assert!(q.optimize);

    let q2: SimpleEulerAngleParam<f32> =
        serde_json::from_str(r#"{"value": {"x": 0.1, "y": 0.2, "z": 0.3}}"#)
        .expect("deserialize without optimize field");
    assert!(q2.optimize);
}

#[test]
fn quaternion_param_serde_roundtrip() {
    use arael::model::QuaternionParam;
    use arael::quatern::quaternd;
    let p = QuaternionParam::from_euler_angles(vect3::new(0.1, -0.2, 0.3));
    let json = serde_json::to_string(&p).expect("serialize");
    let back: QuaternionParam<f64> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(p.optimize, back.optimize);
    // The reference quaternion round-trips and stays unit.
    assert!((p.value.t - back.value.t).abs() < 1e-12
        && (p.value.v.x - back.value.v.x).abs() < 1e-12
        && (p.value.v.y - back.value.v.y).abs() < 1e-12
        && (p.value.v.z - back.value.v.z).abs() < 1e-12);
    assert!((back.value.norm() - 1.0).abs() < 1e-12, "deserialized reference not unit");
    // A fixed param round-trips its optimize flag.
    let f: QuaternionParam<f64> =
        QuaternionParam::fixed(quaternd::from_euler_angles(vect3::new(0.2, 0.0, -0.1)));
    let back_f: QuaternionParam<f64> =
        serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
    assert!(!back_f.optimize);
}

// ---------------------------------------------------------------------------
// The compound parameter types and the math types they are built from.
//
// Each of these carries derived state -- a reference frame, a rotation
// matrix, Jacobian caches -- that is a function of the value it comes from.
// None of it is written out: it is rebuilt on load, so a file cannot
// disagree with itself. Checking only that the value survives would pass
// with every cache left at its default, so these check the rebuilt state
// too, and that the derived fields are absent from the wire.
// ---------------------------------------------------------------------------

use arael::matrix::matrix3d;
use arael::model::SelfBlock;
use arael::quatern::quaternd;
use arael::se3::se3d;
use arael::transform::TransformParam;
use arael::unitvec::UnitVecParam;
use arael::vect::{vect2d, vect3d};

fn rt<T: serde::Serialize + serde::de::DeserializeOwned>(v: &T) -> T {
    serde_json::from_str(&serde_json::to_string(v).unwrap()).unwrap()
}

// ------------------------------------------------------------------- se3

#[test]
fn se3_round_trips() {
    let a = se3d::new(vect3d::new(1.3, 0.5, -0.9), vect3d::new(0.31, 0.22, -0.44));
    let b: se3d = rt(&a);
    assert!((b.d - a.d).norm() < 1e-15);
    assert!((b.w - a.w).norm() < 1e-15);
    // And it still means the same transform.
    let (ta, qa) = a.translation_rotation();
    let (tb, qb) = b.translation_rotation();
    assert!((tb - ta).norm() < 1e-15);
    assert!((qb.v - qa.v).norm() < 1e-15 && (qb.t - qa.t).abs() < 1e-15);
}

// -------------------------------------------------------- TransformParam

#[test]
fn transform_param_round_trips_with_its_caches_rebuilt() {
    let t = vect3d::new(0.4, -0.3, 0.15);
    let q = quaternd::from_euler_angles(vect3d::new(0.05, -0.08, 0.2));
    let a = TransformParam::new(t, q);
    let b: TransformParam = rt(&a);

    assert!((b.translation - t).norm() < 1e-15);
    let dot = (b.rotation.t * q.t + b.rotation.v * q.v).abs();
    assert!((dot - 1.0).abs() < 1e-15, "rotation off, |dot| = {dot}");

    // The cached matrix is derived state and is NOT in the file; it has to
    // come back agreeing with the rotation.
    let want = q.rotation_matrix();
    for r in 0..3 {
        for c in 0..3 {
            assert!((b.rotation_matrix[r][c] - want[r][c]).abs() < 1e-15,
                "rotation_matrix[{r}][{c}] was not rebuilt");
        }
    }
    // A default-constructed cache would be the identity -- check we did not
    // simply get that.
    let ident = matrix3d::identity();
    let differs = (0..3).any(|r| (0..3).any(|c| (want[r][c] - ident[r][c]).abs() > 1e-9));
    assert!(differs, "the test rotation must not be the identity");
}

#[test]
fn transform_param_keeps_its_optimize_flags() {
    for (ot, orr) in [(true, true), (true, false), (false, true), (false, false)] {
        let mut a = TransformParam::new(vect3d::new(1.0, 2.0, 3.0), quaternd::identity());
        a.optimize_translation = ot;
        a.optimize_rotation = orr;
        let b: TransformParam = rt(&a);
        assert_eq!(b.optimize_translation, ot);
        assert_eq!(b.optimize_rotation, orr);
    }
}

#[test]
fn transform_param_writes_no_derived_state() {
    let a = TransformParam::new(vect3d::new(0.4, -0.3, 0.15),
        quaternd::from_euler_angles(vect3d::new(0.05, -0.08, 0.2)));
    let json = serde_json::to_string(&a).unwrap();
    for cache in ["rotation_matrix", "ref_translation", "ref_rotation",
                  "rotation_matrix_dw", "translation_dd", "translation_dw", "ref_value"] {
        assert!(!json.contains(cache), "{cache} must not be written: {json}");
    }
    assert!(json.contains("translation") && json.contains("rotation"));
}

// --------------------------------------------------------- UnitVecParam

#[test]
fn unitvec_param_round_trips_with_its_caches_rebuilt() {
    let dir = vect3d::new(0.3, -0.5, 0.81).unit();
    let a = UnitVecParam::new(dir);
    let b: UnitVecParam = rt(&a);

    assert!((b.unit - a.unit).norm() < 1e-15, "direction must survive");
    assert!((b.unit.norm() - 1.0).abs() < 1e-12, "and stay a unit vector");

    // The chart matrix and the Jacobian cache are derived; both must come
    // back matching what a fresh construction produces.
    for r in 0..3 {
        for c in 0..3 {
            assert!((b.rot[r][c] - a.rot[r][c]).abs() < 1e-15,
                "rot[{r}][{c}] was not rebuilt");
        }
    }
    // The chart must not be left at the identity -- that is what a
    // default-constructed cache would give.
    let ident = matrix3d::identity();
    let differs = (0..3).any(|r| (0..3).any(|c| (b.rot[r][c] - ident[r][c]).abs() > 1e-9));
    assert!(differs, "the chart was not rebuilt from the direction");
    // unit_d is solve-time state: it is filled only once the delta has an
    // index, so it is zero here and matching zero proves nothing. The solve
    // round trip below is what covers it.
    for k in 0..2 {
        assert!((b.unit_d[k] - a.unit_d[k]).norm() < 1e-15);
    }
}

// ------------------------------------------------------ through a solve
// The caches that only exist during a solve are covered by solving: a
// reloaded model has to reach the same answer as the one it came from.

#[arael::model]
#[arael(constraint(hb, {
    let fwd = pv.r2w.rotation_matrix * vect3sym::from_components(1.0, 0.0, 0.0);
    let up = pv.r2w.rotation_matrix * vect3sym::from_components(0.0, 0.0, 1.0);
    let d = pv.r2w.translation - pv.measured_translation;
    [d.x, d.y, d.z,
     (fwd.x - pv.measured_forward.x) * 2.0,
     (fwd.y - pv.measured_forward.y) * 2.0,
     (fwd.z - pv.measured_forward.z) * 2.0,
     (up.x - pv.measured_up.x) * 2.0,
     (up.y - pv.measured_up.y) * 2.0,
     (up.z - pv.measured_up.z) * 2.0,
     (pv.dir.unit.x - pv.measured_dir.x) * 3.0,
     (pv.dir.unit.y - pv.measured_dir.y) * 3.0,
     (pv.dir.unit.z - pv.measured_dir.z) * 3.0]
}))]
#[arael(root)]
#[derive(serde::Serialize, serde::Deserialize)]
struct Pv {
    r2w: TransformParam,
    dir: UnitVecParam,
    measured_translation: vect3d,
    measured_forward: vect3d,
    measured_up: vect3d,
    measured_dir: vect3d,
    #[serde(skip)]
    hb: SelfBlock<Pv>,
}

#[test]
fn a_model_of_these_params_solves_the_same_after_a_round_trip() {
    use arael::model::Model;
    use arael::simple_lm::{LmConfig, LmProblem};
    const H: f64 = std::f64::consts::FRAC_1_SQRT_2;

    let build = || Pv {
        r2w: TransformParam::new(vect3d::new(0.0, 0.0, 0.0), quaternd::identity()),
        dir: UnitVecParam::new(vect3d::new(1.0, 0.0, 0.0)),
        measured_translation: vect3d::new(1.0, 2.0, 3.0),
        measured_forward: vect3d::new(H, H, 0.0),
        measured_up: vect3d::new(0.0, 0.0, 1.0),
        measured_dir: vect3d::new(H, 0.0, H),
        hb: SelfBlock::new(),
    };

    let mut a = build();
    let mut b: Pv = rt(&build());
    let ra = a.solve_sparse(&LmConfig::default());
    let rb = b.solve_sparse(&LmConfig::default());

    assert!(ra.iterations > 1, "the model must actually solve");
    assert_eq!(ra.iterations, rb.iterations);
    assert!((ra.end_cost - rb.end_cost).abs() < 1e-14,
        "end {} vs {}", ra.end_cost, rb.end_cost);
    assert!(ra.end_cost < 1e-12, "and reach the answer");
    // The recovered transform and direction agree.
    assert!((b.r2w.translation - a.r2w.translation).norm() < 1e-12);
    assert!((b.dir.unit - a.dir.unit).norm() < 1e-12);
}

#[test]
fn unitvec_param_keeps_its_optimize_flag() {
    for opt in [true, false] {
        let mut a = UnitVecParam::new(vect3d::new(0.0, 1.0, 0.0));
        a.d.optimize = opt;
        let b: UnitVecParam = rt(&a);
        assert_eq!(b.d.optimize, opt);
    }
}

#[test]
fn unitvec_param_writes_no_derived_state() {
    let a = UnitVecParam::new(vect3d::new(0.3, -0.5, 0.81));
    let json = serde_json::to_string(&a).unwrap();
    for cache in ["ref_q", "rot", "unit_d"] {
        assert!(!json.contains(cache), "{cache} must not be written: {json}");
    }
    assert!(json.contains("unit"));
}

// --------------------------------------------------------------- matrix

#[test]
fn matrices_round_trip() {
    let m = quaternd::from_euler_angles(vect3d::new(0.1, 0.2, 0.3)).rotation_matrix();
    let b: matrix3d = rt(&m);
    for r in 0..3 {
        for c in 0..3 {
            assert!((b[r][c] - m[r][c]).abs() < 1e-15);
        }
    }
    let m2 = arael::matrix::matrix2d::from_rows(vect2d::new(1.0, 2.0), vect2d::new(3.0, 4.0));
    let b2: arael::matrix::matrix2d = rt(&m2);
    assert!((b2[0][0] - 1.0).abs() < 1e-15 && (b2[1][1] - 4.0).abs() < 1e-15);
}

