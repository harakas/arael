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
