// Imported models on the device (feature gpu-cuda or gpu-hip): roots of an
// imported generic entity, a `vect<T, 4>` parameter and a `matrix<T, 2, 4>`
// datum, assemble on the device as on the CPU in f64 and f32.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]

use arael::__gpu::Wide;
use arael::matrix::matrixd;
use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmProblemInternals, RootProblem};
use arael::vect::vectd;
use export_consumer::{CalWorld32, CalWorld64};
use export_models::Cal;

#[cfg(feature = "gpu-cuda")]
fn device() -> std::sync::Arc<arael::__gpu::cuda::CudaDevice> {
    std::sync::Arc::new(arael::__gpu::cuda::CudaDevice::new(0).unwrap())
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn device() -> std::sync::Arc<arael::__gpu::hip::HipDevice> {
    std::sync::Arc::new(arael::__gpu::hip::HipDevice::new(0).unwrap())
}

macro_rules! cals {
    ($t:ty) => {{
        let mut cals = refs::Vec::new();
        for k in 0..5 {
            let s = 0.2 * k as f64;
            cals.push(Cal {
                v: Param::new(vectd::new([0.4 + s, -0.1, 0.9 - s, 0.0]).cast()),
                t: vectd::new([0.1, 0.2 + s, 0.5, -0.3]).cast(),
                h: matrixd::from_array([[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4 + s]]).cast(),
                wp: 0.7 as $t,
                w: 1.3 as $t,
                hb: SelfBlock::new(),
            });
        }
        cals
    }};
}

/// The parameters moved off the serialized point.
fn moved<T: arael::utils::Float>(x: &[T]) -> Vec<T> {
    x.iter().enumerate().map(|(i, &v)| v + T::from(1e-3 * ((i % 7) as f64 - 3.0)).unwrap()).collect()
}

#[test]
fn imported_models_assemble_on_the_device_as_on_the_cpu() {
    let mut w = CalWorld64 { cals: cals!(f64) };
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&w), None);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let x1 = moved(&x);
    for wide in [Wide::F64, Wide::Df32] {
        let chk = arael::__gpu::device_check(&mut w, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-11 * chk.cpu_cost, "f64 {wide:?}: {chk:?}");
        assert!(chk.grad_err < 1e-11 && chk.hess_err < 1e-11, "f64 {wide:?}: {chk:?}");
    }
    let mut w = CalWorld32 { cals: cals!(f32) };
    assert_eq!(LmProblemInternals::<f32>::device_unsupported(&w), None);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let x1 = moved(&x);
    for wide in [Wide::F64, Wide::Df32] {
        let chk = arael::__gpu::device_check(&mut w, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-5 * chk.cpu_cost, "f32 {wide:?}: {chk:?}");
        assert!(chk.grad_err < 1e-4 && chk.hess_err < 1e-4, "f32 {wide:?}: {chk:?}");
    }
}
