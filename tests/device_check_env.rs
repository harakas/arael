// ARAEL_DEVICE_CHECK: the first solve of each root type checks its device
// assembly against the CPU's and logs it (ARAEL_DEVICE_CHECK_LOG), a root
// without one logs why, a root whose CPU assembly panics on its data logs
// that and is checked at a later solve; an f64 root on a device without f64
// (Metal) logs that it assembles on the CPU there. The variables are read
// once per process: this file is a process of its own, and sets them before
// the first solve.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal))]

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{GpuDevice, LmConfig, LmProblem, LmProblemInternals, LmSolver, RootProblem, SparseFaer};
use arael::utils::Float;

/// A point pulled toward two targets.
#[arael::model]
#[arael(constraint(hb, { [(pt.x - pt.a) * 2.0, (pt.x * pt.x - pt.a) * 0.5] }))]
struct Pt<T: Float> {
    x: Param<T>,
    a: T,
    hb: SelfBlock<Pt<T>, T>,
}

#[arael::model]
#[arael(root)]
struct Pts {
    pts: refs::Vec<Pt<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct PtsF {
    pts: refs::Vec<Pt<f32>>,
}

/// A function whose value is Rust code: no device assembly.
#[arael::function(opaque_check, derivs = [x * 0.0 + 1.5])]
fn opaque_check_eval(x: f64) -> f64 {
    x * 1.5
}

#[arael::model]
#[arael(constraint(hb, { [opaque_check(qt.x) - 1.0] }))]
struct Qt {
    x: Param<f64>,
    hb: SelfBlock<Qt>,
}

#[arael::model]
#[arael(root)]
struct Qts {
    qts: refs::Vec<Qt>,
}

/// A point whose select panics on the CPU for an index outside its arms.
#[arael::model]
#[arael(constraint(hb, { [select(sp.k, sp.x - 1.0, sp.x * 2.0 - 1.0)] }))]
struct Sp<T: Float> {
    x: Param<T>,
    k: u32,
    hb: SelfBlock<Sp<T>, T>,
}

#[arael::model]
#[arael(root)]
struct Sps {
    sps: refs::Vec<Sp<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct SpsF {
    sps: refs::Vec<Sp<f32>>,
}

fn sp_list<T: Float>(k: u32) -> refs::Vec<Sp<T>> {
    let mut v = refs::Vec::new();
    v.push(Sp { x: Param::new(T::from(0.2).unwrap()), k, hb: SelfBlock::new() });
    v
}

fn pt_list<T: Float>() -> refs::Vec<Pt<T>> {
    let mut v = refs::Vec::new();
    for k in 0..4 {
        v.push(Pt { x: Param::new(T::from(0.5 + 0.1 * k as f64).unwrap()), a: T::from(1.0 + 0.2 * k as f64).unwrap(), hb: SelfBlock::new() });
    }
    v
}

/// The device the GPU solves and the check run on: CUDA when it is compiled
/// in, else HIP, else Metal, else the CPU device.
#[cfg(feature = "gpu-cuda")]
fn gpu() -> GpuDevice {
    GpuDevice::cuda(0)
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn gpu() -> GpuDevice {
    GpuDevice::hip(0)
}
#[cfg(all(gpu_metal, not(any(feature = "gpu-cuda", feature = "gpu-hip"))))]
fn gpu() -> GpuDevice {
    GpuDevice::metal(0)
}
#[cfg(all(feature = "gpu-cpu", not(any(feature = "gpu-cuda", feature = "gpu-hip", gpu_metal))))]
fn gpu() -> GpuDevice {
    GpuDevice::cpu()
}

/// `s` solved on the device, the whole system.
fn solve_on_device<T: Float, P: LmProblemInternals<T> + RootProblem<T>>(mut s: P) -> Result<arael::simple_lm::LmResult<T>, arael::simple_lm::SolveFailure<T>>
where
    SparseFaer<T>: LmSolver<T>,
{
    let mut x = Vec::new();
    s.serialize(&mut x);
    let opts = arael::simple_lm::SparseFaerOptions::default()
        .with_policy(arael::simple_lm::SchurPolicy::Never)
        .with_block_supernodal(arael::simple_lm::BlockSupernodalMode::Always)
        .with_gpu(Some(gpu()));
    let mut solver = SparseFaer::<T>::from_options(&opts);
    arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &LmConfig::<T>::default())
}

/// The message of the panic `solve` meets.
fn panic_of(solve: impl FnOnce()) -> String {
    let why = std::panic::catch_unwind(std::panic::AssertUnwindSafe(solve)).expect_err("the select panics on the CPU");
    why.downcast_ref::<String>().cloned().or_else(|| why.downcast_ref::<&str>().map(|m| m.to_string())).unwrap_or_default()
}

#[test]
fn a_checked_solve_logs_each_root_type_once() {
    let log = std::env::temp_dir().join(format!("arael-device-check-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    // SAFETY: this test is alone in its binary, so no other thread reads
    // the environment while it is written.
    unsafe {
        std::env::set_var("ARAEL_DEVICE_CHECK", "1");
        std::env::set_var("ARAEL_DEVICE_CHECK_LOG", &log);
    }
    let f64s = gpu().has_f64();
    let mut want: Vec<&str> = Vec::new();
    let cfg = LmConfig::<f64>::default();
    let cfg32 = LmConfig::<f32>::default();
    Pts { pts: pt_list() }.solve_dense(&cfg).unwrap();
    Pts { pts: pt_list() }.solve_dense(&cfg).unwrap();
    want.push(if f64s { "Pts (f64): checked: cost" } else { "Pts (f64): no device assembly here: the device has no f64" });
    PtsF { pts: pt_list() }.solve_dense(&cfg32).unwrap();
    PtsF { pts: pt_list() }.solve_dense(&cfg32).unwrap();
    want.push("PtsF (f32): checked: cost");
    let mut q = Qts { qts: refs::Vec::new() };
    q.qts.push(Qt { x: Param::new(0.3), hb: SelfBlock::new() });
    q.solve_dense(&cfg).unwrap();
    want.push("Qts: no device assembly:");
    // a root whose CPU assembly panics on its data (the select's index
    // outside its arms) is not checked: the CPU solve meets the model's
    // own panic, the device solve fails as it does unchecked, and a later
    // solve checks the root
    let why = panic_of(|| drop(Sps { sps: sp_list(5) }.solve_dense(&cfg)));
    assert!(why.contains("select index 5 out of range"), "the model's own panic: {why}");
    if f64s {
        want.extend(["Sps (f64): not checked here: the CPU assembly panics: select index 5 out of range"; 2]);
        assert!(solve_on_device(Sps { sps: sp_list(5) }).is_err(), "the device solve fails without a panic");
        Sps { sps: sp_list(1) }.solve_dense(&cfg).unwrap();
        want.push("Sps (f64): checked: cost");
    } else {
        want.push("Sps (f64): no device assembly here: the device has no f64");
        Sps { sps: sp_list(1) }.solve_dense(&cfg).unwrap();
    }
    let why = panic_of(|| drop(SpsF { sps: sp_list(5) }.solve_dense(&cfg32)));
    assert!(why.contains("select index 5 out of range"), "the model's own panic: {why}");
    want.extend(["SpsF (f32): not checked here: the CPU assembly panics: select index 5 out of range"; 2]);
    assert!(solve_on_device(SpsF { sps: sp_list(5) }).is_err(), "the f32 device solve fails without a panic");
    SpsF { sps: sp_list(1) }.solve_dense(&cfg32).unwrap();
    want.push("SpsF (f32): checked: cost");
    let text = std::fs::read_to_string(&log).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), want.len(), "{text}");
    for (line, want) in lines.iter().zip(&want) {
        assert!(line.contains(want), "`{want}` in\n{text}");
    }
    assert!(lines[2].contains("opaque_check"), "{text}");
}
