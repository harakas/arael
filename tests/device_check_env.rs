// ARAEL_DEVICE_CHECK: the first solve of each root type checks its device
// assembly against the CPU's and logs it (ARAEL_DEVICE_CHECK_LOG), a root
// without one logs why, a root whose CPU assembly panics on its data logs
// that and is checked at a later solve. The variables are read once per
// process: this file is a process of its own, and sets them before the
// first solve.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu"))]

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{GpuDevice, LmConfig, LmProblem, RootProblem};

/// A point pulled toward two targets.
#[arael::model]
#[arael(constraint(hb, { [(pt.x - pt.a) * 2.0, (pt.x * pt.x - pt.a) * 0.5] }))]
struct Pt {
    x: Param<f64>,
    a: f64,
    hb: SelfBlock<Pt>,
}

#[arael::model]
#[arael(root)]
struct Pts {
    pts: refs::Vec<Pt>,
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
struct Sp {
    x: Param<f64>,
    k: u32,
    hb: SelfBlock<Sp>,
}

#[arael::model]
#[arael(root)]
struct Sps {
    sps: refs::Vec<Sp>,
}

fn sps(k: u32) -> Sps {
    let mut v = refs::Vec::new();
    v.push(Sp { x: Param::new(0.2), k, hb: SelfBlock::new() });
    Sps { sps: v }
}

/// The device the GPU solves run on: CUDA when it is compiled in, else HIP,
/// else the CPU device.
#[cfg(feature = "gpu-cuda")]
fn gpu() -> GpuDevice {
    GpuDevice::cuda(0)
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn gpu() -> GpuDevice {
    GpuDevice::hip(0)
}
#[cfg(all(feature = "gpu-cpu", not(any(feature = "gpu-cuda", feature = "gpu-hip"))))]
fn gpu() -> GpuDevice {
    GpuDevice::cpu()
}

/// `s` solved on the device, the whole system.
fn solve_on_device(mut s: Sps) -> Result<arael::simple_lm::LmResult<f64>, arael::simple_lm::SolveFailure<f64>> {
    let mut x = Vec::new();
    s.serialize(&mut x);
    let opts = arael::simple_lm::SparseFaerOptions::default()
        .with_policy(arael::simple_lm::SchurPolicy::Never)
        .with_block_supernodal(arael::simple_lm::BlockSupernodalMode::Always)
        .with_gpu(Some(gpu()));
    let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&opts);
    arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &LmConfig::<f64>::default())
}

fn pts() -> Pts {
    let mut v = refs::Vec::new();
    for k in 0..4 {
        v.push(Pt { x: Param::new(0.5 + 0.1 * k as f64), a: 1.0 + 0.2 * k as f64, hb: SelfBlock::new() });
    }
    Pts { pts: v }
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
    let cfg = LmConfig::<f64>::default();
    pts().solve_dense(&cfg).unwrap();
    pts().solve_dense(&cfg).unwrap();
    let mut q = Qts { qts: refs::Vec::new() };
    q.qts.push(Qt { x: Param::new(0.3), hb: SelfBlock::new() });
    q.solve_dense(&cfg).unwrap();
    // a root whose CPU assembly panics on its data (the select's index
    // outside its arms) is not checked: the CPU solve meets the model's
    // own panic, the device solve fails as it does unchecked, and a later
    // solve checks the root
    let bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sps(5).solve_dense(&cfg)));
    let why = bad.expect_err("the select panics on the CPU");
    let why = why.downcast_ref::<String>().map(String::as_str).or_else(|| why.downcast_ref::<&str>().copied()).unwrap_or_default();
    assert!(why.contains("select index 5 out of range"), "the model's own panic: {why}");
    assert!(solve_on_device(sps(5)).is_err(), "the device solve fails without a panic");
    sps(1).solve_dense(&cfg).unwrap();
    let text = std::fs::read_to_string(&log).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5, "{text}");
    assert!(lines[0].contains("Pts (f64): checked: cost"), "{text}");
    assert!(lines[1].contains("Qts: no device assembly:") && lines[1].contains("opaque_check"), "{text}");
    for line in &lines[2..4] {
        assert!(line.contains("Sps (f64): not checked here: the CPU assembly panics: select index 5 out of range"), "{text}");
    }
    assert!(lines[4].contains("Sps (f64): checked: cost"), "{text}");
}
