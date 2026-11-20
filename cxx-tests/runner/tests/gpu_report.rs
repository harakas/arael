// What the GPU report consumers print (gpu_main.py, cmake_consumer/gpu.cpp):
// the fixture solved on the CPU device with the device assembly on and off,
// and without a GPU. Shared by python.rs and cmake.rs.

use std::collections::HashMap;

/// Check the `name value` lines of a GPU report consumer against arael's
/// `GpuReport` of the same solves.
pub fn verify(text: &str) {
    let got: HashMap<&str, f64> = text
        .lines()
        .filter_map(|l| l.split_once(' ').map(|(n, v)| (n, v.trim().parse().unwrap())))
        .collect();
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("no `{n}` in\n{text}"));
    assert_eq!(g("none"), 1.0, "no report without a GPU");
    for tag in ["on", "off"] {
        let v = |n: &str| g(&format!("{tag}_{n}"));
        // GpuBackend::Cpu, device 0, named, f64 wide values, an f64 solve
        // by an f64 factor (GpuSolve::Factor)
        assert_eq!((v("backend"), v("index"), v("named"), v("wide")), (3.0, 0.0, 1.0, 0.0), "{tag}:\n{text}");
        assert_eq!((v("solve"), v("f32_system")), (1.0, 0.0), "{tag}:\n{text}");
        assert_eq!((v("reserved"), v("buffers")), (1.0, 1.0), "{tag}:\n{text}");
        // GpuWork::Solves: the fixture has no device form (its `Rig` reads
        // a cache computed from data), so the damped solves alone run on
        // the device
        assert_eq!(v("work"), 1.0, "{tag}:\n{text}");
    }
    // HostAssembly::NoDeviceForm, with its reason
    assert_eq!(g("on_host_assembly"), 2.0, "{text}");
    assert!(g("on_why") > 0.0, "the reason the fixture has no device form\n{text}");
    assert_eq!(g("on_report"), 1.0, "the text report has the gpu block");
    // HostAssembly::Asked, which comes before the model's form is asked
    assert_eq!((g("off_host_assembly"), g("off_why")), (1.0, 0.0), "{text}");
    // the timed solves' setup parts, within the analysis: the device's
    // plan among them on the device alone
    for (tag, on_device) in [("none", 0.0), ("on", 1.0)] {
        let v = |n: &str| g(&format!("{tag}_{n}"));
        assert_eq!((v("setup"), v("setup_within"), v("setup_gpu")), (1.0, 1.0, on_device), "{tag}:\n{text}");
    }
    // the device's own time: every attempt timed, no linearization (the
    // CPU assembles the fixture); none without a device
    assert_eq!(g("none_gpu_time"), 0.0, "{text}");
    assert_eq!((g("on_gpu_time"), g("on_attempts_timed"), g("on_assembly_timed")), (1.0, 1.0, 0.0), "{text}");
    // a factor has no iterations to time; the conjugate gradients' are a
    // part of their attempts
    assert_eq!((g("on_cg_timed"), g("cg_gpu_time"), g("cg_attempts_timed"), g("cg_cg_timed")), (0.0, 1.0, 1.0, 1.0), "{text}");
    // conjugate gradients on the reduced system: GpuSolve::ConjugateGradients,
    // the damped solves on the device (GpuWork::Solves), iterations counted
    assert_eq!((g("cg_solve"), g("cg_is_cg"), g("cg_work"), g("cg_iters")), (3.0, 1.0, 1.0, 1.0), "{text}");
}
