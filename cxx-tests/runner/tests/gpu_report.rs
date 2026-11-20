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
        assert_eq!((v("backend"), v("index"), v("named"), v("wide")), (3.0, 0.0, 1.0, 0.0), "{tag}:\n{text}");
        assert_eq!((v("f32_factor"), v("f32_system")), (0.0, 0.0), "{tag}:\n{text}");
        assert_eq!((v("reserved"), v("buffers")), (1.0, 1.0), "{tag}:\n{text}");
        // GpuWork::Solves: the fixture has no device form (its `Rig` reads
        // a cache computed from data), so the damped solves alone run on
        // the device
        assert_eq!(v("work"), 1.0, "{tag}:\n{text}");
    }
    // CpuAssembly::NoDeviceForm, with its reason
    assert_eq!(g("on_cpu_assembly"), 2.0, "{text}");
    assert!(g("on_why") > 0.0, "the reason the fixture has no device form\n{text}");
    assert_eq!(g("on_report"), 1.0, "the text report has the gpu block");
    // CpuAssembly::Asked, which comes before the model's form is asked
    assert_eq!((g("off_cpu_assembly"), g("off_why")), (1.0, 0.0), "{text}");
}
