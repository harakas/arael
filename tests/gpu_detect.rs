// GPU detection (`GpuDevice::available`, `GpuDevice::detect`): the
// machine's GPUs of the runtimes compiled in, the preferred first, and a
// solve on the one detected. Runs in every build, also without a GPU
// feature, where nothing is detected. One test in a binary of its own: the
// log's sink is the process's.

use std::sync::{Arc, Mutex};

use arael::log::{self, Level};
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, BlockSupernodalMode, EnvelopeMode, GpuBackend, GpuDevice, GpuInfo, GpuWork, LmConfig, MixedPrecision, RootProblem, SchurPolicy,
    SparseFaer, SparseFaerOptions,
};

#[arael::model]
#[arael(constraint(hb, { [(pose.x - pose.ax) * 0.1, (pose.y - pose.ay) * 0.1] }))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct Landmark {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Landmark>,
}

#[arael::model]
#[arael(constraint(hb, { [l.x - p.x - obs.dx, l.y - p.y - obs.dy] }))]
struct Obs {
    #[arael(ref = root.poses)]
    p: Ref<Pose>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Landmark>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    obs: std::vec::Vec<Obs>,
}

/// Six poses near their anchors, each seeing eight landmarks.
fn build() -> World {
    let mut w = World { poses: refs::Vec::new(), landmarks: refs::Vec::new(), obs: std::vec::Vec::new() };
    let pose = |i: usize| (i as f64, 0.5 * i as f64);
    let landmark = |j: usize| (j as f64, 2.0 + j as f64);
    for i in 0..6 {
        let (x, y) = pose(i);
        w.poses.push(Pose { x: Param::new(x + 0.05), y: Param::new(y - 0.05), ax: x, ay: y, hb: SelfBlock::new() });
    }
    for j in 0..8 {
        let (x, y) = landmark(j);
        w.landmarks.push(Landmark { x: Param::new(x - 0.05), y: Param::new(y + 0.05), hb: SelfBlock::new() });
    }
    for i in 0..6 {
        for j in 0..8 {
            let ((px, py), (lx, ly)) = (pose(i), landmark(j));
            w.obs.push(Obs { p: w.poses.ref_at(i), l: w.landmarks.ref_at(j), dx: lx - px, dy: ly - py, hb: CrossBlock::new() });
        }
    }
    w
}

#[test]
fn detection_lists_the_machines_gpus_and_picks_the_preferred_one() {
    let seen: Arc<Mutex<Vec<(Level, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    log::set_sink(move |level, msg| sink.lock().unwrap().push((level, msg.to_string())));

    // the listing: preferred first, no CPU device, and nothing logged
    let listed = GpuDevice::available();
    assert!(seen.lock().unwrap().is_empty(), "a listing logs nothing");
    let rank = |g: &GpuInfo| {
        let runtime = match g.backend {
            GpuBackend::Cuda => 0,
            GpuBackend::Hip => 1,
            GpuBackend::Metal => 2,
            GpuBackend::Cpu => panic!("the CPU device is not a GPU: {g}"),
        };
        (g.integrated, runtime, g.index)
    };
    assert!(listed.windows(2).all(|w| rank(&w[0]) < rank(&w[1])), "{listed:?}");
    assert_eq!(GpuDevice::available(), listed, "a second listing");
    // a runtime compiled in lists its first device, the one the tests run on
    #[cfg(feature = "gpu-cuda")]
    assert!(listed.iter().any(|g| g.backend == GpuBackend::Cuda && g.index == 0), "{listed:?}");
    #[cfg(feature = "gpu-hip")]
    assert!(listed.iter().any(|g| g.backend == GpuBackend::Hip && g.index == 0), "{listed:?}");
    #[cfg(gpu_metal)]
    assert!(listed.iter().any(|g| g.backend == GpuBackend::Metal && g.index == 0), "{listed:?}");
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-hip", gpu_metal)))]
    assert!(listed.is_empty(), "no GPU runtime is compiled in: {listed:?}");

    // the detection: the first of the listing, and one line saying so
    let gpu = GpuDevice::detect();
    let lines: Vec<String> = seen.lock().unwrap().iter().filter(|(l, _)| *l == Level::Info).map(|(_, m)| m.clone()).collect();
    assert_eq!(lines.len(), 1, "{lines:?}");
    let (Some(gpu), Some(best)) = (gpu.clone(), listed.first()) else {
        assert!(gpu.is_none() && listed.is_empty(), "{gpu:?} of {listed:?}");
        assert!(lines[0].starts_with("gpu: none detected -- "), "{}", lines[0]);
        log::reset_sink();
        return;
    };
    assert_eq!((gpu.backend(), gpu.index()), (best.backend, best.index));
    assert!(lines[0].starts_with(&format!("gpu: detected {best}")), "{}", lines[0]);
    for other in &listed[1..] {
        assert!(lines[0].contains(&other.to_string()), "{} names {other}", lines[0]);
    }

    // a solve on the detected GPU runs there, on the device the listing named
    let mixed = (!gpu.has_f64()).then(MixedPrecision::default);
    let opts = SparseFaerOptions::default()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_mixed_precision(mixed)
        .with_gpu(Some(gpu));
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let r = lm_solve(&params, &mut SparseFaer::from_options(&opts), &mut w, &LmConfig { max_iters: 20, ..Default::default() }).unwrap();
    log::reset_sink();
    assert!(r.end_cost < 1e-12, "{}", r.end_cost);
    let report = r.gpu.expect("the solve's GPU report");
    assert_ne!(report.work, GpuWork::Unused);
    assert_eq!((report.backend, report.index, report.name.as_str(), report.integrated), (best.backend, best.index, best.name.as_str(), best.integrated));
}
