// What a solve logs about its GPU: a warning when one is given and the
// route does not run on it, and under `verbose` the device and its share of
// the work. One test in a binary of its own: the log's sink is the
// process's, and the test asserts what is absent too.
#![cfg(feature = "gpu-cpu")]

use std::sync::{Arc, Mutex};

use arael::log::{self, Level};
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, BlockSupernodalMode, EnvelopeMode, GpuDevice, LmConfig, RootProblem, SchurPolicy, SolveOrdering, SparseFaer, SparseFaerOptions,
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
fn a_solve_logs_what_its_gpu_does() {
    let seen: Arc<Mutex<Vec<(Level, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    log::set_sink(move |level, msg| sink.lock().unwrap().push((level, msg.to_string())));
    // a solve's log
    let solve = |opts: SparseFaerOptions, verbose: bool| {
        seen.lock().unwrap().clear();
        let mut w = build();
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let r = lm_solve(&params, &mut SparseFaer::from_options(&opts), &mut w, &LmConfig { max_iters: 20, verbose, ..Default::default() }).unwrap();
        (seen.lock().unwrap().clone(), r)
    };
    let reduced = || SparseFaerOptions::default().with_policy(SchurPolicy::Force);
    let supernodal = || reduced().with_envelope_schur(EnvelopeMode::Never).with_block_supernodal(BlockSupernodalMode::Always);
    let unused = |log: &[(Level, String)]| -> Vec<String> {
        log.iter().filter(|(l, m)| *l == Level::Warn && m.contains("was given and is not used")).map(|(_, m)| m.clone()).collect()
    };
    let about_gpu = |log: &[(Level, String)]| log.iter().filter(|(_, m)| m.starts_with("gpu: cpu:0 ")).count();

    // no GPU: nothing about one
    let (log, _) = solve(supernodal(), true);
    assert_eq!(about_gpu(&log), 0, "{log:?}");

    // on the device: no warning, and quiet unless verbose
    let (log, r) = solve(supernodal().with_gpu(Some(GpuDevice::cpu())), false);
    assert_ne!(r.gpu.unwrap().work, arael::simple_lm::GpuWork::Unused);
    assert_eq!(about_gpu(&log), 0, "{log:?}");
    let (log, _) = solve(supernodal().with_gpu(Some(GpuDevice::cpu())), true);
    assert!(unused(&log).is_empty(), "{log:?}");
    let line = log.iter().find(|(l, m)| *l == Level::Info && m.starts_with("gpu: cpu:0 ")).unwrap_or_else(|| panic!("the device's line in {log:?}"));
    assert!(line.1.contains(", wide f64 -- ") && line.1.contains("on the device"), "{}", line.1);

    // given and not used: one warning a solve, naming the route that ran
    let cases = [
        (supernodal().with_iterative_schur(Default::default()), "the reduced system is solved by conjugate gradients"),
        (reduced().with_envelope_schur(EnvelopeMode::Always).with_ordering(SolveOrdering::Natural), "the reduced system is factored under its envelope"),
        (reduced().with_envelope_schur(EnvelopeMode::Never).with_block_supernodal(BlockSupernodalMode::Never), "the reduced system is factored by the scalar sparse Cholesky"),
        (SparseFaerOptions::default().with_policy(SchurPolicy::Never).with_block_supernodal(BlockSupernodalMode::Never), "the whole system is factored by the scalar sparse Cholesky"),
    ];
    for (opts, route) in cases {
        let (log, r) = solve(opts.with_gpu(Some(GpuDevice::cpu())), false);
        assert_eq!(r.gpu.unwrap().work, arael::simple_lm::GpuWork::Unused, "{route}");
        let warnings = unused(&log);
        assert_eq!(warnings.len(), 1, "{route}: {log:?}");
        assert!(warnings[0].starts_with("gpu: cpu:0 was given and is not used -- ") && warnings[0].contains(route), "{route}: {}", warnings[0]);
    }
    log::reset_sink();
}
