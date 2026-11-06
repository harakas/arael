// ARAEL_DEVICE_CHECK: the first solve of each root type checks its device
// assembly against the CPU's and logs it (ARAEL_DEVICE_CHECK_LOG), a root
// without one logs why. The variables are read once per process: this file
// is a process of its own, and sets them before the first solve.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmConfig, LmProblem};

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
    // a check a panic cuts short (the select's index outside its arms) runs
    // again at the next solve
    let bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sps(5).solve_dense(&cfg)));
    assert!(bad.is_err(), "the select panics on the CPU");
    sps(1).solve_dense(&cfg).unwrap();
    let text = std::fs::read_to_string(&log).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3, "{text}");
    assert!(lines[0].contains("Pts (f64): checked: cost"), "{text}");
    assert!(lines[1].contains("Qts: no device assembly:") && lines[1].contains("opaque_check"), "{text}");
    assert!(lines[2].contains("Sps (f64): checked: cost"), "{text}");
}
