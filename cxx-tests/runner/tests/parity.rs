// End-to-end parity: build the fixture problem through the GENERATED
// C++ interface (compiled and linked against the capi staticlib), and
// through the model crate directly in Rust; the two paths run the same
// deterministic solver, so every printed value must round-trip
// EXACTLY. Skipped with a note when no C++ compiler is available.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem, LmStatus, SolveFailureKind};
use arael::vect::{vect2d, vect3d};
use cxx_fit::{Fit, GpsObs, N, Obs, Pose, Tie};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fill(fit: &mut Fit) {
    for i in 0..6 {
        let x = i as f64;
        fit.obs.push(Obs { x, y: 2.0 * x + 1.0 + if i % 2 == 0 { 0.05 } else { -0.05 } });
    }
    let t = [1.5, -0.3, 0.7];
    let w = [1.0, 2.0, 0.5];
    for i in 0..3 {
        fit.items.push(N {
            v: Param::default(),
            t: t[i],
            w: w[i],
            hb: SelfBlock::new(),
        });
    }
}

/// The shim's status mapping (kept in lockstep with emit_ffi).
fn code(s: &LmStatus) -> f64 {
    (match s {
        LmStatus::Converged => 0,
        LmStatus::CostThreshold => 1,
        LmStatus::MaxIterations => 2,
        LmStatus::GradientTolerance => 3,
        LmStatus::ParameterTolerance => 4,
        LmStatus::PredictedReduction => 5,
        LmStatus::LambdaCeiling => 6,
        LmStatus::DriverTerminated => 7,
        LmStatus::ObserverTerminated => 8,
        LmStatus::TimeLimit => 9,
        LmStatus::RetryBudgetExhausted => 10,
        LmStatus::Aborted => 11,
    }) as f64
}

fn find_compiler() -> Option<&'static str> {
    for cc in ["c++", "g++", "clang++"] {
        if Command::new(cc).arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().map(|s| s.success()).unwrap_or(false)
        {
            return Some(cc);
        }
    }
    None
}

#[test]
fn cxx_interface_matches_rust_exactly() {
    let Some(cc) = find_compiler() else {
        eprintln!("cxx parity: no C++ compiler found, skipping");
        return;
    };
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    // The staticlib the C++ program links.
    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-fit-capi", "--release"])
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");

    let bin = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_parity");
    let status = Command::new(cc)
        .arg("-std=c++17").arg("-O2")
        .arg("-I").arg(ws.join("model/cxx/include"))
        .arg(ws.join("runner/tests/parity_main.cpp"))
        .arg(ws.join("target/release/libcxx_fit_capi.a"))
        .arg("-lpthread").arg("-ldl").arg("-lm")
        .arg("-o").arg(&bin)
        .status().expect("compiler spawn");
    assert!(status.success(), "C++ compile/link failed");

    let out = Command::new(&bin).output().expect("run");
    assert!(out.status.success(), "C++ run failed: {}",
        String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let mut got: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if let (Some(n), Some(v)) = (it.next(), it.next()) {
            got.insert(n.to_string(), v.parse().unwrap());
        }
    }
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("C++ output missing `{n}`"));

    // The same problem in Rust; every C++ value must match it exactly
    // (deterministic same code path; %.17e round-trips f64 exactly).
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut fit = Fit::default();
    fill(&mut fit);
    assert!(fit.validate().is_clean());
    assert_eq!(g("clean"), 1.0);
    assert_eq!(g("n_obs"), 6.0);
    assert_eq!(g("n_items"), 3.0);
    assert_eq!(g("obs3_y"), fit.obs[3].y);
    assert_eq!(g("item1_t"), fit.items[1].t);

    let r = fit.solve_dense(&cfg).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert_eq!(g("dense_status"), code(&r.status));
    assert_eq!(g("dense_start"), r.start_cost);
    assert_eq!(g("dense_end"), r.end_cost);
    assert_eq!(g("dense_iters"), r.iterations as f64);
    assert_eq!(g("dense_m"), fit.m.value);
    assert_eq!(g("dense_c"), fit.c.value);
    for i in 0..3 {
        assert_eq!(g(&format!("dense_v{i}")), fit.items[i].v.value, "v{i}");
    }
    // And the solution is the right one (least squares of the +-0.05
    // pattern keeps m at 2, c near 1).
    assert!((fit.m.value - 2.0).abs() < 0.05, "m {}", fit.m.value);
    assert!((fit.c.value - 1.0).abs() < 0.1, "c {}", fit.c.value);

    // Covariance parity: same assembly, same marginal, exact.
    {
        use arael::covariance::{CovMode, Covariance};
        let cov = fit.assemble_covariance(CovMode::AllMarginals).unwrap();
        let m = cov.marginal_cov(&fit.items[0]).unwrap();
        assert_eq!(g("cov_ok"), 1.0);
        assert_eq!(g("cov_item0_ok"), 1.0);
        assert_eq!(g("cov_item0"), m[(0, 0)]);
    }

    let mut fit2 = Fit::default();
    fill(&mut fit2);
    let r2 = fit2.solve_sparse(&cfg).unwrap();
    assert_eq!(g("sparse_status"), code(&r2.status));
    assert_eq!(g("sparse_end"), r2.end_cost);
    assert_eq!(g("sparse_m"), fit2.m.value);
    assert_eq!(g("sparse_c"), fit2.c.value);

    // Stage 3 surface, mirrored: deque chain, ties through refs, arena
    // with a removal, Option entity, math data, fixed euler param.
    let mut f3 = Fit::default();
    fill(&mut f3);
    f3.cal = vect2d::new(0.25, -0.5);
    let targets = [
        vect3d::new(0.0, 0.0, 0.0),
        vect3d::new(1.0, 0.5, 0.0),
        vect3d::new(2.0, 1.0, 0.0),
    ];
    f3.poses.push_back(Pose::default());
    f3.poses.push_back(Pose::default());
    f3.poses.push_front(Pose::default());
    for i in 0..3 {
        let p = &mut f3.poses[i];
        p.target = targets[i];
        p.pos.value = vect3d::new(0.1 * i as f64, -0.1 * i as f64, 0.05);
        p.ea.value = vect3d::new(0.1, 0.2, 0.3 * i as f64);
        p.ea.optimize = false;
    }
    f3.poses[0].info.gps = Some(GpsObs {
        pos: vect3d::new(7.0, 8.0, 9.0),
        isigma: 2.5,
    });
    f3.ties.push(Tie {
        a: f3.poses.ref_at(0), b: f3.poses.ref_at(1),
        d: vect3d::new(1.0, 0.4, 0.0), w: 3.0, hb: CrossBlock::new(),
    });
    f3.ties.push(Tie {
        a: f3.poses.ref_at(1), b: f3.poses.ref_at(2),
        d: vect3d::new(1.0, 0.6, 0.0), w: 3.0, hb: CrossBlock::new(),
    });
    let m0 = f3.marks.push(N { v: Param::default(), t: 0.4, w: 1.0, hb: SelfBlock::new() });
    let m1 = f3.marks.push(N { v: Param::default(), t: 9.0, w: 1.0, hb: SelfBlock::new() });
    let m2 = f3.marks.push(N { v: Param::default(), t: -0.6, w: 2.0, hb: SelfBlock::new() });
    f3.marks.remove(m1).unwrap();

    assert!(f3.validate().is_clean());
    assert_eq!(g("s3_clean"), 1.0);
    let r3 = f3.solve_dense(&cfg).unwrap();
    assert!(r3.status.is_success(), "{:?}", r3.status);
    assert_eq!(g("s3_status"), code(&r3.status));
    assert_eq!(g("s3_end"), r3.end_cost);
    assert_eq!(g("s3_cal_x"), f3.cal.x);
    assert_eq!(g("s3_cal_y"), f3.cal.y);
    for i in 0..3 {
        let q = f3.poses[i].pos.value;
        assert_eq!(g(&format!("s3_p{i}_x")), q.x, "p{i}.x");
        assert_eq!(g(&format!("s3_p{i}_y")), q.y, "p{i}.y");
        assert_eq!(g(&format!("s3_p{i}_z")), q.z, "p{i}.z");
    }
    // The fixed rotation neither moved nor was dropped.
    assert_eq!(g("s3_ea0_z"), f3.poses[0].ea.value.z);
    assert_eq!(g("s3_has_gps0"), 1.0);
    assert_eq!(g("s3_has_gps1"), 0.0);
    assert_eq!(g("s3_gps0_y"), 8.0);
    assert_eq!(g("s3_gps0_isigma"), 2.5);
    assert_eq!(g("s3_marks_len"), f3.marks.len() as f64);
    assert_eq!(g("s3_mark0_v"), f3.marks[m0].v.value);
    assert_eq!(g("s3_mark2_v"), f3.marks[m2].v.value);
    // Marks solved to their targets; the removed slot is gone.
    assert!((f3.marks[m0].v.value - 0.4).abs() < 1e-9);
    assert_eq!(f3.marks.len(), 2);

    // The degenerate model (unconstrained root params, nonzero cost)
    // fails with DegenerateDiagonal in Rust and status -1 + text in C++.
    let mut bad = Fit::default();
    bad.items.push(N { v: Param::default(), t: 1.0, w: 1.0, hb: SelfBlock::new() });
    let e = bad.solve_dense(&cfg).expect_err("degenerate must fail");
    assert!(matches!(e.kind, SolveFailureKind::DegenerateDiagonal { .. }),
        "{:?}", e.kind);
    assert_eq!(g("bad_status"), -1.0);
    assert_eq!(g("bad_has_error"), 1.0);
}

#[test]
fn generated_tree_is_current() {
    // `cargo arael check` as a test: the committed generated files must
    // match what the tool generates from the model today.
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let status = Command::new("cargo")
        .args(["run", "-q", "-p", "cargo-arael", "--manifest-path"])
        .arg(ws.join("../Cargo.toml"))
        .args(["--", "check", "--manifest-dir"])
        .arg(ws.join("model"))
        .status().expect("cargo spawn");
    assert!(status.success(), "generated tree is stale -- rerun `cargo arael export`");
}
