// End-to-end parity: build the fixture problem through the GENERATED
// C++ interface (compiled and linked against the capi staticlib), and
// through the model crate directly in Rust; the two paths run the same
// deterministic solver, so every printed value must round-trip
// EXACTLY. Skipped with a note when no C++ compiler is available.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::Ref;
use arael::simple_lm::{LmConfig, LmProblem, LmStatus, SolveFailureKind};
use arael::vect::{vect2d, vect3d};
use cxx_fit::{Fit, GpsObs, N, Obs, Pose, Rig, Tie};
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
        .arg("-std=c++17").arg("-O2").arg("-ffp-contract=off")
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

    // Config layout parity: the C++ defaults are the Rust defaults.
    {
        let d = LmConfig::<f64>::default();
        assert_eq!(g("cfg_abs"), d.abs_precision);
        assert_eq!(g("cfg_rel"), d.rel_precision);
        assert_eq!(g("cfg_max_iters"), d.max_iters as f64);
        assert_eq!(g("cfg_min_iters"), d.min_iters as f64);
        assert_eq!(g("cfg_patience"), d.patience as f64);
        assert_eq!(g("cfg_threads"), d.num_threads as f64);
        assert_eq!(g("cfg_verbose"), d.verbose as u8 as f64);
        assert_eq!(g("cfg_lambda"), d.initial_lambda);
        assert_eq!(g("cfg_cost_threshold"), d.cost_threshold);
        assert_eq!(g("cfg_lambda_floor"), d.lambda_floor);
        assert_eq!(g("cfg_grad_has"), d.gradient_tolerance.is_some() as u8 as f64);
        assert_eq!(g("cfg_time_has"), d.time_limit.is_some() as u8 as f64);
        assert_eq!(g("cfg_wc_lambda"), LmConfig::<f64>::well_conditioned().initial_lambda);
    }

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

    // Observer + timing + report + conditional covariance mirrored.
    {
        use std::cell::Cell;
        use std::ops::ControlFlow;
        use std::rc::Rc;
        let mut f7 = Fit::default();
        fill(&mut f7);
        let calls = Rc::new(Cell::new(0u32));
        let plen = Rc::new(Cell::new(0u32));
        let (c2, p2) = (calls.clone(), plen.clone());
        let cfg7 = cfg.clone().with_gather_timing(true).with_observer(
            move |it: &arael::simple_lm::LmIter<'_, f64>| {
                c2.set(c2.get() + 1);
                p2.set(it.params.len() as u32);
                ControlFlow::Continue(())
            });
        let r7 = f7.solve_dense(&cfg7).unwrap();
        assert_eq!(g("report_empty_before"), 1.0);
        assert_eq!(g("obs_calls_eq_iters"),
            (calls.get() == r7.iterations as u32) as i32 as f64);
        assert_eq!(g("obs_params_len"), plen.get() as f64);
        assert_eq!(g("obs_end"), r7.end_cost);
        let t = r7.timing.as_ref().expect("timing gathered");
        assert_eq!(g("tm_has"), 1.0);
        assert_eq!(g("tm_total_pos"), 1.0);
        assert!(t.total.as_secs_f64() > 0.0);
        assert_eq!(g("tm_assembly_count"), t.assembly_count as f64);
        assert_eq!(g("tm_solve_count"), t.linear_solve_count as f64);
        assert_eq!(g("tm_cost_count"), t.cost_eval_count as f64);
        assert!(!r7.report().is_empty());
        assert_eq!(g("report_nonempty"), 1.0);
        assert_eq!(g("report_pretty_nonempty"), 1.0);
        {
            use arael::covariance::{CovMode, Covariance};
            let cov = f7.assemble_covariance(CovMode::AllMarginals).unwrap();
            let cc = cov.conditional_cov(&f7.items[0]).unwrap();
            assert_eq!(g("cond_n"), cc.nrows() as f64);
            assert_eq!(g("cond_item0"), cc[(0, 0)]);
        }

        // Compound params mirrored: value fields set exactly like the
        // FFI setters do (Default, then .value).
        {
            use arael::matrix::matrix3d;
            use arael::quatern::quaternd;
            let mut f10 = Fit::default();
            fill(&mut f10);
            let ea_a = vect3d::new(0.2, -0.3, 0.7);
            let ea_b = vect3d::new(-0.4, 0.1, -1.2);
            let rot_a = matrix3d::rotation_from_euler_angles(ea_a);
            let rot_b = matrix3d::rotation_from_euler_angles(ea_b);
            let mut rig = Rig::default();
            rig.target_u0 = rot_a[0];
            rig.target_u2 = rot_a[2];
            rig.target_q0 = rot_b[0];
            rig.target_q2 = rot_b[2];
            rig.target_g = 1.75;
            rig.ea_u.value = vect3d::new(0.15, -0.25, 0.6);
            rig.q.value = quaternd::from_euler_angles(vect3d::new(-0.35, 0.05, -1.1));
            rig.gain.g = 0.25;
            let r0 = f10.rigs.push(rig);
            let mut rig = Rig::default();
            rig.target_u0 = rot_b[0];
            rig.target_u2 = rot_b[2];
            rig.target_q0 = rot_a[0];
            rig.target_q2 = rot_a[2];
            rig.target_g = -0.5;
            rig.ea_u.value = ea_a;
            rig.ea_u.optimize = false;
            rig.q.value = quaternd::from_euler_angles(ea_a);
            rig.gain.g = -0.75;
            let r1 = f10.rigs.push(rig);
            let r10 = f10.solve_dense(&cfg).unwrap();
            assert_eq!(g("rig_status"), code(&r10.status));
            assert_eq!(g("rig_end"), r10.end_cost);
            let e0 = f10.rigs[r0].ea_u.value;
            assert_eq!(g("rig0_ea_x"), e0.x);
            assert_eq!(g("rig0_ea_y"), e0.y);
            assert_eq!(g("rig0_ea_z"), e0.z);
            let q0 = f10.rigs[r0].q.value;
            assert_eq!(g("rig0_q_t"), q0.t);
            assert_eq!(g("rig0_q_x"), q0.v.x);
            assert_eq!(g("rig0_q_y"), q0.v.y);
            assert_eq!(g("rig0_q_z"), q0.v.z);
            assert_eq!(g("rig0_g"), f10.rigs[r0].gain.g);
            // The solved values actually moved to their targets.
            assert!((e0 - ea_a).norm() < 1e-9, "{:?}", e0);
            assert!((f10.rigs[r0].gain.g - 1.75).abs() < 1e-9);
            // The frozen euler param stayed put.
            let e1 = f10.rigs[r1].ea_u.value;
            assert!((e1 - ea_a).norm() == 0.0, "{:?}", e1);
            assert_eq!(g("rig1_ea_x"), e1.x);
            assert_eq!(g("rig1_ea_y"), e1.y);
            assert_eq!(g("rig1_ea_z"), e1.z);
            assert_eq!(g("rig1_g"), f10.rigs[r1].gain.g);
        }

        // Observer termination: Break stops the solve.
        let mut f8 = Fit::default();
        fill(&mut f8);
        let cfg8 = cfg.clone().with_observer(
            |_: &arael::simple_lm::LmIter<'_, f64>| ControlFlow::Break(()));
        let r8 = f8.solve_dense(&cfg8).unwrap();
        assert!(matches!(r8.status, LmStatus::ObserverTerminated), "{:?}", r8.status);
        assert_eq!(g("obs_stop_status"), code(&r8.status));
        assert_eq!(g("obs_stop_iters"), r8.iterations as f64);
    }

    // Band solve mirrored: kd spans the whole parameter vector.
    let mut fitb = Fit::default();
    fill(&mut fitb);
    let mut x0 = std::vec::Vec::new();
    fitb.serialize64(&mut x0);
    let rb2 = arael::simple_lm::solve_band(&x0, 4, &mut fitb, &cfg).unwrap();
    fitb.deserialize64(&rb2.x);
    assert_eq!(g("band_status"), code(&rb2.status));
    assert_eq!(g("band_end"), rb2.end_cost);
    assert_eq!(g("band_m"), fitb.m.value);
    assert_eq!(g("band_c"), fitb.c.value);
    {
        use arael::covariance::{CovMode, Covariance};
        let cov = fitb.assemble_covariance(CovMode::AllMarginals).unwrap();
        let sd = cov.std_dev(&fitb.items[0]).unwrap();
        assert_eq!(g("band_cov_ok"), 1.0);
        assert_eq!(g("band_sd_n"), sd.len() as f64);
        assert_eq!(g("band_sd_item0"), sd[0]);
    }

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
    // Iteration parity over every container kind.
    let it_obs: f64 = f3.obs.iter().map(|o| o.y).sum();
    assert_eq!(g("it_obs_sum"), it_obs);
    let it_pose: f64 = f3.poses.iter().map(|p| p.pos.value.x).sum();
    assert_eq!(g("it_pose_sum"), it_pose);
    let it_marks: f64 = f3.marks.iter().map(|n| n.t).sum();
    assert_eq!(g("it_marks_sum"), it_marks);
    assert_eq!(g("it_marks_n"), f3.marks.iter().count() as f64);
    let it_arrow: f64 = f3.obs.iter().map(|o| o.x).sum::<f64>()
        + f3.marks.iter().map(|n| n.w).sum::<f64>();
    assert_eq!(g("it_arrow_sum"), it_arrow);
    let back_obs: f64 = f3.obs.iter().rev().enumerate()
        .map(|(k, o)| (k + 1) as f64 * o.y).sum();
    assert_eq!(g("back_obs"), back_obs);
    let back_marks: f64 = f3.marks.iter().collect::<Vec<_>>().iter().rev().enumerate()
        .map(|(k, n)| (k + 1) as f64 * n.t).sum();
    assert_eq!(g("back_marks"), back_marks);
    assert_eq!(g("r_obs"), back_obs);
    assert_eq!(g("r_marks"), back_marks);
    assert_eq!(g("s3_mark0_v"), f3.marks[m0].v.value);
    assert_eq!(g("s3_mark2_v"), f3.marks[m2].v.value);
    // Marks solved to their targets; the removed slot is gone.
    assert!((f3.marks[m0].v.value - 0.4).abs() < 1e-9);
    assert_eq!(f3.marks.len(), 2);

    // Container removal ops mirror Rust exactly.
    {
        let mut f4 = Fit::default();
        fill(&mut f4);
        f4.obs.pop();
        assert_eq!(g("ops_obs_after_pop"), f4.obs.len() as f64);
        f4.obs.truncate(2);
        assert_eq!(g("ops_obs_after_trunc"), f4.obs.len() as f64);
        f4.obs.clear();
        assert_eq!(g("ops_obs_after_clear"), 0.0);
        f4.poses.push_back(Pose::default());
        f4.poses.push_back(Pose::default());
        f4.poses.push_front(Pose::default());
        f4.poses.pop_front();
        f4.poses.pop_back();
        assert_eq!(g("ops_poses_left"), f4.poses.len() as f64);
        assert_eq!(g("ops_pop_empty"), 0.0);
        f4.marks.push(N::default());
        f4.marks.push(N::default());
        f4.marks.clear();
        assert_eq!(g("ops_marks_after_clear"), f4.marks.len() as f64);
    }

    // reserve/empty/contains/try_get/front/back mirror Rust; a C++
    // default-constructed ref is Rust's Ref::default() sentinel.
    {
        let b = |v: bool| v as i32 as f64;
        let mut f5 = Fit::default();
        f5.obs.reserve(64);
        f5.items.reserve(64);
        f5.poses.reserve(64);
        f5.marks.reserve(64);
        assert_eq!(g("cap_obs_empty"), b(f5.obs.is_empty()));
        f5.items.push(N { t: 0.25, ..Default::default() });
        f5.poses.push_back(Pose::default());
        f5.poses.push_back(Pose::default());
        f5.poses[0].pos.value.x = 1.5;
        f5.poses[1].pos.value.x = 2.5;
        let a5 = f5.marks.push(N::default());
        f5.marks[a5].t = 0.75;
        let a5b = f5.marks.push(N::default());
        assert_eq!(g("cap_obs_still_empty"), b(f5.obs.is_empty()));
        assert_eq!(g("cap_items_nonempty"), b(f5.items.is_empty()));
        let i5r = f5.items.ref_at(0);
        assert_eq!(g("cap_items_contains"), b(f5.items.contains_ref(i5r)));
        assert_eq!(g("cap_items_contains_default"),
            b(f5.items.contains_ref(Ref::default())));
        assert_eq!(g("cap_items_try_get"), f5.items.get(i5r).unwrap().t);
        assert_eq!(g("cap_poses_contains"),
            b(f5.poses.contains_ref(f5.poses.ref_at(1))));
        assert_eq!(g("cap_poses_front_x"), f5.poses.front().unwrap().pos.value.x);
        assert_eq!(g("cap_poses_back_x"), f5.poses.back().unwrap().pos.value.x);
        assert_eq!(g("cap_marks_contains"), b(f5.marks.contains_ref(a5)));
        assert_eq!(g("cap_marks_try_get"), f5.marks.get(a5).unwrap().t);
        f5.marks.remove(a5b);
        assert_eq!(g("cap_marks_stale_contains"), b(f5.marks.contains_ref(a5b)));
        assert_eq!(g("cap_marks_stale_try_get"), b(f5.marks.get(a5b).is_some()));
        // End refs and the null sentinel on empty containers.
        assert_eq!(g("cap_items_first_valid"), b(f5.items.first_ref().is_some()));
        assert_eq!(g("cap_items_last_get"),
            f5.items[f5.items.last_ref().unwrap()].t);
        assert_eq!(g("cap_poses_front_ref_x"),
            f5.poses[f5.poses.front_ref().unwrap()].pos.value.x);
        assert_eq!(g("cap_poses_back_ref_x"),
            f5.poses[f5.poses.back_ref().unwrap()].pos.value.x);
        let f6 = Fit::default();
        assert_eq!(g("cap_empty_first_valid"), b(f6.items.first_ref().is_some()));
        assert_eq!(g("cap_empty_front_valid"), b(f6.poses.front_ref().is_some()));
    }

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
