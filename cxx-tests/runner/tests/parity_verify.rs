// The Rust mirror shared by the C++ (parity.rs) and Python
// (python.rs) parity tests: rebuilds the fixture problem with the
// model crate directly and compares every value the other side
// printed EXACTLY (same deterministic solver, %.17e round-trip).

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::Ref;
use arael::simple_lm::{LmConfig, LmProblem, LmStatus, RootProblem, SolveFailureKind};
use arael::vect::{vect2d, vect3d};
use cxx_fit::{Fit, GpsObs, N, Obs, Pose, Rig, Tie};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn fill(fit: &mut Fit) {
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
pub fn code(s: &LmStatus) -> f64 {
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

/// The shim's ReducedOrdering mapping (kept in lockstep with
/// emit_ffi); -1 = no reduction.
pub fn ord_code(o: Option<arael::simple_lm::ReducedOrdering>) -> f64 {
    use arael::simple_lm::ReducedOrdering;
    match o {
        Some(ReducedOrdering::NaturalBanded) => 0.0,
        Some(ReducedOrdering::NaturalDense) => 1.0,
        Some(ReducedOrdering::Amd) => 2.0,
        Some(ReducedOrdering::Nd) => 3.0,
        None => -1.0,
    }
}

pub fn verify(got: &std::collections::HashMap<String, f64>) {
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("output missing `{n}`"));

    // The same problem in Rust; every C++ value must match it exactly
    // (deterministic same code path; %.17e round-trips f64 exactly).
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut fit = Fit::default();
    fill(&mut fit);
    assert!(fit.validate().is_clean());
    assert_eq!(g("log_smoke"), 1.0);
    {
        use arael::simple_lm::LmStatus::*;
        for s in [Converged, CostThreshold, MaxIterations, GradientTolerance,
                  ParameterTolerance, PredictedReduction, LambdaCeiling,
                  DriverTerminated, ObserverTerminated, TimeLimit,
                  RetryBudgetExhausted, Aborted] {
            let c = code(&s) as i64;
            assert_eq!(g(&format!("st_ok_{c}")), s.is_success() as u8 as f64);
            assert_eq!(g(&format!("st_len_{c}")), s.as_str().len() as f64);
        }
    }
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

    // The ill_conditioned preset selects the Nielsen lambda driver;
    // its exposed fields equal conservative's, so only a solve's
    // trajectory can pin that the driver crossed the FFI.
    {
        let mut fic = Fit::default();
        fill(&mut fic);
        let ric = fic.solve_dense(&LmConfig::ill_conditioned()).unwrap();
        assert_eq!(g("ic_status"), code(&ric.status));
        assert_eq!(g("ic_iters"), ric.iterations as f64);
        assert_eq!(g("ic_end"), ric.end_cost);
        assert_eq!(g("ic_lambda"), ric.final_lambda);
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

    // Owned, independent assemblies.
    assert_eq!(g("cov2_ok"), 1.0);
    assert_eq!(g("cov_independent"), 1.0);

    // Per-constraint cost breakdown: identical labels, exact sums.
    {
        use arael::model::JacobianModel;
        use arael::simple_lm::RootProblem;
        let mut x = Vec::new();
        RootProblem::serialize(&mut fit, &mut x);
        let table = fit.calc_cost_table(&x);
        assert_eq!(g("ct_n"), table.len() as f64);
        for (label, value) in &table {
            assert_eq!(g(&format!("ct_{label}")), *value, "label {label}");
        }

        // Jacobian diagnostics mirrored exactly.
        let jac = fit.calc_jacobian(&x);
        assert_eq!(g("jac_m"), jac.num_residuals() as f64);
        assert_eq!(g("jac_n"), jac.num_params as f64);
        let sv = jac.singular_values();
        assert_eq!(g("jac_sv_n"), sv.len() as f64);
        assert_eq!(g("jac_sv0"), sv[0]);
        assert_eq!(g("jac_sv_last"), *sv.last().unwrap());
        let svn = jac.singular_values_column_normalised();
        assert_eq!(g("jac_svn0"), svn[0]);
        assert_eq!(g("jac_svn_last"), *svn.last().unwrap());
        let cn = jac.column_l2_norms();
        assert_eq!(g("jac_cn_n"), cn.len() as f64);
        assert_eq!(g("jac_cn0"), cn[0]);
        assert_eq!(g("jac_cn_last"), *cn.last().unwrap());
    }

    let mut fit2 = Fit::default();
    fill(&mut fit2);
    let r2 = fit2.solve_sparse(&cfg).unwrap();
    assert_eq!(g("sparse_status"), code(&r2.status));
    assert_eq!(g("sparse_end"), r2.end_cost);
    assert_eq!(g("sparse_m"), fit2.m.value);
    assert_eq!(g("sparse_c"), fit2.c.value);

    // The sparse backend's plan crosses the FFI field for field; the
    // dense result carries none.
    {
        use arael::simple_lm::{ReducedOrdering, SolverReport};
        let plan = match &r2.solver {
            Some(SolverReport::Schur(p)) => *p,
            _ => panic!("rust sparse solve carried no plan"),
        };
        assert_eq!(g("plan_has"), 1.0);
        assert_eq!(g("plan_reduced"), plan.reduced as u8 as f64);
        assert_eq!(g("plan_elim_blocks"), plan.eliminated_blocks as f64);
        assert_eq!(g("plan_elim_params"), plan.eliminated_params as f64);
        assert_eq!(g("plan_kept_params"), plan.kept_params as f64);
        assert_eq!(g("plan_bandwidth"), plan.kept_bandwidth as f64);
        assert_eq!(g("plan_envelope"), plan.envelope as u8 as f64);
        let ord = match plan.ordering {
            Some(ReducedOrdering::NaturalBanded) => 0.0,
            Some(ReducedOrdering::NaturalDense) => 1.0,
            Some(ReducedOrdering::Amd) => 2.0,
            Some(ReducedOrdering::Nd) => 3.0,
            None => -1.0,
        };
        assert_eq!(g("plan_ordering"), ord);
        assert_eq!(g("plan_flop_ratio_has"),
            plan.flop_ratio.is_some() as u8 as f64);
        assert_eq!(g("plan_flop_ratio"), plan.flop_ratio.unwrap_or(-1.0));
        assert!(r.solver.is_none(), "dense result must carry no plan");
        assert_eq!(g("plan_dense_none"), 1.0);
    }

    // Sparse options: the defaults are the Rust defaults, and each
    // knob drives the backend (pinned by the plan it produces).
    {
        use arael::simple_lm::{
            BlockSupernodalMode, EnvelopeMode, FaerOrdering, SchurPolicy,
            SolverReport, SparseFaer, SparseFaerOptions,
        };
        let d = SparseFaerOptions::default();
        let (fm, ofr) = match d.policy {
            SchurPolicy::Auto { flop_margin, obvious_flop_ratio } => {
                (flop_margin, obvious_flop_ratio)
            }
            _ => panic!("the default policy must be Auto"),
        };
        assert_eq!(g("so_schur"), 0.0);
        assert!(matches!(d.ordering, FaerOrdering::Auto));
        assert_eq!(g("so_ordering"), 0.0);
        assert!(matches!(d.envelope, EnvelopeMode::Auto));
        assert_eq!(g("so_envelope"), 0.0);
        assert_eq!(g("so_panel"), d.envelope_panel_width.unwrap_or(0) as f64);
        assert_eq!(g("so_supernodal"), d.supernodal as u8 as f64);
        assert_eq!(g("so_narrow_band"), d.narrow_band as u8 as f64);
        assert_eq!(g("so_flop_margin"), fm);
        assert_eq!(g("so_obvious"), ofr);
        assert!(matches!(d.block_supernodal, BlockSupernodalMode::Auto));
        assert_eq!(g("so_block_sn"), 0.0);
        assert_eq!(g("so_bs_batch"), d.block_supernodal_batch.unwrap_or(0.0));
        assert_eq!(g("so_bs_lean"), d.block_supernodal_memory_lean as u8 as f64);

        let mut f11 = Fit::default();
        fill(&mut f11);
        let mut s11 = SparseFaer::from_options(&SparseFaerOptions::auto()
            .with_policy(SchurPolicy::Force)
            .with_ordering(FaerOrdering::Natural)
            .with_envelope_schur(EnvelopeMode::Always));
        let r11 = f11.solve_with(&mut s11, &cfg).unwrap();
        assert_eq!(g("opt_end"), r11.end_cost);
        let p11 = match r11.solver {
            Some(SolverReport::Schur(p)) => p,
            _ => panic!("forced sparse solve carried no plan"),
        };
        assert!(p11.reduced, "Force must reduce");
        assert!(p11.envelope, "Always with a natural order must take the envelope");
        assert_eq!(g("opt_reduced"), p11.reduced as u8 as f64);
        assert_eq!(g("opt_envelope"), p11.envelope as u8 as f64);
        assert_eq!(g("opt_ordering"), ord_code(p11.ordering));

        let mut f12 = Fit::default();
        fill(&mut f12);
        let mut s12 = SparseFaer::from_options(&SparseFaerOptions::auto()
            .with_policy(SchurPolicy::Force)
            .with_ordering(FaerOrdering::Amd)
            .with_envelope_schur(EnvelopeMode::Never)
            .with_supernodal(false));
        let r12 = f12.solve_with(&mut s12, &cfg).unwrap();
        assert_eq!(g("opt2_end"), r12.end_cost);
        let p12 = match r12.solver {
            Some(SolverReport::Schur(p)) => p,
            _ => panic!("forced sparse solve carried no plan"),
        };
        assert!(!p12.envelope, "Never must not take the envelope");
        assert_eq!(g("opt2_reduced"), p12.reduced as u8 as f64);
        assert_eq!(g("opt2_envelope"), p12.envelope as u8 as f64);
        assert_eq!(g("opt2_ordering"), ord_code(p12.ordering));

        // The block supernodal knobs, against the same solves in Rust.
        let mut f13 = Fit::default();
        fill(&mut f13);
        let mut s13 = SparseFaer::from_options(&SparseFaerOptions::auto()
            .with_policy(SchurPolicy::Force)
            .with_envelope_schur(EnvelopeMode::Never)
            .with_block_supernodal(BlockSupernodalMode::Always)
            .with_block_supernodal_memory_lean(true));
        let r13 = f13.solve_with(&mut s13, &cfg).unwrap();
        let p13 = match r13.solver {
            Some(SolverReport::Schur(p)) => p,
            _ => panic!("forced sparse solve carried no plan"),
        };
        assert!(p13.block_supernodal, "Always must take the block supernodal");
        assert_eq!(g("opt3_end"), r13.end_cost);
        assert_eq!(g("opt3_block_sn"), p13.block_supernodal as u8 as f64);

        let mut f14 = Fit::default();
        fill(&mut f14);
        let mut s14 = SparseFaer::from_options(&SparseFaerOptions::auto()
            .with_policy(SchurPolicy::Force)
            .with_envelope_schur(EnvelopeMode::Never)
            .with_block_supernodal(BlockSupernodalMode::Never));
        let r14 = f14.solve_with(&mut s14, &cfg).unwrap();
        let p14 = match r14.solver {
            Some(SolverReport::Schur(p)) => p,
            _ => panic!("forced sparse solve carried no plan"),
        };
        assert!(!p14.block_supernodal, "Never must leave it to the scalar route");
        assert_eq!(g("opt4_end"), r14.end_cost);
        assert_eq!(g("opt4_block_sn"), p14.block_supernodal as u8 as f64);
    }

    // LmSession: warm solves reuse the analysis and stay bit-identical
    // to cold ones; a parameter-count change re-analyzes by itself.
    {
        use arael::simple_lm::{
            EnvelopeMode, FaerOrdering, LmSession, SchurPolicy, SolverReport,
            SparseFaer, SparseFaerOptions,
        };
        let mut f13 = Fit::default();
        fill(&mut f13);
        let mut sess = LmSession::new(SparseFaer::new());
        let rs1 = sess.solve(&mut f13, &cfg).unwrap();
        assert_eq!(g("sess_end1"), rs1.end_cost);
        f13.m.value = 0.0;
        f13.c.value = 0.0;
        for i in 0..3 {
            f13.items[i].v.value = 0.0;
        }
        let rs2 = sess.solve(&mut f13, &cfg).unwrap();
        assert_eq!(g("sess_end2"), rs2.end_cost);
        assert_eq!(rs2.end_cost, rs1.end_cost, "warm must equal cold");
        assert_eq!(g("sess_warm_equals_cold"), 1.0);
        sess.invalidate();
        f13.m.value = 0.0;
        f13.c.value = 0.0;
        for i in 0..3 {
            f13.items[i].v.value = 0.0;
        }
        let rs3 = sess.solve(&mut f13, &cfg).unwrap();
        assert_eq!(rs3.end_cost, rs1.end_cost, "cold-again must agree");
        assert_eq!(g("sess_invalidate_agrees"), 1.0);
        f13.items.push(N {
            v: Param::default(),
            t: 0.5,
            w: 1.0,
            hb: SelfBlock::new(),
        });
        let rs4 = sess.solve(&mut f13, &cfg).unwrap();
        assert_eq!(g("sess_end4"), rs4.end_cost);

        // A session built over explicit options follows them.
        let mut f14 = Fit::default();
        fill(&mut f14);
        let mut sessf = LmSession::new(SparseFaer::from_options(
            &SparseFaerOptions::auto()
                .with_policy(SchurPolicy::Force)
                .with_ordering(FaerOrdering::Natural)
                .with_envelope_schur(EnvelopeMode::Always),
        ));
        let rs5 = sessf.solve(&mut f14, &cfg).unwrap();
        assert_eq!(g("sessf_end"), rs5.end_cost);
        let p5 = match rs5.solver {
            Some(SolverReport::Schur(p)) => p,
            _ => panic!("session solve carried no plan"),
        };
        assert!(p5.envelope, "the options must reach the session's backend");
        assert_eq!(g("sessf_envelope"), p5.envelope as u8 as f64);
    }

    // The iterative Schur route through the options struct.
    {
        use arael::simple_lm::{
            CgOptions, SolverReport, SparseFaer, SparseFaerOptions,
        };
        let mut f15 = Fit::default();
        fill(&mut f15);
        let mut s15 = SparseFaer::from_options(
            &SparseFaerOptions::forced_schur()
                .with_iterative_schur(CgOptions::default()));
        let r15 = f15.solve_with(&mut s15, &cfg).unwrap();
        assert_eq!(g("cg_end"), r15.end_cost);
        let p15 = match r15.solver {
            Some(SolverReport::Schur(p)) => p,
            _ => panic!("iterative solve carried no plan"),
        };
        assert!(p15.cg_iterations.is_some(), "CG total must be reported");
        assert_eq!(g("cg_iters_has"), 1.0);
        assert_eq!(g("cg_iters"), p15.cg_iterations.unwrap() as f64);
    }

    // The Python enum setters validate (C++ is typed; the C++ driver
    // does not print this name, so it is optional).
    if let Some(v) = got.get("enum_setter_validates") {
        assert_eq!(*v, 1.0, "python enum setter must reject bad values");
    }
    if let Some(v) = got.get("bad_tag_raises") {
        assert_eq!(*v, 1.0, "a raw bad tag must raise, not abort");
    }

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
        assert_eq!(g("report_default_empty"), 1.0);
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
        // Per-attempt timeline mirrored exactly.
        assert!(!t.steps.is_empty());
        assert_eq!(g("steps_len"), t.steps.len() as f64);
        let s0 = &t.steps[0];
        let sn = t.steps.last().unwrap();
        assert_eq!(g("step0_iter"), s0.iter as f64);
        assert_eq!(g("step0_inner"), s0.inner as f64);
        assert_eq!(g("step0_accepted"), s0.accepted as i32 as f64);
        assert_eq!(g("step0_lambda"), s0.lambda);
        assert_eq!(g("step0_cost"), s0.cost);
        assert_eq!(g("step0_new_cost"), s0.new_cost);
        assert_eq!(g("step0_step_norm"), s0.step_norm);
        assert_eq!(g("step0_grad_max"), s0.grad_max);
        assert_eq!(g("stepN_iter"), sn.iter as f64);
        assert_eq!(g("stepN_accepted"), sn.accepted as i32 as f64);
        assert_eq!(g("stepN_cost"), sn.cost);
        assert_eq!(g("stepN_new_cost"), sn.new_cost);
        assert_eq!(g("steps_ok"), 1.0);
        assert!(!r7.report().is_empty());
        assert_eq!(g("report_nonempty"), 1.0);
        assert_eq!(g("report_pretty_nonempty"), 1.0);
        assert_eq!(g("report_survives_next_solve"), 1.0);
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
    fitb.serialize(&mut x0);
    let rb2 = arael::simple_lm::solve_band(&x0, 4, &mut fitb, &cfg).unwrap();
    fitb.deserialize(&rb2.x);
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
        // 2D heading: solves so R(heading).col(0) matches target_dir at th.
        let th = 0.2 + 0.3 * i as f64;
        p.target_dir = vect2d::new(th.cos(), th.sin());
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
        // The 2D heading solved through the computed rotation matrix.
        assert_eq!(g(&format!("s3_p{i}_h")), f3.poses[i].heading.angle.value, "p{i}.h");
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
    assert_eq!(g("bad_partial_has"), e.partial.is_some() as u8 as f64);
    // The structured failure crosses with its indices intact.
    {
        use arael::simple_lm::DiagonalFault;
        let (fault, param) = match e.kind {
            SolveFailureKind::DegenerateDiagonal { param, fault } => (
                match fault {
                    DiagonalFault::Nan => 0.0,
                    DiagonalFault::Negative => 1.0,
                    DiagonalFault::Zero => 2.0,
                },
                param as f64,
            ),
            _ => unreachable!(),
        };
        assert_eq!(g("bad_kind"), 9.0);
        assert_eq!(g("bad_fault"), fault);
        assert_eq!(g("bad_param"), param);
    }
}
