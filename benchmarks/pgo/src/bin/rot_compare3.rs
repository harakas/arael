// Interleaved comparison of the three SO(3) rotation parameterizations on the
// two 3D (SE3) pose-graph datasets (sphere2500, parking-garage). The model is
// the canonical quaternion-vector between residual from arael_runner3.rs,
// defined once per primitive -- SimpleEulerAngleParam (euler optimized
// directly), EulerAngleParam (euler delta re-centered on a matrix reference),
// and QuaternionParam (rotation-vector delta re-centered on a unit quaternion).
//
// The three run INTERLEAVED per round (Simple, Euler, Quaternion, repeat) so
// machine drift is shared across the comparison rather than accumulating
// against one primitive; each keeps its fastest round. Reports ms/iter,
// iteration count, and whether it reached the common optimum -- so gimbal-lock
// non-convergence (Simple, on the sphere's arbitrary orientations) shows up
// explicitly. Answers: which is faster AND converges on both 3D datasets.
//
// Run: cargo run -r --bin rot_compare3        (ROUNDS env overrides, default 20)

#[path = "../g2o3.rs"]
mod g2o3;

use arael::matrix::{matrix3d, matrix3f};
use arael::model::{
    CrossBlock, EulerAngleParam, Model, Param, QuaternionParam, SelfBlock, SimpleEulerAngleParam,
};
use arael::refs::{self, Ref};
use arael::vect::{vect3d, vect3f};
use g2o3::{aligned_rmse3, load3, matrix_to_quat, quat_to_matrix, reference_cost3, Dataset3, Pose3In};

// Same LM knobs as the main arael 3D runner (arael_runner.rs::cfg64_with_lambda),
// inlined here so this bin does not pull in the 2D g2o module that runner needs.
fn cfg64(max_iters: usize, initial_lambda: f64) -> arael::simple_lm::LmConfig<f64> {
    arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        max_iters,
        initial_lambda,
        ..Default::default()
    }
}

fn cfg32(max_iters: usize, initial_lambda: f32) -> arael::simple_lm::LmConfig<f32> {
    arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        max_iters,
        initial_lambda,
        ..Default::default()
    }
}

// Initial damping, problem-appropriate for the 3D datasets (matches
// arael_runner3.rs::lambda0_3d; env ARAEL_LAMBDA0 overrides).
fn lambda0_3d() -> f64 {
    std::env::var("ARAEL_LAMBDA0").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-10)
}

// ===================================================================== Simple

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = spose3.prior_rot_t * spose3.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [spose3.pos.x - spose3.prior.x,
     spose3.pos.y - spose3.prior.y,
     spose3.pos.z - spose3.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
struct SPose3 {
    pos: Param<vect3d>,
    ea: SimpleEulerAngleParam<f64>,
    prior: vect3d,
    prior_rot_t: matrix3d,
    has_prior: bool,
    hb: SelfBlock<SPose3>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = sedge3.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - sedge3.dt;
    let wt = sedge3.u_tt * terr + sedge3.u_tr * rrot;
    let wr = sedge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
struct SEdge3 {
    #[arael(ref = root.poses)]
    a: Ref<SPose3>,
    #[arael(ref = root.poses)]
    b: Ref<SPose3>,
    dt: vect3d,
    rmeas_t: matrix3d,
    u_tt: matrix3d,
    u_tr: matrix3d,
    u_rr: matrix3d,
    hb: CrossBlock<SPose3, SPose3>,
}

#[arael::model]
#[arael(root)]
struct SGraph3 {
    poses: refs::Vec<SPose3>,
    edges: std::vec::Vec<SEdge3>,
}

fn build_s(ds: &Dataset3) -> SGraph3 {
    let mut g = SGraph3 { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        g.poses.push(SPose3 {
            pos: Param::new(p.t),
            ea: SimpleEulerAngleParam::new(rot.get_euler_angles()),
            prior: p.t,
            prior_rot_t: rot.transpose(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(SEdge3 {
            a, b, dt: e.dt,
            rmeas_t: quat_to_matrix(e.dq).transpose(),
            u_tt, u_tr, u_rr,
            hb: CrossBlock::new(),
        });
    }
    g
}

fn run_s(ds: &Dataset3, lambda0: f64) -> RunOut {
    let mut g = build_s(ds);
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(1, lambda0));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(100, lambda0));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize64(&result.x);
    let poses = g.poses.iter()
        .map(|p| Pose3In {
            t: p.pos.value,
            q: matrix_to_quat(matrix3d::rotation_from_euler_angles(p.ea.value)),
        })
        .collect();
    RunOut { solve_ms, first_ms, iters: result.iterations, accepted: result.accepted_iterations, poses }
}

// ====================================================================== Euler

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = epose3.prior_rot_t * epose3.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [epose3.pos.x - epose3.prior.x,
     epose3.pos.y - epose3.prior.y,
     epose3.pos.z - epose3.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
struct EPose3 {
    pos: Param<vect3d>,
    ea: EulerAngleParam<f64>,
    prior: vect3d,
    prior_rot_t: matrix3d,
    has_prior: bool,
    hb: SelfBlock<EPose3>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = eedge3.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - eedge3.dt;
    let wt = eedge3.u_tt * terr + eedge3.u_tr * rrot;
    let wr = eedge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
struct EEdge3 {
    #[arael(ref = root.poses)]
    a: Ref<EPose3>,
    #[arael(ref = root.poses)]
    b: Ref<EPose3>,
    dt: vect3d,
    rmeas_t: matrix3d,
    u_tt: matrix3d,
    u_tr: matrix3d,
    u_rr: matrix3d,
    hb: CrossBlock<EPose3, EPose3>,
}

#[arael::model]
#[arael(root)]
struct EGraph3 {
    poses: refs::Vec<EPose3>,
    edges: std::vec::Vec<EEdge3>,
}

fn build_e(ds: &Dataset3) -> EGraph3 {
    let mut g = EGraph3 { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        g.poses.push(EPose3 {
            pos: Param::new(p.t),
            ea: EulerAngleParam::new(rot.get_euler_angles()),
            prior: p.t,
            prior_rot_t: rot.transpose(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(EEdge3 {
            a, b, dt: e.dt,
            rmeas_t: quat_to_matrix(e.dq).transpose(),
            u_tt, u_tr, u_rr,
            hb: CrossBlock::new(),
        });
    }
    g
}

fn run_e(ds: &Dataset3, lambda0: f64) -> RunOut {
    let mut g = build_e(ds);
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(1, lambda0));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(100, lambda0));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize64(&result.x);
    let poses = g.poses.iter()
        .map(|p| Pose3In {
            t: p.pos.value,
            q: matrix_to_quat(matrix3d::rotation_from_euler_angles(p.ea.value)),
        })
        .collect();
    RunOut { solve_ms, first_ms, iters: result.iterations, accepted: result.accepted_iterations, poses }
}

// ================================================================= Quaternion

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = qpose3.prior_rot_t * qpose3.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [qpose3.pos.x - qpose3.prior.x,
     qpose3.pos.y - qpose3.prior.y,
     qpose3.pos.z - qpose3.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
struct QPose3 {
    pos: Param<vect3d>,
    ea: QuaternionParam<f64>,
    prior: vect3d,
    prior_rot_t: matrix3d,
    has_prior: bool,
    hb: SelfBlock<QPose3>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = qedge3.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - qedge3.dt;
    let wt = qedge3.u_tt * terr + qedge3.u_tr * rrot;
    let wr = qedge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
struct QEdge3 {
    #[arael(ref = root.poses)]
    a: Ref<QPose3>,
    #[arael(ref = root.poses)]
    b: Ref<QPose3>,
    dt: vect3d,
    rmeas_t: matrix3d,
    u_tt: matrix3d,
    u_tr: matrix3d,
    u_rr: matrix3d,
    hb: CrossBlock<QPose3, QPose3>,
}

#[arael::model]
#[arael(root)]
struct QGraph3 {
    poses: refs::Vec<QPose3>,
    edges: std::vec::Vec<QEdge3>,
}

fn build_q(ds: &Dataset3) -> QGraph3 {
    let mut g = QGraph3 { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        g.poses.push(QPose3 {
            pos: Param::new(p.t),
            // Same initial orientation as the other two (euler extraction),
            // routed through the quaternion reference.
            ea: QuaternionParam::from_euler_angles(rot.get_euler_angles()),
            prior: p.t,
            prior_rot_t: rot.transpose(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(QEdge3 {
            a, b, dt: e.dt,
            rmeas_t: quat_to_matrix(e.dq).transpose(),
            u_tt, u_tr, u_rr,
            hb: CrossBlock::new(),
        });
    }
    g
}

fn run_q(ds: &Dataset3, lambda0: f64) -> RunOut {
    let mut g = build_q(ds);
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(1, lambda0));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer(&params, &mut g, &cfg64(100, lambda0));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize64(&result.x);
    let poses = g.poses.iter()
        .map(|p| {
            // QuaternionParam.value is the solved unit quaternion (delta folded
            // in on deserialize); read it out directly as (x, y, z, w).
            let v = p.ea.value;
            Pose3In { t: p.pos.value, q: [v.v.x, v.v.y, v.v.z, v.t] }
        })
        .collect();
    RunOut { solve_ms, first_ms, iters: result.iterations, accepted: result.accepted_iterations, poses }
}

// ================================================================= Simple f32

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = spose3f.prior_rot_t * spose3f.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [spose3f.pos.x - spose3f.prior.x,
     spose3f.pos.y - spose3f.prior.y,
     spose3f.pos.z - spose3f.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
struct SPose3F {
    pos: Param<vect3f>,
    ea: SimpleEulerAngleParam<f32>,
    prior: vect3f,
    prior_rot_t: matrix3f,
    has_prior: bool,
    hb: SelfBlock<SPose3F, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = sedge3f.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - sedge3f.dt;
    let wt = sedge3f.u_tt * terr + sedge3f.u_tr * rrot;
    let wr = sedge3f.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
struct SEdge3F {
    #[arael(ref = root.poses)]
    a: Ref<SPose3F>,
    #[arael(ref = root.poses)]
    b: Ref<SPose3F>,
    dt: vect3f,
    rmeas_t: matrix3f,
    u_tt: matrix3f,
    u_tr: matrix3f,
    u_rr: matrix3f,
    hb: CrossBlock<SPose3F, SPose3F, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct SGraph3F {
    poses: refs::Vec<SPose3F>,
    edges: std::vec::Vec<SEdge3F>,
}

// ================================================================== Euler f32

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = epose3f.prior_rot_t * epose3f.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [epose3f.pos.x - epose3f.prior.x,
     epose3f.pos.y - epose3f.prior.y,
     epose3f.pos.z - epose3f.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
struct EPose3F {
    pos: Param<vect3f>,
    ea: EulerAngleParam<f32>,
    prior: vect3f,
    prior_rot_t: matrix3f,
    has_prior: bool,
    hb: SelfBlock<EPose3F, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = eedge3f.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - eedge3f.dt;
    let wt = eedge3f.u_tt * terr + eedge3f.u_tr * rrot;
    let wr = eedge3f.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
struct EEdge3F {
    #[arael(ref = root.poses)]
    a: Ref<EPose3F>,
    #[arael(ref = root.poses)]
    b: Ref<EPose3F>,
    dt: vect3f,
    rmeas_t: matrix3f,
    u_tt: matrix3f,
    u_tr: matrix3f,
    u_rr: matrix3f,
    hb: CrossBlock<EPose3F, EPose3F, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct EGraph3F {
    poses: refs::Vec<EPose3F>,
    edges: std::vec::Vec<EEdge3F>,
}

// ============================================================= Quaternion f32

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = qpose3f.prior_rot_t * qpose3f.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [qpose3f.pos.x - qpose3f.prior.x,
     qpose3f.pos.y - qpose3f.prior.y,
     qpose3f.pos.z - qpose3f.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
struct QPose3F {
    pos: Param<vect3f>,
    ea: QuaternionParam<f32>,
    prior: vect3f,
    prior_rot_t: matrix3f,
    has_prior: bool,
    hb: SelfBlock<QPose3F, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = qedge3f.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - qedge3f.dt;
    let wt = qedge3f.u_tt * terr + qedge3f.u_tr * rrot;
    let wr = qedge3f.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
struct QEdge3F {
    #[arael(ref = root.poses)]
    a: Ref<QPose3F>,
    #[arael(ref = root.poses)]
    b: Ref<QPose3F>,
    dt: vect3f,
    rmeas_t: matrix3f,
    u_tt: matrix3f,
    u_tr: matrix3f,
    u_rr: matrix3f,
    hb: CrossBlock<QPose3F, QPose3F, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct QGraph3F {
    poses: refs::Vec<QPose3F>,
    edges: std::vec::Vec<QEdge3F>,
}

// f32 builders + runners. The dataset is f64; cast per-pose/edge to f32.

fn build_s_f32(ds: &Dataset3) -> SGraph3F {
    let mut g = SGraph3F { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        g.poses.push(SPose3F {
            pos: Param::new(vect3f::from(p.t)),
            ea: SimpleEulerAngleParam::new(vect3f::from(rot.get_euler_angles())),
            prior: vect3f::from(p.t),
            prior_rot_t: matrix3f::from(rot.transpose()),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(SEdge3F {
            a, b, dt: vect3f::from(e.dt),
            rmeas_t: matrix3f::from(quat_to_matrix(e.dq).transpose()),
            u_tt: matrix3f::from(u_tt), u_tr: matrix3f::from(u_tr), u_rr: matrix3f::from(u_rr),
            hb: CrossBlock::new(),
        });
    }
    g
}

fn build_e_f32(ds: &Dataset3) -> EGraph3F {
    let mut g = EGraph3F { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        g.poses.push(EPose3F {
            pos: Param::new(vect3f::from(p.t)),
            ea: EulerAngleParam::new(vect3f::from(rot.get_euler_angles())),
            prior: vect3f::from(p.t),
            prior_rot_t: matrix3f::from(rot.transpose()),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(EEdge3F {
            a, b, dt: vect3f::from(e.dt),
            rmeas_t: matrix3f::from(quat_to_matrix(e.dq).transpose()),
            u_tt: matrix3f::from(u_tt), u_tr: matrix3f::from(u_tr), u_rr: matrix3f::from(u_rr),
            hb: CrossBlock::new(),
        });
    }
    g
}

fn build_q_f32(ds: &Dataset3) -> QGraph3F {
    let mut g = QGraph3F { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        g.poses.push(QPose3F {
            pos: Param::new(vect3f::from(p.t)),
            ea: QuaternionParam::from_euler_angles(vect3f::from(rot.get_euler_angles())),
            prior: vect3f::from(p.t),
            prior_rot_t: matrix3f::from(rot.transpose()),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(QEdge3F {
            a, b, dt: vect3f::from(e.dt),
            rmeas_t: matrix3f::from(quat_to_matrix(e.dq).transpose()),
            u_tt: matrix3f::from(u_tt), u_tr: matrix3f::from(u_tr), u_rr: matrix3f::from(u_rr),
            hb: CrossBlock::new(),
        });
    }
    g
}

fn run_s_f32(ds: &Dataset3, lambda0: f64) -> RunOut {
    let mut g = build_s_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(1, lambda0 as f32));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(100, lambda0 as f32));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize32(&result.x);
    let poses = g.poses.iter()
        .map(|p| Pose3In {
            t: vect3d::from(p.pos.value),
            q: matrix_to_quat(matrix3d::from(matrix3f::rotation_from_euler_angles(p.ea.value))),
        })
        .collect();
    RunOut { solve_ms, first_ms, iters: result.iterations, accepted: result.accepted_iterations, poses }
}

fn run_e_f32(ds: &Dataset3, lambda0: f64) -> RunOut {
    let mut g = build_e_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(1, lambda0 as f32));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(100, lambda0 as f32));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize32(&result.x);
    let poses = g.poses.iter()
        .map(|p| Pose3In {
            t: vect3d::from(p.pos.value),
            q: matrix_to_quat(matrix3d::from(matrix3f::rotation_from_euler_angles(p.ea.value))),
        })
        .collect();
    RunOut { solve_ms, first_ms, iters: result.iterations, accepted: result.accepted_iterations, poses }
}

fn run_q_f32(ds: &Dataset3, lambda0: f64) -> RunOut {
    let mut g = build_q_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(1, lambda0 as f32));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_faer_f32(&params, &mut g, &cfg32(100, lambda0 as f32));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize32(&result.x);
    let poses = g.poses.iter()
        .map(|p| {
            let v = p.ea.value;
            Pose3In { t: vect3d::from(p.pos.value), q: [v.v.x as f64, v.v.y as f64, v.v.z as f64, v.t as f64] }
        })
        .collect();
    RunOut { solve_ms, first_ms, iters: result.iterations, accepted: result.accepted_iterations, poses }
}

// ======================================================================= main

struct RunOut {
    solve_ms: f64,
    first_ms: f64,
    iters: usize,
    accepted: usize,
    poses: Vec<Pose3In>,
}

struct Best {
    name: &'static str,
    solve_ms: f64,
    first_ms: f64,
    iters: usize,
    accepted: usize,
    poses: Vec<Pose3In>,
}

fn main() {
    let bench_dir = std::env::current_dir().unwrap();
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let lambda0 = lambda0_3d();

    let datasets = [
        ("sphere2500", bench_dir.join("datasets/sphere2500.g2o")),
        ("parking-garage", bench_dir.join("datasets/parking-garage.g2o")),
    ];

    println!(
        "3D rotation-parameterization comparison (interleaved, median of fastest of {} rounds, lambda0={:.0e})\n",
        rounds, lambda0
    );

    type Runner = fn(&Dataset3, f64) -> RunOut;
    // f64 trio then f32 trio; the interleave runs them in this order per round.
    let primitives: [(&str, Runner); 6] = [
        ("SimpleEulerAngleParam f64", run_s),
        ("EulerAngleParam f64", run_e),
        ("QuaternionParam f64", run_q),
        ("SimpleEulerAngleParam f32", run_s_f32),
        ("EulerAngleParam f32", run_e_f32),
        ("QuaternionParam f32", run_q_f32),
    ];

    for (name, path) in &datasets {
        let ds = load3(path.to_str().unwrap());
        let init = reference_cost3(&ds, &ds.poses);
        println!("=== {} : {} poses, {} edges (initial cost {:.3}) ===",
            name, ds.poses.len(), ds.edges.len(), init);

        // Interleaved: each round runs all six back to back; keep each one's
        // fastest round (poses/iters are deterministic).
        let mut best: Vec<Best> = primitives.iter()
            .map(|(pname, _)| Best {
                name: pname, solve_ms: f64::INFINITY, first_ms: f64::INFINITY,
                iters: 0, accepted: 0, poses: Vec::new(),
            })
            .collect();
        for _ in 0..rounds {
            for (i, (_, run)) in primitives.iter().enumerate() {
                let o = run(&ds, lambda0);
                if o.solve_ms < best[i].solve_ms {
                    best[i].solve_ms = o.solve_ms;
                    best[i].iters = o.iters;
                    best[i].accepted = o.accepted;
                    best[i].poses = o.poses;
                }
                best[i].first_ms = best[i].first_ms.min(o.first_ms);
            }
        }

        // Common optimum = the lowest final cost across all six (an f64 one);
        // flag each as converged if within 1% of it AND aligned RMSE < 5 cm.
        let costs: Vec<f64> = best.iter().map(|b| reference_cost3(&ds, &b.poses)).collect();
        let best_i = (0..best.len()).min_by(|&a, &b| costs[a].partial_cmp(&costs[b]).unwrap()).unwrap();
        let best_cost = costs[best_i];
        let ref_poses = best[best_i].poses.clone();

        println!("{:<26} {:>9} {:>11} {:>11} {:>14} {:>10} {:>10}",
            "primitive", "ms/iter", "iters(acc)", "1st-iter ms", "final cost", "aln RMSE", "converged");
        for (i, b) in best.iter().enumerate() {
            if i == 3 { println!(); } // blank line between the f64 and f32 trios
            let ms_iter = b.solve_ms / b.iters as f64;
            let rmse = aligned_rmse3(&b.poses, &ref_poses);
            let within = (costs[i] - best_cost) / best_cost.max(1e-30) < 0.01;
            let converged = within && rmse < 0.05;
            println!("{:<26} {:>9.3} {:>7}({:>2}) {:>11.2} {:>14.4} {:>10.4} {:>10}",
                b.name, ms_iter, b.iters, b.accepted, b.first_ms, costs[i], rmse,
                if converged { "yes" } else { "NO" });
        }
        println!();
    }
}
