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

use arael::matrix::{matrix3, matrix3d, matrix3f};
use arael::model::{
    CrossBlock, EulerAngleParam, Param, QuaternionParam, SelfBlock, SimpleEulerAngleParam,
};
use arael::refs::{self, Ref};
use arael::utils::Float;
use arael::vect::{vect3, vect3d};
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

// Each parameterization is one generic model (the pose's rotation
// primitive is the ONLY difference); the f64 and f32 roots instantiate
// it. Bodies are shared; builders are generic with a per-pose closure.

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
struct SPose3<T: Float> {
    pos: Param<vect3<T>>,
    ea: SimpleEulerAngleParam<T>,
    prior: vect3<T>,
    prior_rot_t: matrix3<T>,
    has_prior: bool,
    hb: SelfBlock<SPose3<T>, T>,
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
struct SEdge3<T: Float> {
    #[arael(ref = root.poses)]
    a: Ref<SPose3<T>>,
    #[arael(ref = root.poses)]
    b: Ref<SPose3<T>>,
    dt: vect3<T>,
    rmeas_t: matrix3<T>,
    u_tt: matrix3<T>,
    u_tr: matrix3<T>,
    u_rr: matrix3<T>,
    hb: CrossBlock<SPose3<T>, SPose3<T>, T>,
}

#[arael::model]
#[arael(root)]
struct SGraph3 {
    poses: refs::Vec<SPose3<f64>>,
    edges: std::vec::Vec<SEdge3<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct SGraph3F {
    poses: refs::Vec<SPose3<f32>>,
    edges: std::vec::Vec<SEdge3<f32>>,
}

fn build_s_parts<T: Float>(ds: &Dataset3)
    -> (refs::Vec<SPose3<T>>, std::vec::Vec<SEdge3<T>>)
{
    let mut poses = refs::Vec::new();
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        poses.push(SPose3 {
            pos: Param::new(p.t.cast()),
            ea: SimpleEulerAngleParam::new(rot.get_euler_angles().cast()),
            prior: p.t.cast(),
            prior_rot_t: rot.transpose().cast(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        edges.push(SEdge3 {
            a: poses.ref_at(e.a),
            b: poses.ref_at(e.b),
            dt: e.dt.cast(),
            rmeas_t: quat_to_matrix(e.dq).transpose().cast(),
            u_tt: u_tt.cast(), u_tr: u_tr.cast(), u_rr: u_rr.cast(),
            hb: CrossBlock::new(),
        });
    }
    (poses, edges)
}

fn run_s(ds: &Dataset3, lambda0: f64) -> RunOut {
    let (poses, edges) = build_s_parts(ds);
    let mut g = SGraph3 { poses, edges };
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse(&params, &mut g, &cfg64(1, lambda0));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse(&params, &mut g, &cfg64(100, lambda0));
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

fn run_s_f32(ds: &Dataset3, lambda0: f64) -> RunOut {
    let (poses, edges) = build_s_parts(ds);
    let mut g = SGraph3F { poses, edges };
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_f32(&params, &mut g, &cfg32(1, lambda0 as f32));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_f32(&params, &mut g, &cfg32(100, lambda0 as f32));
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
struct EPose3<T: Float> {
    pos: Param<vect3<T>>,
    ea: EulerAngleParam<T>,
    prior: vect3<T>,
    prior_rot_t: matrix3<T>,
    has_prior: bool,
    hb: SelfBlock<EPose3<T>, T>,
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
struct EEdge3<T: Float> {
    #[arael(ref = root.poses)]
    a: Ref<EPose3<T>>,
    #[arael(ref = root.poses)]
    b: Ref<EPose3<T>>,
    dt: vect3<T>,
    rmeas_t: matrix3<T>,
    u_tt: matrix3<T>,
    u_tr: matrix3<T>,
    u_rr: matrix3<T>,
    hb: CrossBlock<EPose3<T>, EPose3<T>, T>,
}

#[arael::model]
#[arael(root)]
struct EGraph3 {
    poses: refs::Vec<EPose3<f64>>,
    edges: std::vec::Vec<EEdge3<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct EGraph3F {
    poses: refs::Vec<EPose3<f32>>,
    edges: std::vec::Vec<EEdge3<f32>>,
}

fn build_e_parts<T: Float>(ds: &Dataset3)
    -> (refs::Vec<EPose3<T>>, std::vec::Vec<EEdge3<T>>)
{
    let mut poses = refs::Vec::new();
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        poses.push(EPose3 {
            pos: Param::new(p.t.cast()),
            ea: EulerAngleParam::new(rot.get_euler_angles().cast()),
            prior: p.t.cast(),
            prior_rot_t: rot.transpose().cast(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        edges.push(EEdge3 {
            a: poses.ref_at(e.a),
            b: poses.ref_at(e.b),
            dt: e.dt.cast(),
            rmeas_t: quat_to_matrix(e.dq).transpose().cast(),
            u_tt: u_tt.cast(), u_tr: u_tr.cast(), u_rr: u_rr.cast(),
            hb: CrossBlock::new(),
        });
    }
    (poses, edges)
}

fn run_e(ds: &Dataset3, lambda0: f64) -> RunOut {
    let (poses, edges) = build_e_parts(ds);
    let mut g = EGraph3 { poses, edges };
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse(&params, &mut g, &cfg64(1, lambda0));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse(&params, &mut g, &cfg64(100, lambda0));
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

fn run_e_f32(ds: &Dataset3, lambda0: f64) -> RunOut {
    let (poses, edges) = build_e_parts(ds);
    let mut g = EGraph3F { poses, edges };
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_f32(&params, &mut g, &cfg32(1, lambda0 as f32));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_f32(&params, &mut g, &cfg32(100, lambda0 as f32));
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
struct QPose3<T: Float> {
    pos: Param<vect3<T>>,
    ea: QuaternionParam<T>,
    prior: vect3<T>,
    prior_rot_t: matrix3<T>,
    has_prior: bool,
    hb: SelfBlock<QPose3<T>, T>,
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
struct QEdge3<T: Float> {
    #[arael(ref = root.poses)]
    a: Ref<QPose3<T>>,
    #[arael(ref = root.poses)]
    b: Ref<QPose3<T>>,
    dt: vect3<T>,
    rmeas_t: matrix3<T>,
    u_tt: matrix3<T>,
    u_tr: matrix3<T>,
    u_rr: matrix3<T>,
    hb: CrossBlock<QPose3<T>, QPose3<T>, T>,
}

#[arael::model]
#[arael(root)]
struct QGraph3 {
    poses: refs::Vec<QPose3<f64>>,
    edges: std::vec::Vec<QEdge3<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct QGraph3F {
    poses: refs::Vec<QPose3<f32>>,
    edges: std::vec::Vec<QEdge3<f32>>,
}

fn build_q_parts<T: Float>(ds: &Dataset3)
    -> (refs::Vec<QPose3<T>>, std::vec::Vec<QEdge3<T>>)
{
    let mut poses = refs::Vec::new();
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        poses.push(QPose3 {
            pos: Param::new(p.t.cast()),
            // Same initial orientation as the other two (euler extraction),
            // routed through the quaternion reference.
            ea: QuaternionParam::from_euler_angles(rot.get_euler_angles().cast()),
            prior: p.t.cast(),
            prior_rot_t: rot.transpose().cast(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        edges.push(QEdge3 {
            a: poses.ref_at(e.a),
            b: poses.ref_at(e.b),
            dt: e.dt.cast(),
            rmeas_t: quat_to_matrix(e.dq).transpose().cast(),
            u_tt: u_tt.cast(), u_tr: u_tr.cast(), u_rr: u_rr.cast(),
            hb: CrossBlock::new(),
        });
    }
    (poses, edges)
}

fn run_q(ds: &Dataset3, lambda0: f64) -> RunOut {
    let (poses, edges) = build_q_parts(ds);
    let mut g = QGraph3 { poses, edges };
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse(&params, &mut g, &cfg64(1, lambda0));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse(&params, &mut g, &cfg64(100, lambda0));
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

fn run_q_f32(ds: &Dataset3, lambda0: f64) -> RunOut {
    let (poses, edges) = build_q_parts(ds);
    let mut g = QGraph3F { poses, edges };
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);
    let t0 = std::time::Instant::now();
    let _ = arael::simple_lm::solve_sparse_f32(&params, &mut g, &cfg32(1, lambda0 as f32));
    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = std::time::Instant::now();
    let result = arael::simple_lm::solve_sparse_f32(&params, &mut g, &cfg32(100, lambda0 as f32));
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
