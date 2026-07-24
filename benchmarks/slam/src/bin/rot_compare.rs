// Rotation-parameterization comparison on the SLAM scene, three variants:
//   - SimpleEulerAngleParam: naive euler angles optimized directly (no
//     reference frame, no re-centering)
//   - EulerAngleParam:       euler-angle delta around a matrix reference,
//     re-centered each accepted step
//   - QuaternionParam:       rotation-vector / exp-map delta around a
//     quaternion reference, re-centered each accepted step
//
// Same scene, same solver config, runs INTERLEAVED (build+solve each variant
// per round, alternating) so CPU thermal/noise hits all three equally.
// gather_timing is on, so each solve reports where its time went: assembly
// (residual + Jacobian + Hessian), the damped linear solve, trial-point cost
// evaluation, and post-step re-centering. The first assembly and first
// factorization are reported apart -- they establish the sparsity pattern and
// the symbolic factorization, a one-time structure cost.
//
//   cargo run -r --bin rot_compare

#[path = "../scene.rs"]
mod scene;

use arael::model::{CrossBlock, EulerAngleParam, Param, QuaternionParam, SelfBlock, SimpleEulerAngleParam};
use arael::matrix::matrix3d;
use arael::refs::{self, Ref};
use arael::simple_lm::{self, LmConfig, LmProblem, LmTiming};
use arael::vect::{vect2d, vect3d};
use scene::Scene;
use std::time::Duration;

fn cfg() -> LmConfig<f64> {
    LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        min_iters: 1,
        max_iters: 200,
        initial_lambda: 1e-8,
        gather_timing: true,
        ..Default::default()
    }
}

// ===========================================================================
// Naive-euler-angle variant (SimpleEulerAngleParam, optimized directly)
// ===========================================================================
#[arael::model]
#[arael(constraint(hb_pose, {
    let raw = sepose.pos - sepose.gps_pos;
    let rt = sepose.gps_cov_r.transpose() * raw;
    [rt.x * sepose.gps_cov_isigma.x, rt.y * sepose.gps_cov_isigma.y, rt.z * sepose.gps_cov_isigma.z]
}))]
#[arael(constraint(hb_pose, {
    let pd = sepose.pos - sepose.prior_pos;
    let ed = sepose.ea - sepose.prior_ea;
    [pd.x * sepath.drift_pos_isigma, pd.y * sepath.drift_pos_isigma, pd.z * sepath.drift_pos_isigma,
     ed.x * sepath.drift_ea_isigma, ed.y * sepath.drift_ea_isigma, ed.z * sepath.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(sepose.ea.x - sepose.tilt_roll) * sepath.tilt_isigma,
     (sepose.ea.y - sepose.tilt_pitch) * sepath.tilt_isigma]
}))]
struct SePose {
    pos: Param<vect3d>,
    ea: SimpleEulerAngleParam<f64>,
    prior_pos: vect3d,
    prior_ea: vect3d,
    gps_pos: vect3d,
    gps_cov_r: matrix3d,
    gps_cov_isigma: vect3d,
    tilt_roll: f64,
    tilt_pitch: f64,
    hb_pose: SelfBlock<SePose>,
}

#[arael::model]
#[arael(constraint(hb_drift, {
    let d = selm.pos - selm.prior_pos;
    [d.x * sepath.drift_lm_isigma, d.y * sepath.drift_lm_isigma, d.z * sepath.drift_lm_isigma]
}))]
struct SeLm {
    pos: Param<vect3d>,
    prior_pos: vect3d,
    frines: std::vec::Vec<SeFrine>,
    hb_drift: SelfBlock<SeLm>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - sefrine.camera_pos;
    let r_f = sefrine.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * sefrine.isigma.x * sepath.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * sefrine.isigma.y * sepath.frine_isigma_scale]
}))]
struct SeFrine {
    #[arael(ref = root.poses)] pose: Ref<SePose>,
    mf2r: matrix3d,
    camera_pos: vect3d,
    isigma: vect2d,
    hb: CrossBlock<SeLm, SePose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - sepair.delta_pos;
    let pos_w = sepair.pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * sepair.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = sepair.ea_cov_r.transpose() * ea_err;
    [pos_w.x * sepair.pos_cov_isigma.x, pos_w.y * sepair.pos_cov_isigma.y, pos_w.z * sepair.pos_cov_isigma.z,
     ea_w.x * sepair.ea_cov_isigma.x, ea_w.y * sepair.ea_cov_isigma.y, ea_w.z * sepair.ea_cov_isigma.z]
}))]
struct SePair {
    #[arael(ref = root.poses)] prev: Ref<SePose>,
    #[arael(ref = root.poses)] cur: Ref<SePose>,
    delta_pos: vect3d,
    delta_ea: vect3d,
    pos_cov_r: matrix3d,
    pos_cov_isigma: vect3d,
    ea_cov_r: matrix3d,
    ea_cov_isigma: vect3d,
    hb: CrossBlock<SePose, SePose>,
}

#[arael::model]
#[arael(root)]
struct SePath {
    poses: refs::Vec<SePose>,
    landmarks: refs::Vec<SeLm>,
    pose_pairs: std::vec::Vec<SePair>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    drift_lm_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

// ===========================================================================
// Euler-angle-delta variant (EulerAngleParam)
// ===========================================================================
#[arael::model]
#[arael(constraint(hb_pose, {
    let raw = eupose.pos - eupose.gps_pos;
    let rt = eupose.gps_cov_r.transpose() * raw;
    [rt.x * eupose.gps_cov_isigma.x, rt.y * eupose.gps_cov_isigma.y, rt.z * eupose.gps_cov_isigma.z]
}))]
#[arael(constraint(hb_pose, {
    let pd = eupose.pos - eupose.prior_pos;
    let ed = eupose.ea - eupose.prior_ea;
    [pd.x * eupath.drift_pos_isigma, pd.y * eupath.drift_pos_isigma, pd.z * eupath.drift_pos_isigma,
     ed.x * eupath.drift_ea_isigma, ed.y * eupath.drift_ea_isigma, ed.z * eupath.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(eupose.ea.x - eupose.tilt_roll) * eupath.tilt_isigma,
     (eupose.ea.y - eupose.tilt_pitch) * eupath.tilt_isigma]
}))]
struct EuPose {
    pos: Param<vect3d>,
    ea: EulerAngleParam<f64>,
    prior_pos: vect3d,
    prior_ea: vect3d,
    gps_pos: vect3d,
    gps_cov_r: matrix3d,
    gps_cov_isigma: vect3d,
    tilt_roll: f64,
    tilt_pitch: f64,
    hb_pose: SelfBlock<EuPose>,
}

#[arael::model]
#[arael(constraint(hb_drift, {
    let d = eulm.pos - eulm.prior_pos;
    [d.x * eupath.drift_lm_isigma, d.y * eupath.drift_lm_isigma, d.z * eupath.drift_lm_isigma]
}))]
struct EuLm {
    pos: Param<vect3d>,
    prior_pos: vect3d,
    frines: std::vec::Vec<EuFrine>,
    hb_drift: SelfBlock<EuLm>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - eufrine.camera_pos;
    let r_f = eufrine.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * eufrine.isigma.x * eupath.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * eufrine.isigma.y * eupath.frine_isigma_scale]
}))]
struct EuFrine {
    #[arael(ref = root.poses)] pose: Ref<EuPose>,
    mf2r: matrix3d,
    camera_pos: vect3d,
    isigma: vect2d,
    hb: CrossBlock<EuLm, EuPose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - eupair.delta_pos;
    let pos_w = eupair.pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * eupair.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = eupair.ea_cov_r.transpose() * ea_err;
    [pos_w.x * eupair.pos_cov_isigma.x, pos_w.y * eupair.pos_cov_isigma.y, pos_w.z * eupair.pos_cov_isigma.z,
     ea_w.x * eupair.ea_cov_isigma.x, ea_w.y * eupair.ea_cov_isigma.y, ea_w.z * eupair.ea_cov_isigma.z]
}))]
struct EuPair {
    #[arael(ref = root.poses)] prev: Ref<EuPose>,
    #[arael(ref = root.poses)] cur: Ref<EuPose>,
    delta_pos: vect3d,
    delta_ea: vect3d,
    pos_cov_r: matrix3d,
    pos_cov_isigma: vect3d,
    ea_cov_r: matrix3d,
    ea_cov_isigma: vect3d,
    hb: CrossBlock<EuPose, EuPose>,
}

#[arael::model]
#[arael(root)]
struct EuPath {
    poses: refs::Vec<EuPose>,
    landmarks: refs::Vec<EuLm>,
    pose_pairs: std::vec::Vec<EuPair>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    drift_lm_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

// ===========================================================================
// Rotation-vector-delta variant (QuaternionParam, exp map)
// ===========================================================================
#[arael::model]
#[arael(constraint(hb_pose, {
    let raw = qupose.pos - qupose.gps_pos;
    let rt = qupose.gps_cov_r.transpose() * raw;
    [rt.x * qupose.gps_cov_isigma.x, rt.y * qupose.gps_cov_isigma.y, rt.z * qupose.gps_cov_isigma.z]
}))]
#[arael(constraint(hb_pose, {
    let pd = qupose.pos - qupose.prior_pos;
    let ed = qupose.ea - qupose.prior_ea;
    [pd.x * qupath.drift_pos_isigma, pd.y * qupath.drift_pos_isigma, pd.z * qupath.drift_pos_isigma,
     ed.x * qupath.drift_ea_isigma, ed.y * qupath.drift_ea_isigma, ed.z * qupath.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(qupose.ea.x - qupose.tilt_roll) * qupath.tilt_isigma,
     (qupose.ea.y - qupose.tilt_pitch) * qupath.tilt_isigma]
}))]
struct QuPose {
    pos: Param<vect3d>,
    ea: QuaternionParam<f64>,
    prior_pos: vect3d,
    prior_ea: vect3d,
    gps_pos: vect3d,
    gps_cov_r: matrix3d,
    gps_cov_isigma: vect3d,
    tilt_roll: f64,
    tilt_pitch: f64,
    hb_pose: SelfBlock<QuPose>,
}

#[arael::model]
#[arael(constraint(hb_drift, {
    let d = qulm.pos - qulm.prior_pos;
    [d.x * qupath.drift_lm_isigma, d.y * qupath.drift_lm_isigma, d.z * qupath.drift_lm_isigma]
}))]
struct QuLm {
    pos: Param<vect3d>,
    prior_pos: vect3d,
    frines: std::vec::Vec<QuFrine>,
    hb_drift: SelfBlock<QuLm>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - qufrine.camera_pos;
    let r_f = qufrine.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * qufrine.isigma.x * qupath.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * qufrine.isigma.y * qupath.frine_isigma_scale]
}))]
struct QuFrine {
    #[arael(ref = root.poses)] pose: Ref<QuPose>,
    mf2r: matrix3d,
    camera_pos: vect3d,
    isigma: vect2d,
    hb: CrossBlock<QuLm, QuPose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - qupair.delta_pos;
    let pos_w = qupair.pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * qupair.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = qupair.ea_cov_r.transpose() * ea_err;
    [pos_w.x * qupair.pos_cov_isigma.x, pos_w.y * qupair.pos_cov_isigma.y, pos_w.z * qupair.pos_cov_isigma.z,
     ea_w.x * qupair.ea_cov_isigma.x, ea_w.y * qupair.ea_cov_isigma.y, ea_w.z * qupair.ea_cov_isigma.z]
}))]
struct QuPair {
    #[arael(ref = root.poses)] prev: Ref<QuPose>,
    #[arael(ref = root.poses)] cur: Ref<QuPose>,
    delta_pos: vect3d,
    delta_ea: vect3d,
    pos_cov_r: matrix3d,
    pos_cov_isigma: vect3d,
    ea_cov_r: matrix3d,
    ea_cov_isigma: vect3d,
    hb: CrossBlock<QuPose, QuPose>,
}

#[arael::model]
#[arael(root)]
struct QuPath {
    poses: refs::Vec<QuPose>,
    landmarks: refs::Vec<QuLm>,
    pose_pairs: std::vec::Vec<QuPair>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    drift_lm_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

fn build_se(scene: &Scene) -> (SePath, Vec<f64>) {
    let mut path = SePath {
        poses: refs::Vec::new(), landmarks: refs::Vec::new(), pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma as f64, drift_ea_isigma: scene.drift_ea_isigma as f64,
        drift_lm_isigma: scene.drift_lm_isigma as f64, tilt_isigma: scene.tilt_isigma as f64,
        frine_isigma_scale: scene.frine_isigma_scale as f64,
    };
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        path.poses.push(SePose {
            pos: Param::new(vect3d::from(p.init_pos)), ea: SimpleEulerAngleParam::new(vect3d::from(p.init_ea)),
            prior_pos: vect3d::from(p.init_pos), prior_ea: vect3d::from(p.init_ea),
            gps_pos: vect3d::from(g.pos), gps_cov_r: matrix3d::from(g.cov_r), gps_cov_isigma: vect3d::from(g.cov_isigma),
            tilt_roll: p.tilt_roll as f64, tilt_pitch: p.tilt_pitch as f64, hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<SeFrine>> = (0..scene.landmarks_init.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        per_lm[f.landmark as usize].push(SeFrine {
            pose: path.poses.ref_at(f.pose), mf2r: matrix3d::from(f.mf2r), camera_pos: vect3d::from(f.camera_pos),
            isigma: vect2d::new(f.isigma.x as f64, f.isigma.y as f64), hb: CrossBlock::new(),
        });
    }
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        path.landmarks.push(SeLm {
            pos: Param::new(vect3d::from(*init)), prior_pos: vect3d::from(*init),
            frines: std::mem::take(&mut per_lm[i]), hb_drift: SelfBlock::new(),
        });
    }
    for o in &scene.odo {
        path.pose_pairs.push(SePair {
            prev: path.poses.ref_at(o.prev), cur: path.poses.ref_at(o.cur),
            delta_pos: vect3d::from(o.delta_pos), delta_ea: vect3d::from(o.delta_ea),
            pos_cov_r: matrix3d::from(o.pos_cov_r), pos_cov_isigma: vect3d::from(o.pos_cov_isigma),
            ea_cov_r: matrix3d::from(o.ea_cov_r), ea_cov_isigma: vect3d::from(o.ea_cov_isigma), hb: CrossBlock::new(),
        });
    }
    let mut params = Vec::new();
    path.serialize64(&mut params);
    (path, params)
}

fn build_eu(scene: &Scene) -> (EuPath, Vec<f64>) {
    let mut path = EuPath {
        poses: refs::Vec::new(), landmarks: refs::Vec::new(), pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma as f64, drift_ea_isigma: scene.drift_ea_isigma as f64,
        drift_lm_isigma: scene.drift_lm_isigma as f64, tilt_isigma: scene.tilt_isigma as f64,
        frine_isigma_scale: scene.frine_isigma_scale as f64,
    };
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        path.poses.push(EuPose {
            pos: Param::new(vect3d::from(p.init_pos)), ea: EulerAngleParam::new(vect3d::from(p.init_ea)),
            prior_pos: vect3d::from(p.init_pos), prior_ea: vect3d::from(p.init_ea),
            gps_pos: vect3d::from(g.pos), gps_cov_r: matrix3d::from(g.cov_r), gps_cov_isigma: vect3d::from(g.cov_isigma),
            tilt_roll: p.tilt_roll as f64, tilt_pitch: p.tilt_pitch as f64, hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<EuFrine>> = (0..scene.landmarks_init.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        per_lm[f.landmark as usize].push(EuFrine {
            pose: path.poses.ref_at(f.pose), mf2r: matrix3d::from(f.mf2r), camera_pos: vect3d::from(f.camera_pos),
            isigma: vect2d::new(f.isigma.x as f64, f.isigma.y as f64), hb: CrossBlock::new(),
        });
    }
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        path.landmarks.push(EuLm {
            pos: Param::new(vect3d::from(*init)), prior_pos: vect3d::from(*init),
            frines: std::mem::take(&mut per_lm[i]), hb_drift: SelfBlock::new(),
        });
    }
    for o in &scene.odo {
        path.pose_pairs.push(EuPair {
            prev: path.poses.ref_at(o.prev), cur: path.poses.ref_at(o.cur),
            delta_pos: vect3d::from(o.delta_pos), delta_ea: vect3d::from(o.delta_ea),
            pos_cov_r: matrix3d::from(o.pos_cov_r), pos_cov_isigma: vect3d::from(o.pos_cov_isigma),
            ea_cov_r: matrix3d::from(o.ea_cov_r), ea_cov_isigma: vect3d::from(o.ea_cov_isigma), hb: CrossBlock::new(),
        });
    }
    let mut params = Vec::new();
    path.serialize64(&mut params);
    (path, params)
}

fn build_qu(scene: &Scene) -> (QuPath, Vec<f64>) {
    let mut path = QuPath {
        poses: refs::Vec::new(), landmarks: refs::Vec::new(), pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma as f64, drift_ea_isigma: scene.drift_ea_isigma as f64,
        drift_lm_isigma: scene.drift_lm_isigma as f64, tilt_isigma: scene.tilt_isigma as f64,
        frine_isigma_scale: scene.frine_isigma_scale as f64,
    };
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        path.poses.push(QuPose {
            pos: Param::new(vect3d::from(p.init_pos)), ea: QuaternionParam::from_euler_angles(vect3d::from(p.init_ea)),
            prior_pos: vect3d::from(p.init_pos), prior_ea: vect3d::from(p.init_ea),
            gps_pos: vect3d::from(g.pos), gps_cov_r: matrix3d::from(g.cov_r), gps_cov_isigma: vect3d::from(g.cov_isigma),
            tilt_roll: p.tilt_roll as f64, tilt_pitch: p.tilt_pitch as f64, hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<QuFrine>> = (0..scene.landmarks_init.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        per_lm[f.landmark as usize].push(QuFrine {
            pose: path.poses.ref_at(f.pose), mf2r: matrix3d::from(f.mf2r), camera_pos: vect3d::from(f.camera_pos),
            isigma: vect2d::new(f.isigma.x as f64, f.isigma.y as f64), hb: CrossBlock::new(),
        });
    }
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        path.landmarks.push(QuLm {
            pos: Param::new(vect3d::from(*init)), prior_pos: vect3d::from(*init),
            frines: std::mem::take(&mut per_lm[i]), hb_drift: SelfBlock::new(),
        });
    }
    for o in &scene.odo {
        path.pose_pairs.push(QuPair {
            prev: path.poses.ref_at(o.prev), cur: path.poses.ref_at(o.cur),
            delta_pos: vect3d::from(o.delta_pos), delta_ea: vect3d::from(o.delta_ea),
            pos_cov_r: matrix3d::from(o.pos_cov_r), pos_cov_isigma: vect3d::from(o.pos_cov_isigma),
            ea_cov_r: matrix3d::from(o.ea_cov_r), ea_cov_isigma: vect3d::from(o.ea_cov_isigma), hb: CrossBlock::new(),
        });
    }
    let mut params = Vec::new();
    path.serialize64(&mut params);
    (path, params)
}

fn solve_and_time<P: LmProblem<f64>>(path: &mut P, params: &[f64]) -> (usize, usize, f64, LmTiming) {
    let result = simple_lm::solve_sparse(params, path, &cfg()).unwrap();
    (result.iterations, result.accepted_iterations, result.end_cost,
     result.timing.expect("gather_timing is on"))
}

// Accumulates one variant's results across the timed rounds. Iteration count,
// accepted-step count, and final cost are deterministic per round, so we keep
// the last; every round's timing is kept for the median.
struct Variant {
    name: &'static str,
    iters: usize,
    accepted: usize,
    cost: f64,
    timings: Vec<LmTiming>,
}

impl Variant {
    fn new(name: &'static str) -> Self {
        Variant { name, iters: 0, accepted: 0, cost: 0.0, timings: Vec::new() }
    }
    fn record(&mut self, r: (usize, usize, f64, LmTiming)) {
        self.iters = r.0; self.accepted = r.1; self.cost = r.2;
        self.timings.push(r.3);
    }
    // Median across rounds of one phase total, in milliseconds.
    fn med_ms(&self, f: impl Fn(&LmTiming) -> Duration) -> f64 {
        let mut v: Vec<f64> = self.timings.iter().map(|t| f(t).as_secs_f64() * 1e3).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    }
    // Median across rounds of a per-iteration mean (steady state, first
    // iteration excluded), in ms. Rounds with no steady state are skipped.
    fn med_mean_ms(&self, f: impl Fn(&LmTiming) -> Option<Duration>) -> f64 {
        let mut v: Vec<f64> = self.timings.iter()
            .filter_map(|t| f(t).map(|d| d.as_secs_f64() * 1e3))
            .collect();
        if v.is_empty() { return 0.0; }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    }
    fn last(&self) -> &LmTiming { self.timings.last().unwrap() }
}

fn main() {
    let mut cfg = scene::SceneConfig::default();
    cfg.num_poses = std::env::var("POSES").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    cfg.num_landmarks = 4 * cfg.num_poses; // scale landmarks with poses, as main.rs does
    let scene = scene::generate(&cfg);
    println!("scene: {} poses, {} landmarks, {} bearings, {} odometry",
        scene.poses.len(), scene.landmarks_init.len(), scene.frines.len(), scene.odo.len());

    let rounds = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    // Warm up all three (page-in, branch predictor, caches).
    { let (mut p, pp) = build_se(&scene); solve_and_time(&mut p, &pp); }
    { let (mut p, pp) = build_eu(&scene); solve_and_time(&mut p, &pp); }
    { let (mut p, pp) = build_qu(&scene); solve_and_time(&mut p, &pp); }

    let mut se = Variant::new("SimpleEulerAngleParam (naive)");
    let mut eu = Variant::new("EulerAngleParam (euler)");
    let mut qu = Variant::new("QuaternionParam (rotvec)");
    for _ in 0..rounds {
        { let (mut p, pp) = build_se(&scene); se.record(solve_and_time(&mut p, &pp)); }
        { let (mut p, pp) = build_eu(&scene); eu.record(solve_and_time(&mut p, &pp)); }
        { let (mut p, pp) = build_qu(&scene); qu.record(solve_and_time(&mut p, &pp)); }
    }
    let vs = [&se, &eu, &qu];

    println!("\nconvergence");
    println!("{:<30} {:>6} {:>9} {:>16} {:>10}", "algorithm", "iters", "accepted", "final cost", "total ms");
    for v in vs {
        println!("{:<30} {:>6} {:>9} {:>16.6} {:>10.3}",
            v.name, v.iters, v.accepted, v.cost, v.med_ms(|t| t.total));
    }

    println!("\nper-phase median ms  (first_* also shown inside their phase total: assembly includes first_asm, lin_solve includes first_slv)");
    println!("{:<30} {:>9} {:>9} {:>10} {:>10} {:>9} {:>8}",
        "algorithm", "assembly", "first_asm", "lin_solve", "first_slv", "cost_eval", "advance");
    for v in vs {
        println!("{:<30} {:>9.3} {:>9.3} {:>10.3} {:>10.3} {:>9.3} {:>8.3}",
            v.name,
            v.med_ms(|t| t.assembly), v.med_ms(|t| t.first_assembly),
            v.med_ms(|t| t.linear_solve), v.med_ms(|t| t.first_linear_solve),
            v.med_ms(|t| t.cost_eval), v.med_ms(|t| t.advance));
    }

    println!("\nper-iteration mean ms  (steady state: first iteration excluded)");
    println!("{:<30} {:>9} {:>10} {:>9} {:>8}",
        "algorithm", "assembly", "lin_solve", "cost_eval", "advance");
    for v in vs {
        println!("{:<30} {:>9.3} {:>10.3} {:>9.3} {:>8.3}",
            v.name,
            v.med_mean_ms(|t| t.mean_assembly()),
            v.med_mean_ms(|t| t.mean_linear_solve()),
            v.med_mean_ms(|t| t.mean_cost_eval()),
            v.med_mean_ms(|t| t.mean_advance()));
    }

    println!("\nphase call counts  (assembly / linear_solve / cost_eval / advance)");
    for v in vs {
        let t = v.last();
        println!("{:<30} {} / {} / {} / {}",
            v.name, t.assembly_count, t.linear_solve_count, t.cost_eval_count, t.advance_count);
    }
}
