//! Shared 2D multi-run scene + measurements, used by BOTH slam2d_multi_demo
//! (joint solve) and slam2d_align_demo (decoupled solve) so the two run on the
//! IDENTICAL problem and their results can be compared.
//!
//! `generate()` reproduces slam2d_multi_demo's original RNG order exactly, so
//! moving that demo onto this module leaves its output byte-identical.

use arael::vect::vect2f;
use arael::matrix::matrix2f;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::f32::consts::PI;

pub struct Cfg {
    pub n_runs: usize,
    pub n_poses: usize,
    pub n_landmarks: usize,
    pub seed: u64,
    pub step: f32,
    pub turn: f32,
    pub fov_half: f32,
    pub range_min: f32,
    pub range_max: f32,
    pub odo_pos_sigma: f32,
    pub odo_gamma_sigma: f32,
    pub bearing_sigma: f32,
    pub init_range: f32,
    pub gps_noise_sigma: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            n_runs: 3,
            n_poses: 20,
            n_landmarks: 30,
            seed: 7,
            step: 1.5,
            turn: 0.10,
            fov_half: 70.0_f32.to_radians(),
            range_min: 4.0,
            range_max: 55.0,
            odo_pos_sigma: 0.05,
            odo_gamma_sigma: 0.4_f32.to_radians(),
            bearing_sigma: 1.0_f32.to_radians(),
            init_range: 20.0,
            gps_noise_sigma: 0.4,
        }
    }
}

// One run's ground-truth trajectory: a gentle left arc from near the origin,
// perturbed per run (start offset, heading bias, slightly different turn rate).
pub fn truth_poses(cfg: &Cfg, run: usize) -> std::vec::Vec<(vect2f, f32)> {
    let mut out = std::vec::Vec::with_capacity(cfg.n_poses);
    let off = (run as f32) - (cfg.n_runs as f32 - 1.0) * 0.5; // -1, 0, 1
    let mut pos = vect2f::new(0.0, 1.2 * off);
    let mut gamma = 0.04 * off;
    let turn = cfg.turn * (1.0 + 0.15 * off);
    out.push((pos, gamma));
    for _ in 1..cfg.n_poses {
        pos = pos + vect2f::new(cfg.step * gamma.cos(), cfg.step * gamma.sin());
        gamma += turn;
        out.push((pos, gamma));
    }
    out
}

pub fn truth_landmarks(cfg: &Cfg, rng: &mut StdRng, all_poses: &[(vect2f, f32)]) -> std::vec::Vec<vect2f> {
    let mut out = std::vec::Vec::with_capacity(cfg.n_landmarks);
    while out.len() < cfg.n_landmarks {
        let anchor = all_poses[rng.random_range(0..all_poses.len())].0;
        let theta = rng.random::<f32>() * 2.0 * PI;
        let r = cfg.range_min + rng.random::<f32>() * (cfg.range_max - cfg.range_min);
        let lm = anchor + vect2f::new(r * theta.cos(), r * theta.sin());
        let visible = all_poses.iter().any(|(p, _)| {
            let d = (lm - *p).norm();
            d >= cfg.range_min && d <= cfg.range_max
        });
        if visible { out.push(lm); }
    }
    out
}

pub fn observe(cfg: &Cfg, pos: vect2f, gamma: f32, lm: vect2f) -> Option<f32> {
    let d = lm - pos;
    let dist = d.norm();
    if dist < cfg.range_min || dist > cfg.range_max { return None; }
    let local = matrix2f::rotation(gamma).transpose() * d;
    let bearing = local.y.atan2(local.x);
    if bearing.abs() > cfg.fov_half { return None; }
    Some(bearing)
}

// One run's dead-reckoned poses + measured odometry deltas + per-pose GPS fixes.
pub struct RunMeas {
    pub est: std::vec::Vec<(vect2f, f32)>,   // dead-reckoned (pos, gamma) per pose
    pub delta: std::vec::Vec<(vect2f, f32)>, // measured (delta_pos, delta_gamma); [0] = (0,0)
    pub gps: std::vec::Vec<vect2f>,          // GPS position per pose
}

// One bearing sighting (any landmark, any run/pose that saw it).
pub struct Sighting {
    pub gt_id: usize,
    pub run: usize,
    pub pose: usize,
    pub bearing: f32,
}

pub struct Scene {
    pub gt_poses: std::vec::Vec<std::vec::Vec<(vect2f, f32)>>,
    pub gt_lms: std::vec::Vec<vect2f>,
    pub runs: std::vec::Vec<RunMeas>,
    pub sightings: std::vec::Vec<Sighting>,
    pub gps_isigma: f32,
}

// Generate the whole problem deterministically. RNG order matches
// slam2d_multi_demo's original build_map: truth_landmarks, then per-run
// (first GPS, then per pose odometry + GPS), then the bearing sighting loop.
pub fn generate(cfg: &Cfg) -> Scene {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let nd = Normal::new(0.0_f32, 1.0).unwrap();

    let gt_poses: std::vec::Vec<_> = (0..cfg.n_runs).map(|r| truth_poses(cfg, r)).collect();
    let all_poses: std::vec::Vec<(vect2f, f32)> = gt_poses.iter().flatten().copied().collect();
    let gt_lms = truth_landmarks(cfg, &mut rng, &all_poses);

    let gps_isigma = 1.0 / cfg.gps_noise_sigma;
    let mut runs: std::vec::Vec<RunMeas> = std::vec::Vec::new();
    for gp in &gt_poses {
        let mut est = std::vec::Vec::with_capacity(cfg.n_poses);
        let mut delta = std::vec::Vec::with_capacity(cfg.n_poses);
        let mut gps = std::vec::Vec::with_capacity(cfg.n_poses);
        let mut est_pos = gp[0].0;
        let mut est_gamma = gp[0].1;
        // First pose: its own GPS fix (nothing held fixed).
        gps.push(gp[0].0 + vect2f::new(cfg.gps_noise_sigma * rng.sample(nd),
                                       cfg.gps_noise_sigma * rng.sample(nd)));
        est.push((est_pos, est_gamma));
        delta.push((vect2f::new(0.0, 0.0), 0.0));
        for pi in 1..gp.len() {
            let (prev_p, prev_g) = gp[pi - 1];
            let (gt_p, gt_g) = gp[pi];
            let true_delta = matrix2f::rotation(prev_g).transpose() * (gt_p - prev_p);
            let noisy_delta = true_delta + vect2f::new(
                cfg.odo_pos_sigma * rng.sample(nd), cfg.odo_pos_sigma * rng.sample(nd));
            let noisy_dg = (gt_g - prev_g) + cfg.odo_gamma_sigma * rng.sample(nd);
            est_pos = est_pos + matrix2f::rotation(est_gamma) * noisy_delta;
            est_gamma += noisy_dg;
            let g = gt_p + vect2f::new(cfg.gps_noise_sigma * rng.sample(nd),
                                       cfg.gps_noise_sigma * rng.sample(nd));
            est.push((est_pos, est_gamma));
            delta.push((noisy_delta, noisy_dg));
            gps.push(g);
        }
        runs.push(RunMeas { est, delta, gps });
    }

    let mut sightings = std::vec::Vec::new();
    for (gt_li, &gt_lm) in gt_lms.iter().enumerate() {
        for (r, gp) in gt_poses.iter().enumerate() {
            for (pi, &(p, g)) in gp.iter().enumerate() {
                if let Some(true_b) = observe(cfg, p, g, gt_lm) {
                    let bearing = true_b + cfg.bearing_sigma * rng.sample(&nd);
                    sightings.push(Sighting { gt_id: gt_li, run: r, pose: pi, bearing });
                }
            }
        }
    }

    Scene { gt_poses, gt_lms, runs, sightings, gps_isigma }
}
