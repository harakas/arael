//! The reduced system's block supernodal factorization and solve on a GPU
//! against the CPU's, on the slam problem's S
//! (`bench_harness::gpu_factor`, whose documentation lists the knobs).
//!
//!     SLAM_POSES=1000 ROUNDS=20 cargo run -r --features gpu-cuda --bin gpu_factor_bench
//!     ARAEL_GPU=hip SLAM_POSES=300 cargo run -r --features gpu-hip --bin gpu_factor_bench

#[path = "../scene.rs"]
mod scene;
#[path = "../arael_runner.rs"]
mod arael_runner;
#[path = "../reduced.rs"]
mod reduced;

use arael_faer::SparseIndex;
use scene::SceneConfig;

fn main() {
    let mut cfg = SceneConfig::default();
    if let Some(n) = std::env::var("SLAM_POSES").ok().and_then(|v| v.parse().ok()) {
        cfg.num_poses = n;
        cfg.num_landmarks = 4 * n;
    }
    let r = reduced::reduced_system::<SparseIndex>(&cfg);
    bench_harness::gpu_factor::compare(&format!("S of slam {}", cfg.num_poses), &r.s, &r.rhs);
}
