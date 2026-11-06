//! The plane scene's whole-system block supernodal factorization and solve
//! on a GPU against the CPU's (`bench_harness::gpu_factor`, whose
//! documentation lists the knobs): the Hessian at the initial poses and
//! planes.
//!
//!     PLANE_POSES=900 ROUNDS=20 cargo run -r --features gpu-cuda --bin gpu_factor_bench

#[path = "../scene.rs"]
mod scene;
#[path = "../arael_runner.rs"]
mod arael_runner;

use bench_harness::arael::Model as _;

fn main() {
    let sc = scene::make_scene();
    let (h, grad, _) = bench_harness::arael::hessian::<u32, _>(&mut arael_runner::World::build(&sc.raw));
    bench_harness::gpu_factor::compare(&format!("H of plane, {} poses", sc.raw.poses.len()), &h, &grad);
}
