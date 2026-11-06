//! A pose graph's whole-system block supernodal factorization and solve on
//! a GPU against the CPU's (`bench_harness::gpu_factor`, whose
//! documentation lists the knobs): the Hessian at the dataset's initial
//! poses, thousands of tiny supernodes.
//!
//!     PGO_DATASET=M3500 ROUNDS=20 cargo run -r --features gpu-cuda --bin gpu_factor_bench
//!
//! PGO_DATASET: M3500 (default), city10000 (2D), sphere2500, parking-garage (3D).

#[path = "../g2o.rs"]
mod g2o;
#[path = "../g2o3.rs"]
mod g2o3;
#[path = "../arael_runner.rs"]
mod arael_runner;
#[path = "../arael_runner3.rs"]
mod arael_runner3;

use bench_harness::arael::Model as _;

fn main() {
    let name = std::env::var("PGO_DATASET").unwrap_or_else(|_| "M3500".to_string());
    let (h, grad, _) = match name.as_str() {
        "M3500" => bench_harness::arael::hessian::<u32, _>(&mut arael_runner::Graph::build(&g2o::load("datasets/input_M3500_g2o.g2o", false))),
        "city10000" => bench_harness::arael::hessian::<u32, _>(&mut arael_runner::Graph::build(&g2o::load("datasets/city10000.g2o", false))),
        "sphere2500" => bench_harness::arael::hessian::<u32, _>(&mut arael_runner3::Graph3::build(&g2o3::load3("datasets/sphere2500.g2o"))),
        "parking-garage" => bench_harness::arael::hessian::<u32, _>(&mut arael_runner3::Graph3::build(&g2o3::load3("datasets/parking-garage.g2o"))),
        other => panic!("PGO_DATASET={other}: M3500, city10000, sphere2500 or parking-garage"),
    };
    bench_harness::gpu_factor::compare(&format!("H of {name}"), &h, &grad);
}
