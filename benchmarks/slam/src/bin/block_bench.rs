//! Block-CSC assembly cost on the slam problem: what the block pipeline
//! pays on the first iteration (structure + position map) and on every
//! later iteration (indexed refill), side by side with the scalar CSC
//! pipeline it replaces. Single core, min over ROUNDS (default 20).
//!
//!     SLAM_POSES=300 ROUNDS=20 cargo run -r --bin block_bench

#[path = "../scene.rs"]
mod scene;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::{block_partition_from_spans, CooMatrix, LmProblem, RootProblem};
use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use scene::SceneConfig;
use std::time::Instant;

fn pin_single_core() {
    for var in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"] {
        std::env::set_var(var, "1");
    }
    unsafe {
        let core = std::thread::available_parallelism().map(|n| n.get() - 1).unwrap_or(0);
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn min_ms<R>(rounds: usize, mut f: impl FnMut() -> R) -> (f64, R) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..rounds {
        let t0 = Instant::now();
        let r = f();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        out = Some(r);
    }
    (best, out.unwrap())
}

fn main() {
    pin_single_core();
    let mut cfg = SceneConfig::default();
    if let Ok(n) = std::env::var("SLAM_POSES") {
        if let Ok(n) = n.parse() {
            cfg.num_poses = n; cfg.num_landmarks = 4 * n;
        }
    }
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let scene = scene::generate(&cfg);

    let mut path = arael_runner::build(&scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    let n = params.len();

    // -- first-iteration pieces --------------------------------------

    // COO pattern discovery (shared by both pipelines)
    let mut grad = vec![0.0; n];
    let (t_coo, coo) = min_ms(rounds, || {
        let mut coo = CooMatrix::new(n);
        path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        coo
    });

    // scalar: CSC + position map
    let (t_scalar_map, (csc, positions_scalar)) =
        min_ms(rounds, || coo.to_csc_with_map());

    // block: entity partition, then structure + position map
    let (t_spans, partition) = min_ms(rounds, || {
        let spans = RootProblem::param_block_spans(&path);
        block_partition_from_spans(&spans, n)
    });
    let (t_block_map, (sym, positions_block)) = min_ms(rounds, || {
        SymbolicSparseBlockColMat::from_scalar_coords(
            partition.clone(),
            partition.clone(),
            coo.nnz(),
            |k| (coo.rows[k] as usize, coo.cols[k] as usize),
        )
    });
    let nblocks = sym.nblocks();
    let block_vals = sym.val_count();
    let (t_alloc, mut bsc) =
        min_ms(rounds, || SparseBlockColMat::<usize, f64>::zeroed(sym.clone()));

    // landing the values: the COO pass already computed every block's
    // contents, so the first fill is a scatter-only replay (zero + the
    // Model-level indexed accumulate, no recompute)
    let (t_scatter, _) = min_ms(rounds, || {
        bsc.vals_mut().iter_mut().for_each(|v| *v = 0.0);
        let mut cursor = 0usize;
        arael::model::Model::accumulate_hessian_sparse_indexed64(
            &path, bsc.vals_mut(), &positions_block, &mut cursor);
    });

    // -- steady-state refill ------------------------------------------

    let mut vals_s = vec![0.0; csc.vals.len()];
    let (t_fill_scalar, _) = min_ms(rounds, || {
        path.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut vals_s, &positions_scalar)
    });
    let (t_fill_block, _) = min_ms(rounds, || {
        path.calc_grad_hessian_sparse_indexed(&params, &mut grad, bsc.vals_mut(), &positions_block)
    });
    // second reference: direct CSC accumulation (binary search per
    // write, no position map) -- what indexing buys
    let mut csc_direct = coo.to_csc();
    let (t_fill_direct, _) = min_ms(rounds, || {
        path.calc_grad_hessian_sparse_direct(&params, &mut grad, &mut csc_direct)
    });

    println!("scene: {} poses, {} params; COO contributions {}, scalar nnz {}, blocks {} ({} block vals)",
        cfg.num_poses, n, coo.nnz(), csc.vals.len(), nblocks, block_vals);
    println!("rounds: {} (min reported)", rounds);
    println!();
    println!("first iteration                     ms");
    println!("  COO discovery (shared)        {:8.3}", t_coo);
    println!("  scalar: to_csc_with_map       {:8.3}", t_scalar_map);
    println!("  block:  param spans+partition {:8.3}", t_spans);
    println!("  block:  from_scalar_coords    {:8.3}", t_block_map);
    println!("  block:  zeroed alloc          {:8.3}", t_alloc);
    println!("  block:  value scatter         {:8.3}", t_scatter);
    println!("  first-iter total scalar       {:8.3}", t_coo + t_scalar_map);
    println!("  first-iter total block        {:8.3}", t_coo + t_spans + t_block_map + t_alloc + t_scatter);
    println!();
    println!("steady iteration                    ms");
    println!("  scalar CSC direct (search)    {:8.3}", t_fill_direct);
    println!("  scalar CSC indexed            {:8.3}", t_fill_scalar);
    println!("  block CSC indexed             {:8.3}", t_fill_block);
}
