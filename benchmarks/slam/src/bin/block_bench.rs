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

use arael::simple_lm::{block_partition_from_spans, csc_from_cells, CooMatrix, LmProblem, RootProblem};
use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};
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
        min_ms(rounds, || coo.to_csc_with_map().expect("valid COO"));

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

    // -- two-scan route: no COO at all ---------------------------------
    // scan 1: block cells from indices alone (structure-only)
    let (t_cells, cells) = min_ms(rounds, || {
        let mut cells: Vec<(u32, u32)> = Vec::new();
        arael::model::Model::collect_hessian_cells64(&path, &mut cells);
        cells
    });
    // symbolic from the cells (anchors are ordinary scalar coords)
    let (t_sym2, sym2) = min_ms(rounds, || {
        SymbolicSparseBlockColMat::from_scalar_coords(
            partition.clone(),
            partition.clone(),
            cells.len(),
            |k| (cells[k].0 as usize, cells[k].1 as usize),
        ).0
    });
    // scan 2: position map by replaying the emission order
    let (t_pos2, positions2) = min_ms(rounds, || {
        let mut resolver = PositionResolver::new(&sym2);
        let mut out: Vec<usize> = Vec::with_capacity(positions_block.len());
        arael::model::Model::accumulate_hessian_positions64(
            &path,
            &mut |i, j| resolver.resolve(i as usize, j as usize),
            &mut out,
        );
        out
    });
    // both routes must produce the identical structure and map
    assert_eq!(sym2.parts().2, sym.parts().2);
    assert_eq!(sym2.parts().3, sym.parts().3);
    assert_eq!(sym2.parts().4, sym.parts().4);
    assert_eq!(positions2, positions_block);

    // scalar two-scan (SparseFaer's new fast path): tile-expanded CSC
    let (t_scsc, (csc_fast, _)) = min_ms(rounds, || csc_from_cells::<f64>(&partition, &cells));
    let (_, resolver_proto) = csc_from_cells::<f64>(&partition, &cells);
    let (t_spos, _spos) = min_ms(rounds, || {
        let mut resolver = resolver_proto.clone();
        let mut out: Vec<usize> = Vec::with_capacity(positions_scalar.len());
        arael::model::Model::accumulate_hessian_positions64(
            &path,
            &mut |i, j| resolver.resolve(i, j),
            &mut out,
        );
        out
    });
    let scalar_fast_nnz = csc_fast.vals.len();

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
    let mut csc_direct = coo.to_csc().expect("valid COO");
    let (t_fill_direct, _) = min_ms(rounds, || {
        path.calc_grad_hessian_sparse_direct(&params, &mut grad, &mut csc_direct)
    });

    println!("scene: {} poses, {} params; COO contributions {}, scalar nnz {}, blocks {} ({} block vals)",
        cfg.num_poses, n, coo.nnz(), csc.vals.len(), nblocks, block_vals);
    println!("rounds: {} (min reported)", rounds);
    println!();
    println!("Every first-iteration total contains exactly ONE full numeric");
    println!("computation (residuals + Jacobians + block accumulation); the");
    println!("line carrying it is marked [compute].");
    println!();
    println!("scalar CSC (today's pipeline)                    ms");
    println!("  COO pass (compute + push triplets) [compute] {:7.3}", t_coo);
    println!("  to_csc_with_map (pattern + positions + vals) {:7.3}", t_scalar_map);
    println!("  first iteration total                        {:7.3}", t_coo + t_scalar_map);
    println!();
    println!("block CSC via COO                                ms");
    println!("  COO pass (compute + push triplets) [compute] {:7.3}", t_coo);
    println!("  partition + from_scalar_coords               {:7.3}", t_spans + t_block_map);
    println!("  zeroed alloc + value scatter                 {:7.3}", t_alloc + t_scatter);
    println!("  first iteration total                        {:7.3}", t_coo + t_spans + t_block_map + t_alloc + t_scatter);
    println!();
    println!("block CSC two-scan (no COO)                      ms");
    println!("  cells scan (structure only)                  {:7.3}", t_cells);
    println!("  symbolic + partition + alloc                 {:7.3}", t_sym2 + t_spans + t_alloc);
    println!("  position map (emission replay)               {:7.3}", t_pos2);
    println!("  first indexed fill                 [compute] {:7.3}", t_fill_block);
    println!("  first iteration total                        {:7.3}", t_spans + t_cells + t_sym2 + t_pos2 + t_alloc + t_fill_block);
    println!();
    println!("scalar CSC two-scan (SparseFaer fast path; nnz {} = +{:.1}% padding)", scalar_fast_nnz,
        100.0 * (scalar_fast_nnz as f64 / csc.vals.len() as f64 - 1.0));
    println!("  cells scan (structure only)                  {:7.3}", t_cells);
    println!("  tile-expanded CSC (csc_from_cells)           {:7.3}", t_scsc);
    println!("  position map (emission replay)               {:7.3}", t_spos);
    println!("  first indexed fill                 [compute] {:7.3}", t_fill_scalar);
    println!("  first iteration total                        {:7.3}", t_cells + t_scsc + t_spos + t_fill_scalar);
    println!();
    println!("steady iteration (each is one [compute] + scatter)");
    println!("  scalar CSC direct (search per write)         {:7.3}", t_fill_direct);
    println!("  scalar CSC indexed                           {:7.3}", t_fill_scalar);
    println!("  block CSC indexed                            {:7.3}", t_fill_block);
}
