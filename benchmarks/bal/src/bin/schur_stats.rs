//! What the Schur reduction faces on each BAL dataset: the size and
//! density of the reduced camera system S, the observer-pair count that
//! drives the reduction, and the cost split between forming S and
//! factorizing it. Explains where the explicit-Schur route wins and
//! where it stops winning.
//!
//!     cargo run -r --bin schur_stats

#[path = "../bal.rs"]
mod bal;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::{block_partition_from_spans, LmProblem, RootProblem};
use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};
use arael_faer::faer;
use arael_faer::faer::dyn_stack::MemStack;
use arael_faer::faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
};
use arael_faer::schur::{schur_reduce, schur_symbolic, SchurContext};
use std::time::Instant;

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
    let datasets = [
        ("Ladybug-49", "datasets/problem-49-7776-pre.txt"),
        ("Ladybug-138", "datasets/problem-138-19878-pre.txt"),
        ("Ladybug-372", "datasets/problem-372-47423-pre.txt"),
        ("Ladybug-1723", "datasets/problem-1723-156502-pre.txt"),
    ];
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    println!("{:<14} {:>7} {:>8} {:>9} {:>7} {:>8} {:>10} {:>9} {:>9}",
        "dataset", "cams", "S n", "S dens%", "pairs", "L MB", "reduce ms", "sym ms", "numer ms");
    for (name, path) in datasets {
        if !std::path::Path::new(path).exists() {
            println!("{:<14} (missing)", name);
            continue;
        }
        let ds = bal::load(path);
        let mut scene = arael_runner::build_f64(&ds);
        let mut params: Vec<f64> = Vec::new();
        scene.serialize64(&mut params);
        let n = params.len();

        let mut spans = Vec::new();
        scene.collect_param_block_spans(&mut spans);
        let partition = block_partition_from_spans(&spans, n);
        let mut cells: Vec<(u32, u32)> = Vec::new();
        arael::model::Model::collect_hessian_cells64(&scene, &mut cells);
        let (hsym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            partition.clone(),
            partition.clone(),
            cells.len(),
            |k| (cells[k].0 as usize, cells[k].1 as usize),
        );
        let mut resolver = PositionResolver::new(&hsym);
        let mut positions: Vec<usize> = Vec::new();
        arael::model::Model::accumulate_hessian_positions64(
            &scene,
            &mut |i, j| resolver.resolve(i as usize, j as usize),
            &mut positions,
        );
        let mut h = SparseBlockColMat::<usize, f64>::zeroed(hsym);
        let mut grad = vec![0.0; n];
        scene.calc_grad_hessian_sparse_indexed(&params, &mut grad, h.vals_mut(), &positions);

        // LM damping, as a real solve applies before reducing: the BAL
        // Gauss-Newton Hessian is singular at the initial estimate
        // (points behind cameras, unconstrained gauge).
        let nblk = partition.len() - 1;
        let lambda: f64 = std::env::var("BAL_LAMBDA0")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-4);
        for b in 0..nblk {
            let w = partition[b + 1] - partition[b];
            let hs = h.symbolic().clone();
            let d = hs.col_range(b).find(|&x| hs.blk_row(x) == b).unwrap();
            let base = hs.val_range(d).start;
            for k in 0..w {
                h.vals_mut()[base + k * (w + 1)] *= 1.0 + lambda;
            }
        }

        let hint = RootProblem::elimination_hint(&scene);
        let eliminated: Vec<usize> = (0..nblk)
            .filter(|&b| hint.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end))
            .collect();
        let sym = schur_symbolic(h.symbolic(), &eliminated).unwrap();
        let mut s = sym.alloc_s::<f64>();
        let mut ctx = SchurContext::new();
        let nk = sym.s.nrows();
        let mut rhs = vec![0.0; nk];
        let (t_reduce, _) = min_ms(rounds, || {
            schur_reduce(&sym, &h, &grad, &mut ctx, &mut s, &mut rhs).unwrap()
        });

        let s_csc = s.to_csc();
        let s_nnz = s_csc.compute_nnz();
        let density = 100.0 * s_nnz as f64 / (nk as f64 * (nk as f64 + 1.0) / 2.0);
        let ordering = if density > 25.0 {
            SymmetricOrdering::Identity
        } else {
            SymmetricOrdering::Amd
        };
        let (t_sym, llt_sym) = min_ms(rounds, || {
            factorize_symbolic_cholesky(
                s_csc.as_ref().symbolic(),
                faer::Side::Upper,
                ordering,
                CholeskySymbolicParams::default(),
            )
            .unwrap()
        });
        let mut l_vals = vec![0.0f64; llt_sym.len_val()];
        let mut factor_mem = vec![
            std::mem::MaybeUninit::<u8>::uninit();
            llt_sym
                .factorize_numeric_llt_scratch::<f64>(faer::Par::Seq, faer::Spec::default())
                .unaligned_bytes_required()
        ];
        let (t_num, _) = min_ms(rounds, || {
            let stack = MemStack::new(&mut factor_mem);
            llt_sym
                .factorize_numeric_llt(
                    &mut l_vals,
                    s_csc.as_ref(),
                    faer::Side::Upper,
                    faer::linalg::cholesky::llt::factor::LltRegularization::default(),
                    faer::Par::Seq,
                    stack,
                    faer::Spec::default(),
                )
                .unwrap();
        });

        println!("{:<14} {:>7} {:>8} {:>9.1} {:>7} {:>8.1} {:>10.1} {:>9.1} {:>9.1}",
            name,
            ds.cameras.len(),
            nk,
            density,
            sym.pair_count(),
            l_vals.len() as f64 * 8.0 / 1e6,
            t_reduce,
            t_sym,
            t_num);
    }
}
