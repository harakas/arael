//! What the Schur reduction faces on each BAL dataset: the size and
//! density of the reduced camera system S, the observer-pair count that
//! drives the reduction, and the cost split between forming S and
//! factorizing it. Explains where the explicit-Schur route wins and
//! where it stops winning.
//!
//!     cargo run -r --bin schur_stats

// The included modules are the benchmark's, and expose more than this binary
// calls.
#![allow(dead_code)]

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

/// Dump S in Matrix Market symmetric form (lower triangle, 1-based) when
/// SCHUR_DUMP_DIR is set, so the ordering experiments can run on exactly the
/// matrix the solver faces. Research scaffolding.
fn dump_s(name: &str, nk: usize, col_ptr: &[usize], row_idx: &[usize], vals: &[f64]) {
    let Ok(dir) = std::env::var("SCHUR_DUMP_DIR") else { return };
    use std::io::Write;
    let path = format!("{}/{}.mtx", dir, name);
    let f = std::fs::File::create(&path).expect("create mtx");
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "%%MatrixMarket matrix coordinate real symmetric").unwrap();
    // stored upper (i <= j); Matrix Market symmetric wants the lower triangle,
    // so emit each entry transposed
    // Our block CSC is tile-expanded: a diagonal tile carries its strictly
    // lower half as explicit zeros. Those are not part of S -- emit the true
    // upper triangle only, transposed, which is the lower triangle Matrix
    // Market's symmetric format asks for.
    let mut n_emitted = 0usize;
    for j in 0..nk {
        for k in col_ptr[j]..col_ptr[j + 1] {
            if row_idx[k] <= j {
                n_emitted += 1;
            }
        }
    }
    writeln!(w, "{} {} {}", nk, nk, n_emitted).unwrap();
    for j in 0..nk {
        for k in col_ptr[j]..col_ptr[j + 1] {
            if row_idx[k] <= j {
                writeln!(w, "{} {} {:.17e}", j + 1, row_idx[k] + 1, vals[k]).unwrap();
            }
        }
    }
    w.flush().unwrap();
    eprintln!("  dumped {} ({} x {}, {} nnz upper)", path, nk, nk, n_emitted);
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
    let datasets = [
        ("Ladybug-49", "datasets/problem-49-7776-pre.txt"),
        ("Ladybug-138", "datasets/problem-138-19878-pre.txt"),
        ("Ladybug-372", "datasets/problem-372-47423-pre.txt"),
        ("Ladybug-1723", "datasets/problem-1723-156502-pre.txt"),
    ];
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    println!("{:<14} {:>7} {:>8} {:>9} {:>7} {:>8} {:>10} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "dataset", "cams", "S n", "S dens%", "pairs", "L_S MB", "reduce ms", "sym ms", "numer ms",
        "L_H MB", "L_S/L_H", "Hsym ms");
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
        arael::model::Model::collect_hessian_cells(&scene, &mut cells);
        let (hsym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            partition.clone(),
            partition.clone(),
            cells.len(),
            |k| (cells[k].0 as usize, cells[k].1 as usize),
        );
        let mut resolver = PositionResolver::new(&hsym);
        let mut positions: Vec<arael::ValueIndex> = Vec::new();
        arael::model::Model::bind_hessian_positions(
            &mut scene,
            &mut arael::model::HessianBinder::Tiled(&mut |i, j| {
                resolver.resolve_tile(i as usize, j as usize)
            }),
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

        // What the solver marginalizes: the model names nothing, so take the
        // coupling graph's candidates and pick the biggest, exactly as
        // SparseFaer does.
        let blocks_in = |ranges: &[std::ops::Range<usize>]| -> Vec<usize> {
            (0..nblk)
                .filter(|&b| {
                    ranges.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end)
                })
                .collect()
        };
        let hint = RootProblem::marginalize_hint(&scene);
        let candidates: Vec<Vec<usize>> = if hint.is_empty() {
            LmProblem::marginalize_candidates(&scene).iter().map(|r| blocks_in(r)).collect()
        } else {
            vec![blocks_in(&hint)]
        };
        let params_of = |ids: &[usize]| -> usize {
            ids.iter().map(|&b| partition[b + 1] - partition[b]).sum::<usize>()
        };
        let eliminated: Vec<usize> = candidates
            .into_iter()
            .filter(|ids| !ids.is_empty())
            .max_by_key(|ids| params_of(ids))
            .expect("nothing marginalizable in a BAL scene");
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
        if std::env::var("SCHUR_DUMP_DIR").is_ok() {
            // S already exists in the exact CSC the factorization consumes.
            let (cp, ri) = sym.s.csc_pattern();
            let mut vals = vec![0.0f64; cp[nk]];
            s.csc_vals_into(&mut vals);
            dump_s(&format!("bal-{}", name), nk, &cp, &ri, &vals);
        }
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

        // The decision the auto-detector would have to make: is
        // "eliminate the points first" a better ELIMINATION ORDERING than
        // letting AMD order the whole camera+point system? Compare the
        // fill each leaves, symbolically -- no numeric factorization.
        let h_csc = h.to_csc();
        let (t_hsym, h_llt) = min_ms(1, || {
            factorize_symbolic_cholesky(
                h_csc.as_ref().symbolic(),
                faer::Side::Upper,
                SymmetricOrdering::Amd,
                CholeskySymbolicParams::default(),
            )
            .unwrap()
        });
        let fill_schur = l_vals.len() as f64;
        let fill_full = h_llt.len_val() as f64;

        println!("{:<14} {:>7} {:>8} {:>9.1} {:>7} {:>8.1} {:>10.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2} {:>8.0}",
            name,
            ds.cameras.len(),
            nk,
            density,
            sym.pair_count(),
            fill_schur * 8.0 / 1e6,
            t_reduce,
            t_sym,
            t_num,
            fill_full * 8.0 / 1e6,
            fill_schur / fill_full,
            t_hsym);
    }
}
