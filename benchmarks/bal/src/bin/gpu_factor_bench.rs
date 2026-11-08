//! A BAL dataset's reduced camera system S: its block supernodal
//! factorization and solve on a GPU against the CPU's
//! (`bench_harness::gpu_factor`, whose documentation lists the knobs). S is
//! the solver's: the Hessian at the initial point, damped by 1 + BAL_LAMBDA0
//! (1e-4: the Gauss-Newton Hessian is singular there), the points
//! eliminated.
//!
//!     BAL_DATASET=372 ROUNDS=20 cargo run -r --features gpu-cuda --bin gpu_factor_bench
//!
//! BAL_DATASET: the Ladybug problem's camera count, 49, 138, 372 (the
//! default) or 1723, read from datasets/problem-<n>-*-pre.txt.

// The included modules are the benchmark's, and expose more than this binary
// calls.
#![allow(dead_code)]

#[path = "../bal.rs"]
mod bal;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::LmProblemInternals;
use arael_faer::SparseIndex;
use arael_faer::schur::{SchurContext, schur_reduce, schur_symbolic};

fn main() {
    let cams = std::env::var("BAL_DATASET").unwrap_or_else(|_| "372".to_string());
    let prefix = format!("problem-{cams}-");
    let path = std::fs::read_dir("datasets")
        .expect("the datasets directory (run from benchmarks/bal)")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix) && n.ends_with("-pre.txt")))
        .unwrap_or_else(|| panic!("BAL_DATASET={cams}: no datasets/{prefix}*-pre.txt"));
    let ds = bal::load(path.to_str().unwrap());
    let mut scene = arael_runner::build_f64(&ds);
    let (mut h, grad, partition) = bench_harness::arael::hessian::<SparseIndex, _>(&mut scene);
    let lambda: f64 = std::env::var("BAL_LAMBDA0").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-4);
    let nblk = partition.len() - 1;
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
    // coupling graph's candidates and pick the biggest, as SparseFaer does.
    let blocks_in = |ranges: &[std::ops::Range<usize>]| -> Vec<usize> {
        (0..nblk).filter(|&b| ranges.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end)).collect()
    };
    let hint = LmProblemInternals::marginalize_hint(&scene);
    let candidates: Vec<Vec<usize>> =
        if hint.is_empty() { LmProblemInternals::marginalize_candidates(&scene).iter().map(|r| blocks_in(r)).collect() } else { vec![blocks_in(&hint)] };
    let eliminated: Vec<usize> = candidates
        .into_iter()
        .filter(|ids| !ids.is_empty())
        .max_by_key(|ids| ids.iter().map(|&b| partition[b + 1] - partition[b]).sum::<usize>())
        .expect("nothing marginalizable in a BAL scene");
    let sym = schur_symbolic(h.symbolic(), &eliminated).unwrap();
    let mut s = sym.alloc_s::<f64>();
    let mut ctx = SchurContext::new();
    let mut rhs = vec![0.0; sym.s.nrows()];
    schur_reduce(&sym, &h, &grad, &mut ctx, &mut s, &mut rhs).unwrap();
    bench_harness::gpu_factor::compare(&format!("S of Ladybug-{cams}"), &s, &rhs);
}
