//! The slam problem's Schur-reduced system S, built as the solver's
//! backend builds it: the Hessian at the initial point, LM-damped
//! (`bench_harness::arael::hessian`), the biggest marginalizable set
//! eliminated. Shared by the factorization
//! benchmarks.

use arael::simple_lm::LmProblemInternals;
use arael_faer::bsc::SparseBlockColMat;
use arael_faer::faer::Index;
use arael_faer::schur::{SchurContext, SchurSymbolic, schur_reduce, schur_symbolic};

use crate::scene::SceneConfig;
use crate::{arael_runner, scene};

/// S, its reduction's symbolic analysis, its right-hand side, and the
/// damped Hessian it was reduced from.
pub struct Reduced<I: Index> {
    pub sym: SchurSymbolic<I>,
    pub s: SparseBlockColMat<I, f64>,
    pub rhs: Vec<f64>,
    pub h: SparseBlockColMat<I, f64>,
}

/// The reduced system of the slam scene `cfg` describes.
pub fn reduced_system<I: Index>(cfg: &SceneConfig) -> Reduced<I> {
    let scene = scene::generate(cfg);
    let mut path = arael_runner::build(&scene);
    let (h, grad, partition) = bench_harness::arael::hessian::<I, _>(&mut path);
    let nblk = partition.len() - 1;
    // What the solver marginalizes: the model names nothing, so take the
    // coupling graph's candidates and pick the biggest, exactly as SparseFaer
    // does.
    let blocks_in = |ranges: &[std::ops::Range<usize>]| -> Vec<usize> {
        (0..nblk).filter(|&b| ranges.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end)).collect()
    };
    let hint = LmProblemInternals::marginalize_hint(&path);
    let candidates: Vec<Vec<usize>> = if hint.is_empty() {
        LmProblemInternals::marginalize_candidates(&path).iter().map(|r| blocks_in(r)).collect()
    } else {
        vec![blocks_in(&hint)]
    };
    let eliminated: Vec<usize> = candidates
        .into_iter()
        .filter(|ids| !ids.is_empty())
        .max_by_key(|ids| ids.iter().map(|&b| partition[b + 1] - partition[b]).sum::<usize>())
        .expect("nothing marginalizable in the slam scene");
    let sym = schur_symbolic(h.symbolic(), &eliminated).unwrap();
    let mut s = sym.alloc_s::<f64>();
    let mut ctx = SchurContext::new();
    let mut rhs = vec![0.0; sym.s.nrows()];
    schur_reduce(&sym, &h, &grad, &mut ctx, &mut s, &mut rhs).unwrap();
    Reduced { sym, s, rhs, h }
}
