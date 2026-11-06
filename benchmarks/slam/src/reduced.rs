//! The slam problem's Schur-reduced system S, built as the solver's
//! backend builds it: the Hessian at the initial point, LM-damped, the
//! biggest marginalizable set eliminated. Shared by the factorization
//! benchmarks.

use arael::simple_lm::{LmProblemInternals, RootProblem};
use arael::store::block_partition_from_spans;
use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};
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
    let mut hctx = arael::threads::Context::new();
    let scene = scene::generate(cfg);
    let mut path = arael_runner::build(&scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize(&mut params);
    let n = params.len();
    let mut spans = Vec::new();
    path.collect_param_block_spans(&mut spans, &mut hctx);
    let partition = block_partition_from_spans(&spans, n);
    let mut cells: Vec<(u32, u32)> = Vec::new();
    LmProblemInternals::collect_hessian_cells(&path, &mut cells, &mut hctx);
    let part: Vec<I> = partition.iter().map(|&p| I::truncate(p)).collect();
    let (hsym, _) = SymbolicSparseBlockColMat::<I>::from_scalar_coords(
        part.clone(),
        part,
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut resolver = PositionResolver::new(&hsym);
    let mut positions = arael::store::PositionStream::new();
    LmProblemInternals::bind_hessian_positions(
        &mut path,
        &mut arael::store::HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i as usize, j as usize)),
        &mut positions,
        &mut hctx,
    );
    let mut h = SparseBlockColMat::<I, f64>::zeroed(hsym);
    let mut grad = vec![0.0; n];
    path.calc_grad_hessian_sparse_indexed(&params, &mut grad, h.vals_mut(), &positions, &mut hctx);
    // LM damping, as a real solve applies it before reducing
    let nblk = partition.len() - 1;
    for b in 0..nblk {
        let w = partition[b + 1] - partition[b];
        let hs = h.symbolic().clone();
        let d = hs.col_range(b).find(|&x| hs.blk_row(x) == b).unwrap();
        let base = hs.val_range(d).start;
        for k in 0..w {
            h.vals_mut()[base + k * (w + 1)] *= 1.0 + 1e-8;
        }
    }
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
