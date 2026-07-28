// Block-CSC assembly: the model's Hessian assembled into
// arael_faer::bsc::SparseBlockColMat must equal the scalar CSC path
// bit for bit. Both paths replay the same generated indexed scatter
// (calc_grad_hessian_sparse_indexed) -- only the position maps differ:
// the scalar one comes from CooMatrix::to_csc_with_map, the block one
// from SymbolicSparseBlockColMat::from_scalar_coords over the entity
// partition (RootProblem::param_block_spans + block_partition_from_spans).

use arael::model::{CrossBlock, HessianBinder, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{block_partition_from_spans, csc_from_cells, CooMatrix, LmProblem, RootProblem};
use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.1, (pose.y - pose.ay) * 0.1]
}))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct Landmark {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Landmark>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy]
}))]
struct Odo {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(constraint(hb, {
    [l.x - p.x - obs.dx, l.y - p.y - obs.dy]
}))]
struct Obs {
    #[arael(ref = root.poses)]
    p: Ref<Pose>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Landmark>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    odos: std::vec::Vec<Odo>,
    obs: std::vec::Vec<Obs>,
}

const N_POSES: usize = 4;
const N_LANDMARKS: usize = 6;

fn build() -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for i in 0..N_POSES {
        w.poses.push(Pose {
            x: Param::new(i as f64),
            y: Param::new(0.1 * i as f64),
            ax: i as f64,
            ay: 0.0,
            hb: SelfBlock::new(),
        });
    }
    for j in 0..N_LANDMARKS {
        w.landmarks.push(Landmark {
            x: Param::new(0.6 * j as f64),
            y: Param::new(1.0 + 0.05 * j as f64),
            hb: SelfBlock::new(),
        });
    }
    for i in 1..N_POSES {
        w.odos.push(Odo {
            a: w.poses.ref_at(i - 1),
            b: w.poses.ref_at(i),
            dx: 1.0,
            dy: 0.0,
            hb: CrossBlock::new(),
        });
    }
    for j in 0..N_LANDMARKS {
        for pi in [j % N_POSES, (j + 1) % N_POSES] {
            w.obs.push(Obs {
                p: w.poses.ref_at(pi),
                l: w.landmarks.ref_at(j),
                dx: 0.3,
                dy: 1.0,
                hb: CrossBlock::new(),
            });
        }
    }
    w
}

/// Densify arael's scalar CSC (upper triangle as stored; no mirroring,
/// both paths store the same upper-only entries).
fn densify_scalar(csc: &arael::simple_lm::CscMatrix<f64>) -> Vec<f64> {
    let n = csc.n;
    let mut full = vec![0.0; n * n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            full[csc.row_idx[k] as usize * n + j] = csc.vals[k];
        }
    }
    full
}

/// Densify the block matrix through its scalar expansion.
fn densify_block(m: &SparseBlockColMat<usize, f64>) -> Vec<f64> {
    let csc = m.to_csc();
    let n = csc.nrows();
    let mut full = vec![0.0; n * n];
    for j in 0..csc.ncols() {
        for (i, v) in std::iter::zip(csc.row_idx_of_col(j), csc.val_of_col(j)) {
            full[i * n + j] = *v;
        }
    }
    full
}

#[test]
fn block_assembly_matches_scalar() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let n = params.len();
    assert_eq!(n, 2 * (N_POSES + N_LANDMARKS));

    // Entity spans: one per pose and landmark, width 2, serialize order.
    let spans = RootProblem::param_block_spans(&w);
    assert_eq!(spans.len(), N_POSES + N_LANDMARKS);
    assert!(spans.iter().enumerate().all(|(k, &(off, width))| {
        off as usize == 2 * k && width == 2
    }));
    let partition = block_partition_from_spans(&spans, n);
    assert_eq!(partition, (0..=N_POSES + N_LANDMARKS).map(|k| 2 * k).collect::<Vec<_>>());

    // Pattern discovery: one COO pass (shared by both pipelines).
    let mut grad = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    w.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);

    // Scalar pipeline: CSC + positions, indexed refill.
    let (mut csc, positions_scalar) = coo.to_csc_with_map().unwrap();
    let mut grad_s = vec![0.0; n];
    let mut vals_s = vec![0.0; csc.vals.len()];
    let cost_s = w.calc_grad_hessian_sparse_indexed(&params, &mut grad_s, &mut vals_s, &positions_scalar);
    csc.vals = vals_s;

    // Block pipeline: bsc structure + positions over the SAME coords,
    // refilled by the SAME generated scatter.
    let (sym, positions_block) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition,
        coo.nnz(),
        |k| (coo.rows[k] as usize, coo.cols[k] as usize),
    );
    assert_eq!(positions_block.len(), positions_scalar.len());
    let mut bsc = SparseBlockColMat::<usize, f64>::zeroed(sym);
    let mut grad_b = vec![0.0; n];
    let cost_b = w.calc_grad_hessian_sparse_indexed(&params, &mut grad_b, bsc.vals_mut(), &positions_block);

    // Same traversal, same emission order, same additions: exact equality.
    assert_eq!(cost_s, cost_b);
    assert_eq!(grad_s, grad_b);
    let dense_s = densify_scalar(&csc);
    let dense_b = densify_block(&bsc);
    assert_eq!(dense_s.len(), dense_b.len());
    for (k, (a, b)) in std::iter::zip(&dense_s, &dense_b).enumerate() {
        assert!(a == b, "H[{}, {}]: scalar {} vs block {}", k / n, k % n, a, b);
    }

    // Structure sanity: diagonal tiles exist for every entity; the
    // pose-landmark tiles sit above the diagonal (upper convention).
    for k in 0..N_POSES + N_LANDMARKS {
        assert!(bsc.get_block(k, k).is_some(), "diagonal tile {}", k);
    }
    let first_lm_block = N_POSES;
    assert!(bsc.get_block(0, first_lm_block).is_some());
    assert!(bsc.get_block(first_lm_block, 0).is_none());
}

/// The two-scan construction (structure from block indices, positions by
/// replaying the emission order -- no COO) must produce the identical
/// symbolic structure and position map as the COO route.
#[test]
fn two_scan_matches_coo_route() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let n = params.len();
    let partition = block_partition_from_spans(&RootProblem::param_block_spans(&w), n);

    // reference: COO route
    let mut grad = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    w.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
    let (sym_coo, pos_coo) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition.clone(),
        coo.nnz(),
        |k| (coo.rows[k] as usize, coo.cols[k] as usize),
    );

    // two-scan: cells from indices alone, positions by emission replay
    let mut cells: Vec<(u32, u32)> = Vec::new();
    arael::model::Model::collect_hessian_cells64(&w, &mut cells);
    let (sym2, _) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition,
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut resolver = PositionResolver::new(&sym2);
    let mut pos2: Vec<arael::ValueIndex> = Vec::new();
    arael::model::Model::bind_hessian_positions64(
        &mut w,
        &mut HessianBinder::Scalar(&mut |i, j| resolver.resolve(i as usize, j as usize)),
        &mut pos2,
    );

    assert_eq!(sym2.parts().0, sym_coo.parts().0);
    assert_eq!(sym2.parts().1, sym_coo.parts().1);
    assert_eq!(sym2.parts().2, sym_coo.parts().2);
    assert_eq!(sym2.parts().3, sym_coo.parts().3);
    assert_eq!(sym2.parts().4, sym_coo.parts().4);
    assert_eq!(pos2, pos_coo);

    // and a fixed param reshapes both routes identically
    let mut w = build();
    w.landmarks[2].x = Param::fixed(1.2);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let n = params.len();
    let partition = block_partition_from_spans(&RootProblem::param_block_spans(&w), n);
    let mut grad = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    w.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
    let (sym_coo, pos_coo) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition.clone(),
        coo.nnz(),
        |k| (coo.rows[k] as usize, coo.cols[k] as usize),
    );
    let mut cells: Vec<(u32, u32)> = Vec::new();
    arael::model::Model::collect_hessian_cells64(&w, &mut cells);
    let (sym2, _) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition,
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut resolver = PositionResolver::new(&sym2);
    let mut pos2: Vec<arael::ValueIndex> = Vec::new();
    arael::model::Model::bind_hessian_positions64(
        &mut w,
        &mut HessianBinder::Scalar(&mut |i, j| resolver.resolve(i as usize, j as usize)),
        &mut pos2,
    );
    assert_eq!(sym2.parts().4, sym_coo.parts().4);
    assert_eq!(pos2, pos_coo);
}

/// The scalar fast path (tile-expanded CSC from cells, SparseFaer's new
/// first iteration) must produce the same dense H as the COO route --
/// the pattern differs only by structural zeros inside stored tiles.
#[test]
fn scalar_fast_path_matches_coo_route() {
    let mut w = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let n = params.len();

    assert!(!LmProblem::hessian_pattern_requires_compute(&w));

    // COO reference
    let mut grad = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    w.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
    let (mut csc_ref, pos_ref) = coo.to_csc_with_map().unwrap();
    let mut vals = vec![0.0; csc_ref.vals.len()];
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut vals, &pos_ref);
    csc_ref.vals = vals;

    // fast path: cells -> tile-expanded CSC -> positions -> indexed fill
    let mut spans = Vec::new();
    LmProblem::collect_param_block_spans(&w, &mut spans);
    let partition = block_partition_from_spans(&spans, n);
    let mut cells = Vec::new();
    LmProblem::collect_hessian_cells(&w, &mut cells);
    let (mut csc_fast, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
    let mut positions: Vec<arael::ValueIndex> = Vec::new();
    LmProblem::bind_hessian_positions(
        &mut w,
        &mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i, j)),
        &mut positions,
    );
    // Every block here has a static tile shape, so nothing needs a map.
    assert!(positions.is_empty());
    let mut vals = vec![0.0; csc_fast.vals.len()];
    let mut grad_f = vec![0.0; n];
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad_f, &mut vals, &positions);
    csc_fast.vals = vals;

    assert_eq!(grad, grad_f);
    // diag_pos must point at true diagonals
    for j in 0..n {
        let d = csc_fast.diag_pos[j] as usize;
        assert_eq!(csc_fast.row_idx[d] as usize, j);
        assert!(d >= csc_fast.col_ptr[j] && d < csc_fast.col_ptr[j + 1]);
    }
    let dense_ref = densify_scalar(&csc_ref);
    let dense_fast = densify_scalar(&csc_fast);
    for (k, (a, b)) in std::iter::zip(&dense_ref, &dense_fast).enumerate() {
        assert!(a == b, "H[{}, {}]: coo {} vs fast {}", k / n, k % n, a, b);
    }
}

/// All construction routes must agree on a model with fixed params:
/// spans cover exactly the live params, the scalar fast path and the
/// COO route produce the same dense H and gradient, diag_pos stays
/// valid, and the block two-scan matches the block COO route.
fn assert_routes_agree(w: &mut World) {
    let mut params = Vec::new();
    RootProblem::serialize(w, &mut params);
    let n = params.len();

    let mut spans = Vec::new();
    LmProblem::collect_param_block_spans(w, &mut spans);
    let live: usize = spans.iter().map(|&(_, width)| width as usize).sum();
    assert_eq!(live, n, "spans must cover exactly the live params");
    let partition = block_partition_from_spans(&spans, n);

    // COO reference, filled
    let mut grad_ref = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    w.calc_grad_hessian_sparse(&params, &mut grad_ref, &mut coo);
    let (mut csc_ref, pos_ref) = coo.to_csc_with_map().unwrap();
    let mut vals = vec![0.0; csc_ref.vals.len()];
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad_ref, &mut vals, &pos_ref);
    csc_ref.vals = vals;

    // scalar fast path, filled
    let mut cells = Vec::new();
    LmProblem::collect_hessian_cells(w, &mut cells);
    let (mut csc_fast, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
    let mut positions: Vec<arael::ValueIndex> = Vec::new();
    LmProblem::bind_hessian_positions(
        w,
        &mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i, j)),
        &mut positions,
    );
    assert!(positions.is_empty());
    let mut vals = vec![0.0; csc_fast.vals.len()];
    let mut grad_fast = vec![0.0; n];
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad_fast, &mut vals, &positions);
    csc_fast.vals = vals;

    assert_eq!(grad_ref, grad_fast);
    for j in 0..n {
        assert_eq!(csc_fast.row_idx[csc_fast.diag_pos[j] as usize] as usize, j);
    }
    let dense_ref = densify_scalar(&csc_ref);
    let dense_fast = densify_scalar(&csc_fast);
    for (k, (a, b)) in std::iter::zip(&dense_ref, &dense_fast).enumerate() {
        assert!(a == b, "H[{}, {}]: coo {} vs fast {}", k / n, k % n, a, b);
    }

    // block two-scan vs block COO route
    let (sym_coo, pos_coo) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition.clone(),
        coo.nnz(),
        |k| (coo.rows[k] as usize, coo.cols[k] as usize),
    );
    let (sym2, _) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition,
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut resolver = PositionResolver::new(&sym2);
    let mut pos2: Vec<arael::ValueIndex> = Vec::new();
    arael::model::Model::bind_hessian_positions64(
        w,
        &mut HessianBinder::Scalar(&mut |i, j| resolver.resolve(i as usize, j as usize)),
        &mut pos2,
    );
    assert_eq!(sym2.parts().3, sym_coo.parts().3);
    assert_eq!(sym2.parts().4, sym_coo.parts().4);
    assert_eq!(pos2, pos_coo);
}

/// Blocks that keep only their tile origin and column stride must scatter
/// into exactly the slots the per-scalar map names -- on the scalar CSC and
/// on the block CSC, with fixed params punching holes in both.
fn assert_tiled_matches_mapped(w: &mut World) {
    let mut params = Vec::new();
    RootProblem::serialize(w, &mut params);
    let n = params.len();
    let mut spans = Vec::new();
    LmProblem::collect_param_block_spans(w, &mut spans);
    let partition = block_partition_from_spans(&spans, n);
    let mut cells = Vec::new();
    LmProblem::collect_hessian_cells(w, &mut cells);
    let mut grad = vec![0.0; n];

    // Scalar CSC: same pattern, filled once through each path.
    let (csc, _) = csc_from_cells::<f64>(&partition, &cells);
    let mut mapped: Vec<arael::ValueIndex> = Vec::new();
    {
        let (_, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
        LmProblem::bind_hessian_positions(
            w,
            &mut HessianBinder::Scalar(&mut |i, j| resolver.resolve(i, j)),
            &mut mapped,
        );
    }
    let mut vals_mapped = vec![0.0; csc.vals.len()];
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut vals_mapped, &mapped);

    let (_, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
    let mut tiled: Vec<arael::ValueIndex> = Vec::new();
    LmProblem::bind_hessian_positions(
        w,
        &mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i, j)),
        &mut tiled,
    );
    assert!(tiled.is_empty(), "static tile shapes need no per-scalar map");
    let mut vals_tiled = vec![0.0; csc.vals.len()];
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut vals_tiled, &tiled);
    assert_eq!(vals_mapped, vals_tiled, "scalar CSC: tiled fill differs from mapped");

    // Block CSC: the other resolver, over the same cells.
    let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition,
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut bmapped: Vec<arael::ValueIndex> = Vec::new();
    {
        let mut resolver = PositionResolver::new(&sym);
        LmProblem::bind_hessian_positions(
            w,
            &mut HessianBinder::Scalar(&mut |i, j| resolver.resolve(i as usize, j as usize)),
            &mut bmapped,
        );
    }
    let mut bsc_mapped = SparseBlockColMat::<usize, f64>::zeroed(sym.clone());
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad, bsc_mapped.vals_mut(), &bmapped);

    let mut resolver = PositionResolver::new(&sym);
    let mut btiled: Vec<arael::ValueIndex> = Vec::new();
    LmProblem::bind_hessian_positions(
        w,
        &mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i as usize, j as usize)),
        &mut btiled,
    );
    assert!(btiled.is_empty());
    let mut bsc_tiled = SparseBlockColMat::<usize, f64>::zeroed(sym);
    w.calc_grad_hessian_sparse_indexed(&params, &mut grad, bsc_tiled.vals_mut(), &btiled);
    assert_eq!(
        bsc_mapped.vals(), bsc_tiled.vals(),
        "block CSC: tiled fill differs from mapped",
    );
}

/// The tile fast path and the per-scalar map must fill identically, with
/// every combination of live and fixed parameters.
#[test]
fn tiled_fill_matches_mapped_fill() {
    assert_tiled_matches_mapped(&mut build());

    // one param fixed inside a pose and inside a landmark
    let mut w = build();
    w.poses[1].y = Param::fixed(0.1);
    w.landmarks[3].x = Param::fixed(1.8);
    assert_tiled_matches_mapped(&mut w);

    // whole entities fixed: they leave the partition entirely
    let mut w = build();
    w.poses[2].x = Param::fixed(2.0);
    w.poses[2].y = Param::fixed(0.2);
    w.landmarks[0].x = Param::fixed(0.0);
    w.landmarks[0].y = Param::fixed(1.0);
    assert_tiled_matches_mapped(&mut w);
}

/// One fixed param inside a pose and inside a landmark: entities
/// shrink but stay live; every route must agree.
#[test]
fn fixed_param_partial_entities() {
    let mut w = build();
    w.poses[1].y = Param::fixed(0.1);
    w.landmarks[3].x = Param::fixed(1.8);
    let mut spans = Vec::new();
    {
        let mut p = Vec::new();
        RootProblem::serialize(&mut w, &mut p);
        LmProblem::collect_param_block_spans(&w, &mut spans);
    }
    assert_eq!(spans.len(), N_POSES + N_LANDMARKS);
    assert_eq!(spans.iter().filter(|&&(_, width)| width == 1).count(), 2);
    assert_routes_agree(&mut w);
}

/// Entire entities fixed (a whole pose and a whole landmark): they
/// vanish from the partition and from every coupling; the routes must
/// still agree on the shrunken system.
#[test]
fn fixed_whole_entities() {
    let mut w = build();
    w.poses[2].x = Param::fixed(2.0);
    w.poses[2].y = Param::fixed(0.2);
    w.landmarks[0].x = Param::fixed(0.0);
    w.landmarks[0].y = Param::fixed(1.0);
    let mut spans = Vec::new();
    {
        let mut p = Vec::new();
        RootProblem::serialize(&mut w, &mut p);
        LmProblem::collect_param_block_spans(&w, &mut spans);
    }
    assert_eq!(spans.len(), N_POSES + N_LANDMARKS - 2);
    assert_routes_agree(&mut w);
}

/// End-to-end: solving through SparseFaer's fast path with fixed
/// params reaches the dense solver's optimum, and the fixed params do
/// not move.
#[test]
fn fixed_params_solve_matches_dense() {
    use arael::simple_lm::LmConfig;
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let fix = |w: &mut World| {
        w.poses[0].x = Param::fixed(0.25);
        w.landmarks[4].y = Param::fixed(1.3);
    };

    let mut wd = build();
    fix(&mut wd);
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut ws = build();
    fix(&mut ws);
    let rs = ws.solve_sparse(&cfg).unwrap();

    assert!((rd.end_cost - rs.end_cost).abs() <= 1e-12 * (1.0 + rd.end_cost),
        "dense {} vs sparse {}", rd.end_cost, rs.end_cost);
    assert_eq!(ws.poses[0].x.value, 0.25);
    assert_eq!(ws.landmarks[4].y.value, 1.3);
    for j in 0..N_LANDMARKS {
        let (a, b) = (
            &wd.landmarks[j as usize],
            &ws.landmarks[j as usize],
        );
        assert!((a.x.value - b.x.value).abs() < 1e-8, "landmark {} x", j);
        assert!((a.y.value - b.y.value).abs() < 1e-8, "landmark {} y", j);
    }
}

/// The complete fixing matrix on objects of one type: (live, live),
/// (fixed, live), (live, fixed), (fixed, fixed) -- applied to both
/// poses 0-3 and landmarks 0-3 simultaneously. Exercises min-live
/// anchors, width-1 partition cells adjacent to width-2 ones, cross
/// tiles of every shape (2x2, 2x1, 1x2, 1x1) and vanished couplings.
#[test]
fn fixed_param_full_matrix() {
    let mut w = build();
    // poses: 0 full, 1 first-fixed, 2 second-fixed, 3 both-fixed
    w.poses[1].x = Param::fixed(1.0);
    w.poses[2].y = Param::fixed(0.2);
    w.poses[3].x = Param::fixed(3.0);
    w.poses[3].y = Param::fixed(0.3);
    // landmarks: same pattern
    w.landmarks[1].x = Param::fixed(0.6);
    w.landmarks[2].y = Param::fixed(1.1);
    w.landmarks[3].x = Param::fixed(1.8);
    w.landmarks[3].y = Param::fixed(1.15);

    let mut spans = Vec::new();
    {
        let mut p = Vec::new();
        RootProblem::serialize(&mut w, &mut p);
        LmProblem::collect_param_block_spans(&w, &mut spans);
    }
    // both-fixed entities vanish; the rest shrink to their live widths
    let widths: Vec<u32> = spans.iter().map(|&(_, width)| width).collect();
    assert_eq!(widths, vec![2, 1, 1, /* poses */ 2, 1, 1, 2, 2 /* landmarks */]);
    // spans are contiguous over the live params
    let mut off = 0u32;
    for &(o, width) in &spans {
        assert_eq!(o, off);
        off += width;
    }

    assert_routes_agree(&mut w);

    // and the solve agrees with dense, leaving fixed params untouched
    use arael::simple_lm::LmConfig;
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let rd = {
        let mut wd = build();
        wd.poses[1].x = Param::fixed(1.0);
        wd.poses[2].y = Param::fixed(0.2);
        wd.poses[3].x = Param::fixed(3.0);
        wd.poses[3].y = Param::fixed(0.3);
        wd.landmarks[1].x = Param::fixed(0.6);
        wd.landmarks[2].y = Param::fixed(1.1);
        wd.landmarks[3].x = Param::fixed(1.8);
        wd.landmarks[3].y = Param::fixed(1.15);
        wd.solve_dense(&cfg).unwrap()
    };
    let rs = w.solve_sparse(&cfg).unwrap();
    assert!((rd.end_cost - rs.end_cost).abs() <= 1e-12 * (1.0 + rd.end_cost),
        "dense {} vs sparse {}", rd.end_cost, rs.end_cost);
    assert_eq!(w.poses[3].x.value, 3.0);
    assert_eq!(w.poses[3].y.value, 0.3);
    assert_eq!(w.landmarks[3].x.value, 1.8);
    assert_eq!(w.landmarks[3].y.value, 1.15);
}

/// The Schur-complement backend (landmarks eliminated every damped
/// solve, only the pose system factorized) must land on the same
/// optimum as the plain sparse backend.
#[test]
fn schur_solve_matches_sparse() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut ws = build();
    let rs = ws.solve_sparse(&cfg).unwrap();

    let mut wq = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut wq, &mut params); // populates block indices
    let lm_start = RootProblem::param_block_spans(&wq)[N_POSES].0 as usize;
    let mut solver = SparseFaer::new().with_marginalize(lm_start..params.len());
    let rq = wq.solve_with(&mut solver, &cfg).unwrap();

    assert!(
        (rs.end_cost - rq.end_cost).abs() <= 1e-10 * (1.0 + rs.end_cost),
        "sparse {} vs schur {}",
        rs.end_cost,
        rq.end_cost
    );
    for i in 0..N_POSES {
        let (a, b) = (&ws.poses[i as usize], &wq.poses[i as usize]);
        assert!((a.x.value - b.x.value).abs() < 1e-8, "pose {} x", i);
        assert!((a.y.value - b.y.value).abs() < 1e-8, "pose {} y", i);
    }
    for j in 0..N_LANDMARKS {
        let (a, b) = (
            &ws.landmarks[j as usize],
            &wq.landmarks[j as usize],
        );
        assert!((a.x.value - b.x.value).abs() < 1e-8, "landmark {} x", j);
        assert!((a.y.value - b.y.value).abs() < 1e-8, "landmark {} y", j);
    }
}

/// Schur backend under partially fixed parameters (shrunken blocks on
/// both sides of the elimination), against the dense reference.
#[test]
fn schur_solve_with_fixed_params() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let fix = |w: &mut World| {
        w.poses[0].x = Param::fixed(0.25);
        w.landmarks[4].y = Param::fixed(1.3);
    };

    let mut wd = build();
    fix(&mut wd);
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut wq = build();
    fix(&mut wq);
    let mut params = Vec::new();
    RootProblem::serialize(&mut wq, &mut params); // populates block indices
    let lm_start = RootProblem::param_block_spans(&wq)[N_POSES].0 as usize;
    let mut solver = SparseFaer::new().with_marginalize(lm_start..params.len());
    let rq = wq.solve_with(&mut solver, &cfg).unwrap();

    assert!(
        (rd.end_cost - rq.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost),
        "dense {} vs schur {}",
        rd.end_cost,
        rq.end_cost
    );
    assert_eq!(wq.poses[0].x.value, 0.25);
    assert_eq!(wq.landmarks[4].y.value, 1.3);
    for j in 0..N_LANDMARKS {
        let (a, b) = (
            &wd.landmarks[j as usize],
            &wq.landmarks[j as usize],
        );
        assert!((a.x.value - b.x.value).abs() < 1e-8, "landmark {} x", j);
        assert!((a.y.value - b.y.value).abs() < 1e-8, "landmark {} y", j);
    }
}

/// Every scalar-CSC backend shares one first-call assembly
/// (assemble_first_csc), so the tile-expanded pattern -- which carries
/// the blocks' structural zeros as explicit entries -- must be as
/// solvable for Eigen and CHOLMOD as it is for faer. Each backend must
/// land on the dense solve's optimum.
#[cfg(feature = "eigen")]
#[test]
fn eigen_backend_matches_dense() {
    use arael::simple_lm::{solve_sparse_eigen, LmConfig};
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wd = build();
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut we = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut we, &mut params);
    let re = solve_sparse_eigen(&params, &mut we, &cfg).unwrap();

    assert!(
        (rd.end_cost - re.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost),
        "dense {} vs eigen {}",
        rd.end_cost,
        re.end_cost
    );
}

#[cfg(feature = "cholmod")]
#[test]
fn cholmod_backend_matches_dense() {
    use arael::simple_lm::{solve_sparse_cholmod, LmConfig};
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wd = build();
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut wc = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut wc, &mut params);
    let rc = solve_sparse_cholmod(&params, &mut wc, &cfg).unwrap();

    assert!(
        (rd.end_cost - rc.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost),
        "dense {} vs cholmod {}",
        rd.end_cost,
        rc.end_cost
    );
}

#[cfg(feature = "cholmod-gpl")]
#[test]
fn cholmod_supernodal_backend_matches_dense() {
    use arael::simple_lm::{solve_sparse_cholmod_supernodal, LmConfig};
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wd = build();
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut wc = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut wc, &mut params);
    let rc = solve_sparse_cholmod_supernodal(&params, &mut wc, &cfg).unwrap();

    assert!(
        (rd.end_cost - rc.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost),
        "dense {} vs cholmod supernodal {}",
        rd.end_cost,
        rc.end_cost
    );
}
