// Block-CSC assembly: the model's Hessian assembled into
// arael_faer::bsc::SparseBlockColMat must equal the scalar CSC path
// bit for bit. Both paths replay the same generated indexed scatter
// (calc_grad_hessian_sparse_indexed) -- only the position maps differ:
// the scalar one comes from CooMatrix::to_csc_with_map, the block one
// from SymbolicSparseBlockColMat::from_scalar_coords over the entity
// partition (RootProblem::param_block_spans + block_partition_from_spans).

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{block_partition_from_spans, CooMatrix, LmProblem, RootProblem};
use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};

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
            a: Ref::new((i - 1) as u32),
            b: Ref::new(i as u32),
            dx: 1.0,
            dy: 0.0,
            hb: CrossBlock::new(),
        });
    }
    for j in 0..N_LANDMARKS {
        for pi in [j % N_POSES, (j + 1) % N_POSES] {
            w.obs.push(Obs {
                p: Ref::new(pi as u32),
                l: Ref::new(j as u32),
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
    let (mut csc, positions_scalar) = coo.to_csc_with_map();
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
