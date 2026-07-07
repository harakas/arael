// Solver / assembly-format equivalence.
//
// One macro-generated model -- a 2D zig-zag chain of points joined by
// nonlinear spring constraints (CrossBlock<Point, Point>), each point
// carrying a weak drift prior and the first a guarded anchor prior
// (SelfBlock) -- is pushed through every LmProblem code path:
//
//   assembly: dense / band / COO sparse / direct CSC / indexed CSC
//   solvers:  solve / solve_band / solve_sparse / solve_sparse_direct /
//             solve_sparse_faer
//
// All assembly formats must produce the identical gradient and Hessian,
// and all solvers must reach the same minimizer. One point is fixed
// (Param::fixed) so the u32::MAX index paths are exercised everywhere.
//
// TripletBlock band-format equivalence is covered by unit tests in
// src/model.rs (tripletblock_band_matches_*).

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::{self, LmProblem, LmConfig, CooMatrix, CscMatrix};
use arael::vect::vect2d;
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [point.pos.x * chain.anchor_isigma,
     point.pos.y * chain.anchor_isigma]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * chain.drift_isigma, d.y * chain.drift_isigma]
}))]
struct Point {
    pos: Param<vect2d>,
    is_anchor: bool,
    hb: SelfBlock<Point>,
}

// Nonlinear spring between consecutive points: residual on the distance.
#[arael::model]
#[arael(constraint(hb, {
    let d = b.pos - a.pos;
    [(d.norm() - link.rest) * chain.spring_isigma]
}))]
struct Link {
    #[arael(ref = root.points)]
    a: Ref<Point>,
    #[arael(ref = root.points)]
    b: Ref<Point>,
    rest: f64,
    hb: CrossBlock<Point, Point>,
}

#[arael::model]
#[arael(root)]
struct Chain {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    anchor_isigma: f64,
    drift_isigma: f64,
    spring_isigma: f64,
}

const N_POINTS: usize = 6;
const FIXED_POINT: usize = 2;
// Points are 2 params each and only consecutive points couple, so the
// widest index distance within one link is 3.
const KD: usize = 3;

fn build() -> Chain {
    let mut chain = Chain {
        points: refs::Vec::new(),
        links: std::vec::Vec::new(),
        anchor_isigma: 100.0,
        drift_isigma: 0.01,
        spring_isigma: 1.0,
    };
    for i in 0..N_POINTS {
        // Zig-zag with deterministic perturbations so spring residuals are
        // nonzero and genuinely two-dimensional at the start point.
        let x = i as f64 + ((i * 7 % 3) as f64 - 1.0) * 0.3;
        let y = if i % 2 == 0 { 0.0 } else { 0.8 } + ((i * 5 % 3) as f64 - 1.0) * 0.2;
        let pos = vect2d::new(x, y);
        chain.points.push(Point {
            pos: if i == FIXED_POINT { Param::fixed(pos) } else { Param::new(pos) },
            is_anchor: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for i in 1..N_POINTS {
        chain.links.push(Link {
            a: Ref::new((i - 1) as u32),
            b: Ref::new(i as u32),
            rest: 1.1,
            hb: CrossBlock::new(),
        });
    }
    chain
}

/// Expand a LAPACK upper-band matrix into a full dense symmetric matrix.
fn densify_band(band: &[f64], n: usize, kd: usize) -> Vec<f64> {
    let ldab = kd + 1;
    let mut full = vec![0.0; n * n];
    for j in 0..n {
        for i in j.saturating_sub(kd)..=j {
            let v = band[(kd + i - j) + j * ldab];
            full[i * n + j] = v;
            full[j * n + i] = v;
        }
    }
    full
}

/// Expand an upper-triangle COO matrix into a full dense symmetric matrix.
fn densify_coo(coo: &CooMatrix<f64>, n: usize) -> Vec<f64> {
    let mut full = vec![0.0; n * n];
    for k in 0..coo.nnz() {
        let (i, j, v) = (coo.rows[k] as usize, coo.cols[k] as usize, coo.vals[k]);
        full[i * n + j] += v;
        if i != j {
            full[j * n + i] += v;
        }
    }
    full
}

/// Expand an upper-triangle CSC matrix into a full dense symmetric matrix.
fn densify_csc(csc: &CscMatrix<f64>) -> Vec<f64> {
    let n = csc.n;
    let mut full = vec![0.0; n * n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k] as usize;
            let v = csc.vals[k];
            full[i * n + j] += v;
            if i != j {
                full[j * n + i] += v;
            }
        }
    }
    full
}

#[test]
fn hessian_formats_agree() {
    let mut chain = build();
    let mut params = std::vec::Vec::new();
    chain.serialize64(&mut params);
    let n = params.len();
    assert_eq!(n, (N_POINTS - 1) * 2, "one point is fixed");

    // Dense is the reference.
    let mut g_dense = vec![0.0; n];
    let mut h_dense = vec![0.0; n * n];
    chain.calc_grad_hessian_dense(&params, &mut g_dense, &mut h_dense);
    assert!(g_dense.iter().any(|&g| g != 0.0), "test problem must be off its minimum");

    // Band.
    let mut g_band = vec![0.0; n];
    let mut band = vec![0.0; (KD + 1) * n];
    chain.calc_grad_hessian_band(&params, &mut g_band, &mut band, KD).unwrap();
    assert_eq!(g_band, g_dense);
    assert_eq!(densify_band(&band, n, KD), h_dense, "band assembly differs from dense");

    // COO sparse.
    let mut g_coo = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    chain.calc_grad_hessian_sparse(&params, &mut g_coo, &mut coo);
    assert_eq!(g_coo, g_dense);
    assert_eq!(densify_coo(&coo, n), h_dense, "COO assembly differs from dense");

    // Direct CSC (find_pos scatter into a pre-built structure).
    let mut csc_direct = coo.to_csc();
    csc_direct.vals.iter_mut().for_each(|v| *v = 0.0);
    let mut g_direct = vec![0.0; n];
    chain.calc_grad_hessian_sparse_direct(&params, &mut g_direct, &mut csc_direct);
    assert_eq!(g_direct, g_dense);
    assert_eq!(densify_csc(&csc_direct), h_dense, "direct CSC assembly differs from dense");

    // Indexed CSC (cached position map, the production steady-state path).
    let (csc, positions) = coo.to_csc_with_map();
    let mut vals = vec![0.0; csc.vals.len()];
    let mut g_indexed = vec![0.0; n];
    chain.calc_grad_hessian_sparse_indexed(&params, &mut g_indexed, &mut vals, &positions);
    let csc_indexed = CscMatrix { vals, ..csc };
    assert_eq!(g_indexed, g_dense);
    assert_eq!(densify_csc(&csc_indexed), h_dense, "indexed CSC assembly differs from dense");
}

#[test]
fn band_error_on_underdeclared_kd() {
    let mut chain = build();
    let mut params = std::vec::Vec::new();
    chain.serialize64(&mut params);
    let n = params.len();
    let kd = 1; // links couple indices up to distance 3
    let mut grad = vec![0.0; n];
    let mut band = vec![0.0; (kd + 1) * n];
    assert!(chain.calc_grad_hessian_band(&params, &mut grad, &mut band, kd).is_err());
}

#[test]
fn solvers_reach_same_minimizer() {
    let cfg = LmConfig::<f64> {
        abs_precision: 1e-14,
        rel_precision: 1e-12,
        max_iters: 200,
        ..Default::default()
    };
    let run = |which: &str| -> std::vec::Vec<f64> {
        let mut chain = build();
        let mut p = std::vec::Vec::new();
        chain.serialize64(&mut p);
        let r = match which {
            "dense" => simple_lm::solve(&p, &mut chain, &cfg),
            "band" => simple_lm::solve_band(&p, KD, &mut chain, &cfg),
            "sparse" => simple_lm::solve_sparse(&p, &mut chain, &cfg),
            "direct" => simple_lm::solve_sparse_direct(&p, &mut chain, &cfg),
            "faer" => simple_lm::solve_sparse_faer(&p, &mut chain, &cfg),
            _ => unreachable!(),
        };
        assert!(
            r.end_cost < r.start_cost,
            "{which}: no improvement ({} -> {})", r.start_cost, r.end_cost
        );
        r.x
    };

    let reference = run("dense");
    for which in ["band", "sparse", "direct", "faer"] {
        let x = run(which);
        for i in 0..reference.len() {
            assert!(
                (x[i] - reference[i]).abs() < 1e-6,
                "{which} disagrees with dense at param {i}: {} vs {}",
                x[i], reference[i]
            );
        }
    }
}

// The generated root convenience methods (solve_with / solve_dense /
// solve_sparse) must match the hand-written serialize -> solve ->
// deserialize dance, and leave the recovered parameters written back into
// the model. solve_dense == free solve; solve_sparse == free
// solve_sparse_faer; solve_with(Dense) == solve_dense.
#[test]
fn generated_solve_methods_match_manual_dance() {
    let cfg = LmConfig::<f64> {
        abs_precision: 1e-14,
        rel_precision: 1e-12,
        max_iters: 200,
        ..Default::default()
    };

    // Reference: the manual dance with the free dense solver.
    let reference = {
        let mut chain = build();
        let mut p = std::vec::Vec::new();
        chain.serialize64(&mut p);
        simple_lm::solve(&p, &mut chain, &cfg).x
    };

    // solve_dense writes the solution back into the model.
    let mut chain = build();
    let r = chain.solve_dense(&cfg);
    let mut back = std::vec::Vec::new();
    chain.serialize64(&mut back);
    assert_eq!(back, r.x, "solve_dense must write the solution back into the model");
    for i in 0..reference.len() {
        assert!((r.x[i] - reference[i]).abs() < 1e-12,
            "solve_dense disagrees with free solve at {i}: {} vs {}", r.x[i], reference[i]);
    }

    // solve_sparse must match the free faer sparse solver bit for bit.
    let faer = {
        let mut chain = build();
        let mut p = std::vec::Vec::new();
        chain.serialize64(&mut p);
        simple_lm::solve_sparse_faer(&p, &mut chain, &cfg).x
    };
    let mut chain = build();
    let r = chain.solve_sparse(&cfg);
    assert_eq!(r.x, faer, "solve_sparse must equal the free solve_sparse_faer");

    // solve_with(Dense) is exactly solve_dense.
    let mut chain = build();
    let r = chain.solve_with(&mut simple_lm::Dense, &cfg);
    for i in 0..reference.len() {
        assert!((r.x[i] - reference[i]).abs() < 1e-12,
            "solve_with(Dense) disagrees with free solve at {i}: {} vs {}", r.x[i], reference[i]);
    }

    // A non-default driver placed on the config must be honored by the
    // generated method: solve_sparse with a Nielsen config is bit-for-bit
    // the free faer solve with that same config (both read config.driver).
    let nielsen_cfg = cfg.clone().with_driver(simple_lm::NielsenLambdaDriver::default());
    let free_nielsen = {
        let mut chain = build();
        let mut p = std::vec::Vec::new();
        chain.serialize64(&mut p);
        simple_lm::solve_sparse_faer(&p, &mut chain, &nielsen_cfg).x
    };
    let mut chain = build();
    let r = chain.solve_sparse(&nielsen_cfg);
    assert_eq!(r.x, free_nielsen,
        "solve_sparse must route config.driver (Nielsen) like the free faer solve");
}
