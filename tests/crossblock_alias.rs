// CrossBlock with both slots resolving to the same entity ("aliased" --
// a legitimate pattern, e.g. a distance between the two endpoints of a
// single sketch line). The shared parameter needs the cross term
// 2 * dr_a * dr_b folded into the Hessian diagonal; the two symmetric
// accumulate writes land on the same cell and sum to exactly that.
// The accumulate paths used to skip gi == gj pairs, silently dropping
// the curvature: correct gradient, wrong Gauss-Newton Hessian, slow
// convergence with no error anywhere.
//
// The residual is nonlinear (a.x * b.x) so the dropped term is visible:
// for the aliased pair the true Hessian diagonal is (2x)^2 * 2, while
// the skipping version produced only half of it.

use arael::model::{Model, Param, SelfBlock, CrossBlock, JacobianModel};
use arael::simple_lm::{self, LmConfig, LmProblem, CooMatrix, CscMatrix};
use arael::refs::{self, Ref};

#[arael::model]
struct Pt {
    x: Param<f64>,
    hb: SelfBlock<Pt>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.x * b.x - m.k) * m.isigma,
     (a.x - 2.0 * b.x) * m.isigma]
}))]
struct Pair {
    #[arael(ref = root.pts)]
    a: Ref<Pt>,
    #[arael(ref = root.pts)]
    b: Ref<Pt>,
    hb: CrossBlock<Pt, Pt>,
}

#[arael::model]
#[arael(root, jacobian)]
struct M {
    pts: refs::Vec<Pt>,
    pairs: std::vec::Vec<Pair>,
    k: f64,
    isigma: f64,
}

/// One aliased pair (0,0) and one distinct pair (1,2), all points free.
fn build() -> (M, Vec<f64>) {
    let mut m = M {
        pts: refs::Vec::new(),
        pairs: std::vec::Vec::new(),
        k: 4.0,
        isigma: 1.3,
    };
    for &x in &[0.5, 1.3, -0.7] {
        m.pts.push(Pt { x: Param::new(x), hb: SelfBlock::new() });
    }
    m.pairs.push(Pair { a: Ref::new(0), b: Ref::new(0), hb: CrossBlock::new() });
    m.pairs.push(Pair { a: Ref::new(1), b: Ref::new(2), hb: CrossBlock::new() });
    let mut params = Vec::new();
    m.serialize64(&mut params);
    (m, params)
}

// --- format densifiers (same shapes as tests/solver_equivalence.rs) ---

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

/// The assembled Hessian must equal 2 * J^T J exactly -- including the
/// aliased pair's diagonal cross contribution. This is the test that
/// catches the dropped-curvature bug.
#[test]
fn aliased_hessian_matches_2jtj() {
    let (mut m, params) = build();
    let n = params.len();

    let mut grad = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut grad, &mut h);

    let j = m.calc_jacobian(&params);
    let dense = j.to_dense();
    let rows = j.num_residuals();
    assert_eq!(j.num_params, n);

    for i in 0..n {
        for jj in 0..n {
            let jtj: f64 = (0..rows).map(|r| dense[r * n + i] * dense[r * n + jj]).sum();
            let expected = 2.0 * jtj;
            assert!((h[i * n + jj] - expected).abs() < 1e-12,
                "H[{},{}] = {} but 2*J^T*J = {}", i, jj, h[i * n + jj], expected);
        }
    }
}

/// Gradient against central finite differences of calc_cost.
#[test]
fn aliased_gradient_matches_fd() {
    let (mut m, params) = build();
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut grad, &mut h);

    let eps = 1e-6;
    for i in 0..n {
        let mut p = params.clone();
        p[i] += eps;
        let cp = m.calc_cost(&p);
        p[i] -= 2.0 * eps;
        let cm = m.calc_cost(&p);
        let fd = (cp - cm) / (2.0 * eps);
        assert!((fd - grad[i]).abs() < 1e-5 * (1.0 + fd.abs()),
            "grad[{}]: analytic={} fd={}", i, grad[i], fd);
    }
}

/// All five assembly formats must agree on the aliased Hessian: each had
/// its own gi == gj skip.
#[test]
fn aliased_all_formats_agree() {
    let (mut m, params) = build();
    let n = params.len();
    let kd = n - 1;

    let mut g_dense = vec![0.0; n];
    let mut h_dense = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut g_dense, &mut h_dense);

    let mut g_band = vec![0.0; n];
    let mut band = vec![0.0; (kd + 1) * n];
    m.calc_grad_hessian_band(&params, &mut g_band, &mut band, kd).unwrap();
    assert_eq!(g_band, g_dense);
    assert_eq!(densify_band(&band, n, kd), h_dense, "band differs from dense");

    let mut g_coo = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    m.calc_grad_hessian_sparse(&params, &mut g_coo, &mut coo);
    assert_eq!(g_coo, g_dense);
    assert_eq!(densify_coo(&coo, n), h_dense, "COO differs from dense");

    let mut csc_direct = coo.to_csc();
    csc_direct.vals.iter_mut().for_each(|v| *v = 0.0);
    let mut g_direct = vec![0.0; n];
    m.calc_grad_hessian_sparse_direct(&params, &mut g_direct, &mut csc_direct);
    assert_eq!(g_direct, g_dense);
    assert_eq!(densify_csc(&csc_direct), h_dense, "direct CSC differs from dense");

    let (csc, positions) = coo.to_csc_with_map();
    let mut vals = vec![0.0; csc.vals.len()];
    let mut g_indexed = vec![0.0; n];
    m.calc_grad_hessian_sparse_indexed(&params, &mut g_indexed, &mut vals, &positions);
    let csc_indexed = CscMatrix { vals, ..csc };
    assert_eq!(g_indexed, g_dense);
    assert_eq!(densify_csc(&csc_indexed), h_dense, "indexed CSC differs from dense");
}

/// The aliased pair must actually optimize: minimize
/// ((x^2 - 4) * s)^2 + ((-x) * s)^2 -- a proper Newton descent needs the
/// full diagonal curvature. The system is inconsistent by design, so the
/// check is a vanishing gradient at the reached stationary point.
#[test]
fn aliased_pair_converges() {
    let mut m = M {
        pts: refs::Vec::new(),
        pairs: std::vec::Vec::new(),
        k: 4.0,
        isigma: 1.0,
    };
    m.pts.push(Pt { x: Param::new(0.5), hb: SelfBlock::new() });
    m.pairs.push(Pair { a: Ref::new(0), b: Ref::new(0), hb: CrossBlock::new() });
    let mut params = Vec::new();
    m.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut m, &LmConfig::default());
    m.deserialize64(&result.x);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    let mut params = Vec::new();
    m.serialize64(&mut params);
    m.calc_grad_hessian_dense(&params, &mut grad, &mut h);
    // Analytic stationary point of (x^2-4)^2 + x^2: x = sqrt(3.5).
    let x = m.pts[Ref::<Pt>::new(0)].x.value;
    assert!((x - 3.5_f64.sqrt()).abs() < 1e-4,
        "expected x = sqrt(3.5) = {}, got {}", 3.5_f64.sqrt(), x);
    assert!(grad.iter().all(|g| g.abs() < 1e-4),
        "gradient must vanish at the stationary point, grad={:?}", grad);
}

/// Distinct entities keep working exactly as before.
#[test]
fn distinct_entities_still_solve() {
    let mut m = M {
        pts: refs::Vec::new(),
        pairs: std::vec::Vec::new(),
        k: 2.0,
        isigma: 1.0,
    };
    m.pts.push(Pt { x: Param::new(0.9), hb: SelfBlock::new() });
    m.pts.push(Pt { x: Param::new(1.8), hb: SelfBlock::new() });
    m.pairs.push(Pair { a: Ref::new(0), b: Ref::new(1), hb: CrossBlock::new() });
    let mut params = Vec::new();
    m.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut m, &LmConfig::default());
    assert!(result.end_cost < 1e-20, "cost={}", result.end_cost);
}

// ---------------------------------------------------------------------------
// Side-by-side: the same constraint as a SelfBlock formulation and as an
// aliased CrossBlock formulation must assemble the identical Hessian.
// ---------------------------------------------------------------------------
//
// Residuals (identical math in both forms, x/y of one entity):
//   r1 = (x^2 + y^2 - k) * s
//   r2 = (-x + 1.5*y) * s
//
// Self form: one entity, full derivatives straight into its SelfBlock.
// Cross form: an aliased pair (a == b) with the body written in slot
// notation (a.x * b.x + a.y * b.y ...); the totals must reassemble from
// SelfBlock (dr_a^2 + dr_b^2 parts) + CrossBlock (2*dr_a*dr_b parts).

#[arael::model]
#[arael(constraint(hb, {
    [(p2.x * p2.x + p2.y * p2.y - w.k) * w.isigma,
     (1.5 * p2.y - p2.x) * w.isigma]
}))]
struct P2 {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<P2>,
}

#[arael::model]
struct Q2 {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Q2>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.x * b.x + a.y * b.y - w.k) * w.isigma,
     (a.x + a.y - 2.0 * b.x + 0.5 * b.y) * w.isigma]
}))]
struct PairQ {
    #[arael(ref = root.q2s)]
    a: Ref<Q2>,
    #[arael(ref = root.q2s)]
    b: Ref<Q2>,
    hb: CrossBlock<Q2, Q2>,
}

#[arael::model]
#[arael(root)]
struct W {
    p2s: refs::Vec<P2>,
    q2s: refs::Vec<Q2>,
    pairs: std::vec::Vec<PairQ>,
    k: f64,
    isigma: f64,
}

fn print_h(label: &str, h: &[f64], n: usize) {
    println!("{}:", label);
    for i in 0..n {
        let row: Vec<String> = (0..n).map(|j| format!("{:>10.4}", h[i * n + j])).collect();
        println!("  [{}]", row.join(", "));
    }
}

#[test]
fn aliased_cross_equals_self_formulation() {
    let (x, y) = (0.7, -1.3);

    // Self formulation.
    let mut ws = W {
        p2s: refs::Vec::new(), q2s: refs::Vec::new(),
        pairs: std::vec::Vec::new(), k: 2.0, isigma: 1.3,
    };
    ws.p2s.push(P2 { x: Param::new(x), y: Param::new(y), hb: SelfBlock::new() });
    let mut params_s = Vec::new();
    ws.serialize64(&mut params_s);
    let n = params_s.len();
    let mut g_s = vec![0.0; n];
    let mut h_s = vec![0.0; n * n];
    ws.calc_grad_hessian_dense(&params_s, &mut g_s, &mut h_s);

    // Aliased cross formulation of the same residuals.
    let mut wc = W {
        p2s: refs::Vec::new(), q2s: refs::Vec::new(),
        pairs: std::vec::Vec::new(), k: 2.0, isigma: 1.3,
    };
    wc.q2s.push(Q2 { x: Param::new(x), y: Param::new(y), hb: SelfBlock::new() });
    wc.pairs.push(PairQ { a: Ref::new(0), b: Ref::new(0), hb: CrossBlock::new() });
    let mut params_c = Vec::new();
    wc.serialize64(&mut params_c);
    assert_eq!(params_c.len(), n);
    let mut g_c = vec![0.0; n];
    let mut h_c = vec![0.0; n * n];
    wc.calc_grad_hessian_dense(&params_c, &mut g_c, &mut h_c);

    println!("grad self : {:?}", g_s);
    println!("grad cross: {:?}", g_c);
    print_h("H self  (single entity, SelfBlock only)", &h_s, n);
    print_h("H cross (aliased pair: SelfBlock + CrossBlock)", &h_c, n);

    // The same assembly through the sparse (COO upper-triangle) path.
    let mut gs_coo = vec![0.0; n];
    let mut coo_s = CooMatrix::new(n);
    ws.calc_grad_hessian_sparse(&params_s, &mut gs_coo, &mut coo_s);
    let mut gc_coo = vec![0.0; n];
    let mut coo_c = CooMatrix::new(n);
    wc.calc_grad_hessian_sparse(&params_c, &mut gc_coo, &mut coo_c);

    let dump = |label: &str, coo: &CooMatrix<f64>| {
        println!("{} raw COO triplets (upper triangle, duplicates sum):", label);
        for k in 0..coo.nnz() {
            println!("  ({}, {}) += {:>10.4}", coo.rows[k], coo.cols[k], coo.vals[k]);
        }
    };
    dump("self ", &coo_s);
    dump("cross", &coo_c);
    print_h("H self  via sparse (densified)", &densify_coo(&coo_s, n), n);
    print_h("H cross via sparse (densified)", &densify_coo(&coo_c, n), n);

    // Summation order differs between the paths (triplet order vs write
    // order), so compare with an ulp-scale tolerance instead of bitwise.
    let approx = |a: &[f64], b: &[f64], what: &str| {
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-12, "{} differs at {}: {} vs {}", what, i, a[i], b[i]);
        }
    };
    approx(&densify_coo(&coo_s, n), &h_s, "self COO");
    approx(&densify_coo(&coo_c, n), &h_c, "cross COO");

    // The production pipeline: CSC scatter (first iteration builds the
    // structure from COO; steady state replays via the cached position
    // map). Exercise both variants for both formulations.
    let run_csc = |w: &mut W, params: &[f64], coo: &CooMatrix<f64>, label: &str| {
        let mut csc_direct = coo.to_csc();
        csc_direct.vals.iter_mut().for_each(|v| *v = 0.0);
        let mut g = vec![0.0; n];
        w.calc_grad_hessian_sparse_direct(params, &mut g, &mut csc_direct);
        print_h(&format!("H {} via CSC direct (densified)", label), &densify_csc(&csc_direct), n);

        let (csc, positions) = coo.to_csc_with_map();
        let mut vals = vec![0.0; csc.vals.len()];
        let mut g2 = vec![0.0; n];
        w.calc_grad_hessian_sparse_indexed(params, &mut g2, &mut vals, &positions);
        let csc_indexed = CscMatrix { vals, ..csc };
        print_h(&format!("H {} via CSC indexed (densified)", label), &densify_csc(&csc_indexed), n);
        (densify_csc(&csc_direct), densify_csc(&csc_indexed))
    };
    let (hs_direct, hs_indexed) = run_csc(&mut ws, &params_s, &coo_s, "self ");
    let (hc_direct, hc_indexed) = run_csc(&mut wc, &params_c, &coo_c, "cross");
    approx(&hs_direct, &h_s, "self CSC direct");
    approx(&hs_indexed, &h_s, "self CSC indexed");
    approx(&hc_direct, &h_c, "cross CSC direct");
    approx(&hc_indexed, &h_c, "cross CSC indexed");

    for i in 0..n {
        assert!((g_s[i] - g_c[i]).abs() < 1e-12, "grad[{}]: {} vs {}", i, g_s[i], g_c[i]);
        for j in 0..n {
            assert!((h_s[i * n + j] - h_c[i * n + j]).abs() < 1e-12,
                "H[{},{}]: self={} cross={}", i, j, h_s[i * n + j], h_c[i * n + j]);
        }
    }
}
