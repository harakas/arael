// Phase-1b prototype: DOF via shift-inverted block subspace iteration
// (SKETCH.md section 5). Finds the near-null subspace of the
// column-normalised Jacobian with a sparse Cholesky of H + lambda*I,
// then takes the rank decision from the singular values of J*V -- the
// same metric as the dense-SVD path. Validates answer and timing
// against the dense SVD.
//
// Usage: cargo run -r -p arael-sketch-solver --example dof_iter -- \
//     <sketch.json> [block_k] [lambda] [sweeps]

use arael::model::JacobianModel;
use arael::simple_lm::RootProblem;
use arael_sketch_solver::Sketch;
use faer::sparse::linalg::cholesky as fchol;
use std::mem::MaybeUninit;

// Mirror of compute_dof's rank_from_svs gap logic (ascending input).
fn rank_cut(sorted: &[f64]) -> (usize, f64) {
    let max_sv = sorted.last().copied().unwrap_or(0.0);
    let upper_bound = max_sv * 0.01;
    let floor = max_sv * 1e-20;
    let mut best_gap = 0.0f64;
    let mut best_cut = 0;
    for i in 0..sorted.len().saturating_sub(1) {
        let lo = sorted[i].max(floor);
        let hi = sorted[i + 1].max(floor);
        if lo > upper_bound { break; }
        let gap = hi / lo;
        if gap > best_gap {
            best_gap = gap;
            best_cut = i + 1;
        }
    }
    if best_gap < 1e3 {
        best_cut = sorted.iter().filter(|&&v| v < 1e-15).count();
    }
    (best_cut, best_gap)
}

fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: dof_iter <sketch.json> [block_k] [lambda] [sweeps]");
        std::process::exit(2);
    }
    let path = &args[0];
    let k_arg: Option<usize> = args.get(1).map(|s| s.parse().expect("block_k"));
    let lambda: f64 = args.get(2).map(|s| s.parse().expect("lambda")).unwrap_or(1e-10);
    let sweeps: usize = args.get(3).map(|s| s.parse().expect("sweeps")).unwrap_or(2);

    let json = std::fs::read_to_string(path).expect("read sketch file");
    let mut sketch: Sketch = serde_json::from_str(&json).expect("parse sketch json");

    sketch.prepare_expr_constraints();
    sketch.update_tangent_flags();
    sketch.update_perpendicular_flags();
    sketch.update_line_dir_flags();
    let saved_drift = sketch.drift_isigma;
    sketch.drift_isigma = 0.0;
    let mut params = Vec::new();
    sketch.serialize(&mut params);
    let n = params.len();
    let mut jacobian = sketch.calc_jacobian(&params);
    sketch.drift_isigma = saved_drift;
    jacobian.rows.retain(|r| r.label != "range");
    let m = jacobian.num_residuals();

    // Reference: the dense path.
    let t = std::time::Instant::now();
    let mut svs_ref: Vec<f64> = jacobian.singular_values_column_normalised();
    let t_dense = t.elapsed();
    svs_ref.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap());
    let (cut_ref, _) = rank_cut(&svs_ref);
    let dof_ref = n.saturating_sub(svs_ref.len() - cut_ref);

    // Column scales, as in singular_values_column_normalised.
    let col_norms = jacobian.column_l2_norms();
    let scale: Vec<f64> = col_norms.iter().map(|c: &f64| c.max(1e-15)).collect();

    let k = k_arg.unwrap_or(dof_ref + 16).min(n);

    // Assemble H = Jn^T Jn + lambda*I as a full symmetric CSC.
    let t = std::time::Instant::now();
    let mut upper: std::collections::HashMap<(u32, u32), f64> = std::collections::HashMap::new();
    for row in &jacobian.rows {
        for (ai, &(ia, va)) in row.entries.iter().enumerate() {
            let van = va / scale[ia as usize];
            for &(ib, vb) in &row.entries[ai..] {
                let vbn = vb / scale[ib as usize];
                let key = if ia <= ib { (ia, ib) } else { (ib, ia) };
                *upper.entry(key).or_insert(0.0) += van * vbn;
            }
        }
    }
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (&(i, j), &v) in &upper {
        let (i, j) = (i as usize, j as usize);
        cols[j].push((i, v));
        if i != j {
            cols[i].push((j, v));
        }
    }
    for (j, col) in cols.iter_mut().enumerate() {
        if !col.iter().any(|&(i, _)| i == j) {
            col.push((j, 0.0));
        }
        col.sort_by_key(|&(i, _)| i);
    }
    let mut col_ptr: Vec<usize> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    col_ptr.push(0);
    for (j, col) in cols.iter().enumerate() {
        for &(i, v) in col {
            row_idx.push(i);
            vals.push(if i == j { v + lambda } else { v });
        }
        col_ptr.push(row_idx.len());
    }
    let t_assemble = t.elapsed();

    // Sparse Cholesky of H + lambda*I.
    let t = std::time::Instant::now();
    let sym_ref = faer::sparse::SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
    let symbolic = fchol::factorize_symbolic_cholesky(
        sym_ref,
        faer::Side::Upper,
        fchol::SymmetricOrdering::Amd,
        fchol::CholeskySymbolicParams::default(),
    )
    .expect("symbolic cholesky");
    let mut l_vals = vec![0.0f64; symbolic.len_val()];
    let factor_req = symbolic.factorize_numeric_llt_scratch::<f64>(faer::Par::Seq, faer::Spec::default());
    let mut factor_mem: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); factor_req.unaligned_bytes_required()];
    let stack = faer::dyn_stack::MemStack::new(&mut factor_mem);
    let mat_ref = faer::sparse::SparseColMatRef::new(sym_ref, &vals);
    symbolic
        .factorize_numeric_llt(
            &mut l_vals,
            mat_ref,
            faer::Side::Upper,
            faer::linalg::cholesky::llt::factor::LltRegularization::default(),
            faer::Par::Seq,
            stack,
            faer::Spec::default(),
        )
        .expect("numeric cholesky");
    let t_factor = t.elapsed();

    // Block inverse iteration: V <- orth((H + lambda*I)^-1 V).
    let t = std::time::Instant::now();
    let mut v = vec![0.0f64; n * k];
    let mut rng = 0x9e3779b97f4a7c15u64;
    for x in v.iter_mut() {
        *x = xorshift(&mut rng);
    }
    let llt = fchol::LltRef::new(&symbolic, &l_vals);
    let solve_req = symbolic.solve_in_place_scratch::<f64>(k, faer::Par::Seq);
    let mut solve_mem: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); solve_req.unaligned_bytes_required()];
    for _ in 0..sweeps {
        let stack = faer::dyn_stack::MemStack::new(&mut solve_mem);
        let rhs = faer::mat::MatMut::from_column_major_slice_mut(&mut v, n, k);
        llt.solve_in_place_with_conj(faer::Conj::No, rhs, faer::Par::Seq, stack);
        // Modified Gram-Schmidt orthonormalization.
        for c in 0..k {
            let (head, tail) = v.split_at_mut(c * n);
            let vc = &mut tail[..n];
            for p in 0..c {
                let vp = &head[p * n..(p + 1) * n];
                let dot: f64 = vp.iter().zip(vc.iter()).map(|(a, b)| a * b).sum();
                for i in 0..n {
                    vc[i] -= dot * vp[i];
                }
            }
            let norm: f64 = vc.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-300 {
                for x in vc.iter_mut() {
                    *x /= norm;
                }
            } else {
                for x in vc.iter_mut() {
                    *x = xorshift(&mut rng);
                }
            }
        }
    }
    let t_iter = t.elapsed();

    // Decision in the J metric: singular values of B = Jn * V (m x k).
    let t = std::time::Instant::now();
    let mut b = vec![0.0f64; m * k];
    for (r, row) in jacobian.rows.iter().enumerate() {
        for &(ci, val) in &row.entries {
            let van = val / scale[ci as usize];
            let base = ci as usize;
            for c in 0..k {
                b[c * m + r] += van * v[c * n + base];
            }
        }
    }
    let bmat = nalgebra::DMatrix::from_column_slice(m, k, &b);
    let mut svs_b: Vec<f64> = bmat.singular_values().iter().cloned().collect();
    svs_b.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap());
    let (cut_b, gap_b) = rank_cut(&svs_b);
    let t_proj = t.elapsed();
    let dof_est = cut_b;

    let total = t_assemble + t_factor + t_iter + t_proj;
    println!("== {}", path);
    println!("m={} n={} k={} lambda={:.1e} sweeps={}", m, n, k, lambda, sweeps);
    println!("dense svd: dof={} in {:?}", dof_ref, t_dense);
    println!(
        "iter:      dof={} in {:?}  (assemble={:?} factor={:?} sweeps={:?} decide={:?})",
        dof_est, total, t_assemble, t_factor, t_iter, t_proj
    );
    println!("boundary:  gap={:.3e}  contained={}  match={}", gap_b, dof_est < k, dof_est == dof_ref);
    let lo = cut_b.saturating_sub(3);
    let hi = (cut_b + 3).min(svs_b.len());
    for i in lo..hi {
        let side = if i < cut_b { "zero" } else { "real" };
        println!("  svB[{:4}] = {:.6e}  [{}]", i, svs_b[i], side);
    }
    if dof_est != dof_ref {
        println!("MISMATCH: dense={} iter={}", dof_ref, dof_est);
        std::process::exit(1);
    }
}
