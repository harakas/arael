//! Factorization of the Schur-reduced system S: where the 23 ms goes and
//! whether faer can be pushed. Builds S from the slam problem (same path
//! as SparseFaer), then factorizes it every way faer offers --
//! sparse LLT across orderings and supernodal/simplicial regimes, and a
//! dense LLT for the roofline -- reporting one-time and per-iteration
//! cost for each. Every variant's solution is checked against the same
//! reference.
//!
//!     SLAM_POSES=300 ROUNDS=20 cargo run -r --bin sfactor_bench

#[path = "../scene.rs"]
mod scene;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::{block_partition_from_spans, LmProblem, RootProblem};
use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};
use arael_faer::faer;
use arael_faer::faer::dyn_stack::MemStack;
use arael_faer::faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
};
use arael_faer::faer::sparse::linalg::SupernodalThreshold;
use arael_faer::schur::{schur_reduce, schur_symbolic, SchurContext};
use scene::SceneConfig;
use std::time::Instant;

fn pin_single_core() {
    for var in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"] {
        std::env::set_var(var, "1");
    }
    unsafe {
        let core = std::thread::available_parallelism().map(|n| n.get() - 1).unwrap_or(0);
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
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

/// Dump S in Matrix Market symmetric form (lower triangle, 1-based) when
/// SCHUR_DUMP_DIR is set, so the ordering experiments run on exactly the
/// matrix the solver faces. Research scaffolding.
fn dump_s(name: &str, nk: usize, col_ptr: &[usize], row_idx: &[usize], vals: &[f64]) {
    let Ok(dir) = std::env::var("SCHUR_DUMP_DIR") else { return };
    use std::io::Write;
    let path = format!("{}/{}.mtx", dir, name);
    let f = std::fs::File::create(&path).expect("create mtx");
    let mut w = std::io::BufWriter::new(f);
    writeln!(w, "%%MatrixMarket matrix coordinate real symmetric").unwrap();
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

fn main() {
    pin_single_core();
    let mut cfg = SceneConfig::default();
    if let Ok(n) = std::env::var("SLAM_POSES") {
        if let Ok(n) = n.parse() {
            cfg.num_poses = n;
            cfg.num_landmarks = 4 * n;
        }
    }
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let scene = scene::generate(&cfg);

    // ---- build S exactly as the solver backend does ----------------------
    let mut path = arael_runner::build(&scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    let n = params.len();
    let mut spans = Vec::new();
    path.collect_param_block_spans(&mut spans);
    let partition = block_partition_from_spans(&spans, n);
    let mut cells: Vec<(u32, u32)> = Vec::new();
    arael::model::Model::collect_hessian_cells64(&path, &mut cells);
    let (hsym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition.clone(),
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut resolver = PositionResolver::new(&hsym);
    let mut positions: Vec<usize> = Vec::new();
    arael::model::Model::accumulate_hessian_positions64(
        &path,
        &mut |i, j| resolver.resolve(i as usize, j as usize),
        &mut positions,
    );
    let mut h = SparseBlockColMat::<usize, f64>::zeroed(hsym);
    let mut grad = vec![0.0; n];
    path.calc_grad_hessian_sparse_indexed(&params, &mut grad, h.vals_mut(), &positions);
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
        (0..nblk)
            .filter(|&b| {
                ranges.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end)
            })
            .collect()
    };
    let hint = RootProblem::marginalize_hint(&path);
    let candidates: Vec<Vec<usize>> = if hint.is_empty() {
        LmProblem::marginalize_candidates(&path).iter().map(|r| blocks_in(r)).collect()
    } else {
        vec![blocks_in(&hint)]
    };
    let eliminated: Vec<usize> = candidates
        .into_iter()
        .filter(|ids| !ids.is_empty())
        .max_by_key(|ids| {
            ids.iter().map(|&b| partition[b + 1] - partition[b]).sum::<usize>()
        })
        .expect("nothing marginalizable in the slam scene");
    let sym = schur_symbolic(h.symbolic(), &eliminated).unwrap();
    let mut s = sym.alloc_s::<f64>();
    let mut ctx = SchurContext::new();
    let nk = sym.s.nrows();
    let mut rhs = vec![0.0; nk];
    schur_reduce(&sym, &h, &grad, &mut ctx, &mut s, &mut rhs).unwrap();

    if std::env::var("SCHUR_DUMP_DIR").is_ok() {
        let (cp, ri) = sym.s.csc_pattern();
        let mut vals = vec![0.0f64; cp[nk]];
        s.csc_vals_into(&mut vals);
        let poses = std::env::var("SLAM_POSES").unwrap_or_else(|_| "300".into());
        dump_s(&format!("slam-{}", poses), nk, &cp, &ri, &vals);
        // Dumping is all we were asked for. Everything below allocates dense
        // n x n matrices, which is not survivable at large n (see the guard).
        return;
    }

    let s_csc = s.to_csc();
    let s_nnz = s_csc.compute_nnz();
    let dense_upper = nk * (nk + 1) / 2;
    println!("S: n = {}, stored upper nnz = {} ({:.1}% of the dense upper triangle)",
        nk, s_nnz, 100.0 * s_nnz as f64 / dense_upper as f64);
    println!("   dense-Cholesky flop count for this n: {:.2} GFLOP (n^3/3)",
        (nk as f64).powi(3) / 3.0 / 1e9);
    println!("rounds: {} (min reported)", rounds);
    println!();

    // reference solution: dense LLT of the mirrored S.
    //
    // This is two n x n dense matrices. At 300 poses (n = 1800) that is 26 MB
    // each; at 6000 poses (n = 36000) it is 10.4 GB each, and the process gets
    // OOM-killed -- which is exactly what happened. The dense route is a
    // roofline for small S, so bound it rather than let it take the machine
    // down.
    const DENSE_MAX: usize = 8000; // 8000^2 * 8 B = 512 MB per matrix
    if nk > DENSE_MAX {
        println!();
        println!("dense reference SKIPPED: S is {} x {}, and the dense route needs two",
            nk, nk);
        println!("  {} x {} matrices = {:.1} GB each. Raise DENSE_MAX if you mean it.",
            nk, nk, (nk as f64).powi(2) * 8.0 / 1e9);
        return;
    }
    let sd = s.to_dense();
    let mut full = faer::Mat::<f64>::zeros(nk, nk);
    for j in 0..nk {
        for i in 0..nk {
            full[(i, j)] = if sd[(i, j)] != 0.0 { sd[(i, j)] } else { sd[(j, i)] };
        }
    }
    let x_ref = {
        use faer::prelude::Solve;
        let mut b = faer::Mat::<f64>::zeros(nk, 1);
        for i in 0..nk {
            b[(i, 0)] = rhs[i];
        }
        let llt = full.llt(faer::Side::Lower).expect("S SPD");
        llt.solve(&b)
    };
    let check = |x: &[f64], label: &str| {
        let mut max_rel = 0.0f64;
        for i in 0..nk {
            max_rel = max_rel.max((x[i] - x_ref[(i, 0)]).abs() / (1.0 + x_ref[(i, 0)].abs()));
        }
        assert!(max_rel < 1e-8, "{}: solution off by {:.2e}", label, max_rel);
        max_rel
    };

    // ---- sparse LLT variants ---------------------------------------------
    println!("faer sparse LLT on S             symbolic   L vals    L MB   numeric    solve   max rel");
    // control: a shuffled permutation. if the ordering is really applied,
    // this must inflate the factor -- proving AMD and natural coinciding
    // is a property of S, not faer ignoring the argument.
    let mut fwd: Vec<usize> = (0..nk).collect();
    let mut state = 0x9e3779b97f4a7c15u64;
    for i in (1..nk).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        fwd.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let mut inv = vec![0usize; nk];
    for (new_i, &old_i) in fwd.iter().enumerate() {
        inv[old_i] = new_i;
    }
    let shuffled = faer::perm::PermRef::new_checked(&fwd, &inv, nk);

    let variants: Vec<(&str, SymmetricOrdering<usize>, SupernodalThreshold)> = vec![
        ("AMD, auto (shipped)", SymmetricOrdering::Amd, SupernodalThreshold::AUTO),
        ("AMD, force supernodal", SymmetricOrdering::Amd, SupernodalThreshold::FORCE_SUPERNODAL),
        ("AMD, force simplicial", SymmetricOrdering::Amd, SupernodalThreshold::FORCE_SIMPLICIAL),
        ("natural, auto", SymmetricOrdering::Identity, SupernodalThreshold::AUTO),
        ("natural, force supernodal", SymmetricOrdering::Identity, SupernodalThreshold::FORCE_SUPERNODAL),
        ("shuffled (control), auto", SymmetricOrdering::Custom(shuffled), SupernodalThreshold::AUTO),
    ];
    for (label, ordering, threshold) in variants {
        let params_c = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: threshold,
            ..Default::default()
        };
        let (t_sym, llt_sym) = min_ms(rounds, || {
            factorize_symbolic_cholesky(
                s_csc.as_ref().symbolic(),
                faer::Side::Upper,
                ordering,
                params_c,
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
        let mut solve_mem = vec![
            std::mem::MaybeUninit::<u8>::uninit();
            llt_sym.solve_in_place_scratch::<f64>(1, faer::Par::Seq).unaligned_bytes_required()
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
        let stack = MemStack::new(&mut factor_mem);
        let llt = llt_sym
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
        let mut x = vec![0.0f64; nk];
        let (t_solve, _) = min_ms(rounds, || {
            x.copy_from_slice(&rhs);
            let stack = MemStack::new(&mut solve_mem);
            llt.solve_in_place_with_conj(
                faer::Conj::No,
                faer::col::ColMut::from_slice_mut(&mut x).as_mat_mut(),
                faer::Par::Seq,
                stack,
            );
        });
        let rel = check(&x, label);
        println!(
            "  {:28} {:7.2}  {:>8}  {:5.1}  {:8.2} {:8.3}  {:8.1e}",
            label,
            t_sym,
            l_vals.len(),
            l_vals.len() as f64 * 8.0 / 1e6,
            t_num,
            t_solve,
            rel
        );
    }

    // ---- the auto-detector's question ------------------------------------
    // Is "eliminate the landmarks first" a better ELIMINATION ORDERING than
    // letting AMD order the whole system? Compare the fill each leaves,
    // symbolically. Each route needs its own symbolic anyway, so the
    // comparison costs exactly one extra symbolic -- the route not taken.
    {
        let h_csc = h.to_csc();
        let (t_hsym, h_llt) = min_ms(rounds, || {
            factorize_symbolic_cholesky(
                h_csc.as_ref().symbolic(),
                faer::Side::Upper,
                SymmetricOrdering::Amd,
                CholeskySymbolicParams::default(),
            )
            .unwrap()
        });
        let s_llt = factorize_symbolic_cholesky(
            s_csc.as_ref().symbolic(),
            faer::Side::Upper,
            SymmetricOrdering::Identity,
            CholeskySymbolicParams::default(),
        )
        .unwrap();
        println!();
        println!("elimination-ordering comparison (symbolic only):");
        println!("  full system, AMD:        L = {:>10} vals ({:6.1} MB), symbolic {:6.2} ms",
            h_llt.len_val(), h_llt.len_val() as f64 * 8.0 / 1e6, t_hsym);
        println!("  landmarks first (S):     L = {:>10} vals ({:6.1} MB)",
            s_llt.len_val(), s_llt.len_val() as f64 * 8.0 / 1e6);
        println!("  fill ratio L_S / L_H:    {:.2}", s_llt.len_val() as f64 / h_llt.len_val() as f64);
    }

    // ---- dense LLT --------------------------------------------------------
    println!();
    println!("dense LLT on S (n = {}, {:.1} MB dense)", nk, (nk * nk) as f64 * 8.0 / 1e6);
    // per-iteration cost: scatter the block S into a dense matrix, factor
    // in place, solve. faer's dense LLT reads the LOWER triangle, and S is
    // stored upper, so each stored value lands transposed -- which fills
    // the lower triangle exactly, no separate mirror pass.
    let mut dense = faer::Mat::<f64>::zeros(nk, nk);
    let (t_fill, _) = min_ms(rounds, || {
        dense.fill(0.0);
        let ssym = sym.s.clone();
        for j in 0..ssym.nblk_cols() {
            let cols = ssym.col_span(j);
            for b in ssym.col_range(j) {
                let rows = ssym.row_span(ssym.blk_row(b));
                let vals = &s.vals()[ssym.val_range(b)];
                let rw = rows.len();
                for (jj, cj) in cols.clone().enumerate() {
                    for (ii, ri) in rows.clone().enumerate() {
                        dense[(cj, ri)] = vals[ii + jj * rw];
                    }
                }
            }
        }
    });
    let mut work = faer::Mat::<f64>::zeros(nk, nk);
    let factor_bytes = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
        nk,
        faer::Par::Seq,
        faer::Spec::default(),
    )
    .unaligned_bytes_required();
    let mut dmem = vec![std::mem::MaybeUninit::<u8>::uninit(); factor_bytes];
    let (t_dfactor, _) = min_ms(rounds, || {
        work.copy_from(&dense);
        let stack = MemStack::new(&mut dmem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            work.as_mut(),
            faer::linalg::cholesky::llt::factor::LltRegularization::default(),
            faer::Par::Seq,
            stack,
            faer::Spec::default(),
        )
        .expect("dense LLT");
    });
    // factor once more for the solve (work now holds L in its lower part)
    work.copy_from(&dense);
    {
        let stack = MemStack::new(&mut dmem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            work.as_mut(),
            faer::linalg::cholesky::llt::factor::LltRegularization::default(),
            faer::Par::Seq,
            stack,
            faer::Spec::default(),
        )
        .expect("dense LLT");
    }
    let mut xd = faer::Mat::<f64>::zeros(nk, 1);
    let (t_dsolve, _) = min_ms(rounds, || {
        for i in 0..nk {
            xd[(i, 0)] = rhs[i];
        }
        faer::linalg::cholesky::llt::solve::solve_in_place(
            work.as_ref(),
            xd.as_mut(),
            faer::Par::Seq,
            MemStack::new(&mut dmem),
        );
    });
    let xdv: Vec<f64> = (0..nk).map(|i| xd[(i, 0)]).collect();
    let rel = check(&xdv, "dense");
    println!(
        "  {:28} {:7} {:>9}  {:5.1}  {:8.2} {:8.3}  {:8.1e}",
        "upper fill + in-place LLT",
        "-",
        nk * nk,
        (nk * nk) as f64 * 8.0 / 1e6,
        t_dfactor,
        t_dsolve,
        rel
    );
    println!("  (dense fill from block S: {:.2} ms/iter)", t_fill);
    println!(
        "  effective dense rate: {:.1} GFLOP/s",
        (nk as f64).powi(3) / 3.0 / 1e9 / (t_dfactor / 1e3)
    );
}
