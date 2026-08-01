//! Blocked Schur reduction cost on the slam problem: one-time symbolic
//! analysis, per-iteration schur_reduce (S + reduced rhs), and the
//! S -> scalar CSC handoff, on the block-CSC H the two-scan assembly
//! builds. Single core, min over ROUNDS (default 20). At 60 poses the
//! result is verified against a dense reference reduction.
//!
//!     SLAM_POSES=300 ROUNDS=20 cargo run -r --bin schur_bench

#[path = "../scene.rs"]
mod scene;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::{block_partition_from_spans, LmProblem, RootProblem};
use arael_faer::bsc::{PositionResolver, SparseBlockColMat, SymbolicSparseBlockColMat};
use arael_faer::schur::{schur_backsub, schur_reduce, schur_symbolic, SchurContext};
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

    let mut path = arael_runner::build(&scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize(&mut params);
    let n = params.len();

    // block H via the two-scan assembly (as SparseFaer's fast path)
    let mut spans = Vec::new();
    path.collect_param_block_spans(&mut spans);
    let partition = block_partition_from_spans(&spans, n);
    let mut cells: Vec<(u32, u32)> = Vec::new();
    arael::model::Model::collect_hessian_cells(&path, &mut cells);
    let (hsym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
        partition.clone(),
        partition.clone(),
        cells.len(),
        |k| (cells[k].0 as usize, cells[k].1 as usize),
    );
    let mut resolver = PositionResolver::new(&hsym);
    let mut positions: Vec<arael::ValueIndex> = Vec::new();
    arael::model::Model::bind_hessian_positions(
        &mut path,
        &mut arael::model::HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i as usize, j as usize)),
        &mut positions,
    );
    let mut h = SparseBlockColMat::<usize, f64>::zeroed(hsym);
    let mut grad = vec![0.0; n];
    path.calc_grad_hessian_sparse_indexed(&params, &mut grad, h.vals_mut(), &positions);

    // What the solver marginalizes: the model names nothing, so take the
    // coupling graph's candidates and pick the biggest, exactly as SparseFaer
    // does (the same derivation bal's schur_stats uses).
    let nblk = partition.len() - 1;
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
        .max_by_key(|ids| ids.iter().map(|&b| partition[b + 1] - partition[b]).sum::<usize>())
        .expect("nothing marginalizable in the slam scene");

    // -- symbolic (one-time) -------------------------------------------
    let (t_sym, sym) = min_ms(rounds, || schur_symbolic(h.symbolic(), &eliminated).unwrap());

    // -- numeric reduce (per iteration and per damping retry) ----------
    let mut s = sym.alloc_s::<f64>();
    let mut ctx = SchurContext::new();
    let mut rhs_out = vec![0.0; s.symbolic().nrows()];
    let (t_reduce, _) = min_ms(rounds, || {
        schur_reduce(&sym, &h, &grad, &mut ctx, &mut s, &mut rhs_out).unwrap()
    });

    // -- S -> scalar CSC handoff ----------------------------------------
    let (t_csc, s_csc) = min_ms(rounds, || s.to_csc());

    println!(
        "scene: {} poses, {} params; H {} blocks / {} vals; eliminated {} blocks",
        cfg.num_poses, n, h.symbolic().nblocks(), h.symbolic().val_count(), eliminated.len(),
    );
    println!(
        "S: {} kept blocks ({} scalar), {} tiles / {} vals ({} scalar nnz), {} pair contributions",
        sym.kept.len(), sym.s.nrows(), sym.s.nblocks(), sym.s.val_count(),
        s_csc.compute_nnz(), sym.pair_count(),
    );
    println!("rounds: {} (min reported)                          ms", rounds);
    println!("  schur_symbolic (one-time)                    {:7.3}", t_sym);
    println!("  schur_reduce (S + rhs, per iteration)        {:7.3}", t_reduce);
    println!("  S.to_csc (handoff to scalar factorization)   {:7.3}", t_csc);

    // instrumented pass: per-stage split of the best round (timing
    // laps add a little overhead, so the headline min above stays
    // uninstrumented)
    ctx.enable_timing();
    let mut best: Option<arael_faer::schur::SchurTiming> = None;
    for _ in 0..rounds {
        schur_reduce(&sym, &h, &grad, &mut ctx, &mut s, &mut rhs_out).unwrap();
        let t = ctx.timing().unwrap().clone();
        if best.as_ref().is_none_or(|b| t.total() < b.total()) {
            best = Some(t);
        }
    }
    let t = best.unwrap();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    println!("  reduce stages (best instrumented round, total {:.3} ms):", ms(t.total()));
    println!("    seed   (zero S + Hkk/rhs copy)             {:7.3}", ms(t.seed));
    println!("    factor (D_e Cholesky)                      {:7.3}", ms(t.factor));
    println!("    panel  (gather + Z = D^-1 [C^T|b])         {:7.3}", ms(t.panel));
    println!("    gemm   (pair contributions into S)         {:7.3}", ms(t.gemm));
    println!("    rhs    (observer rhs updates)              {:7.3}", ms(t.rhs));
    println!("    finish (re-zero diag lower)                {:7.3}", ms(t.finish));

    // -- factor + solve the reduced system --------------------------------
    // faer sparse LLT on S's scalar CSC, mirroring the SparseFaer
    // backend's calls: the symbolic factorization runs once on the
    // first iteration (in the future solver backend it lives next to
    // SchurSymbolic in the solver state); every iteration then pays
    // numeric factorization + triangular solve.
    {
        use arael_faer::faer::dyn_stack::MemStack;
        use arael_faer::faer::sparse::linalg::cholesky::{
            factorize_symbolic_cholesky, CholeskySymbolicParams, SymmetricOrdering,
        };
        use arael_faer::faer;

        let nk = s.symbolic().nrows();
        let (t_llt_sym, llt_sym) = min_ms(rounds, || {
            factorize_symbolic_cholesky(
                s_csc.as_ref().symbolic(),
                faer::Side::Upper,
                SymmetricOrdering::Amd,
                CholeskySymbolicParams::default(),
            )
            .unwrap()
        });

        let mut l_vals = vec![0.0f64; llt_sym.len_val()];
        let factor_bytes = llt_sym
            .factorize_numeric_llt_scratch::<f64>(faer::Par::Seq, faer::Spec::default())
            .unaligned_bytes_required();
        let mut factor_mem = vec![std::mem::MaybeUninit::<u8>::uninit(); factor_bytes];
        let solve_bytes = llt_sym
            .solve_in_place_scratch::<f64>(1, faer::Par::Seq)
            .unaligned_bytes_required();
        let mut solve_mem = vec![std::mem::MaybeUninit::<u8>::uninit(); solve_bytes];

        let (t_factor, _) = min_ms(rounds, || {
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
            x.copy_from_slice(&rhs_out);
            let stack = MemStack::new(&mut solve_mem);
            llt.solve_in_place_with_conj(
                faer::Conj::No,
                faer::col::ColMut::from_slice_mut(&mut x).as_mat_mut(),
                faer::Par::Seq,
                stack,
            );
        });

        // recover the eliminated blocks (landmarks)
        let mut x_full = vec![0.0f64; n];
        let (t_back, _) = min_ms(rounds, || {
            schur_backsub(&sym, &h, &grad, &x, &mut ctx, &mut x_full).unwrap()
        });

        println!();
        println!("reduced system S (n = {}): faer sparse LLT        ms", nk);
        println!("  symbolic factorization (first iteration only){:7.3}", t_llt_sym);
        println!("  numeric factorization (per iteration)        {:7.3}", t_factor);
        println!("  triangular solve (per iteration)             {:7.3}", t_solve);
        println!("  back-substitution (per iteration)            {:7.3}", t_back);
        println!(
            "  Schur route 1st iter: reduce {:.1} + csc {:.1} + sym {:.1} + factor {:.1} + solve {:.1} + back {:.1} = {:.1}",
            t_reduce, t_csc, t_llt_sym, t_factor, t_solve, t_back,
            t_reduce + t_csc + t_llt_sym + t_factor + t_solve + t_back
        );
        println!(
            "  Schur route steady state: {:.1}",
            t_reduce + t_csc + t_factor + t_solve + t_back
        );

        // reduced-solve validation against a dense solve
        if nk <= 2200 {
            let sd = s.to_dense();
            let mut fullm = faer::Mat::<f64>::zeros(nk, nk);
            for j in 0..nk {
                for i in 0..nk {
                    fullm[(i, j)] = if sd[(i, j)] != 0.0 { sd[(i, j)] } else { sd[(j, i)] };
                }
            }
            use arael_faer::faer::prelude::Solve;
            let dl = fullm.llt(faer::Side::Lower).expect("S SPD");
            let mut xr = faer::Mat::<f64>::zeros(nk, 1);
            for i in 0..nk {
                xr[(i, 0)] = rhs_out[i];
            }
            let xd = dl.solve(&xr);
            let mut max_rel = 0.0f64;
            for i in 0..nk {
                max_rel = max_rel.max((x[i] - xd[(i, 0)]).abs() / (1.0 + xd[(i, 0)].abs()));
            }
            assert!(max_rel < 1e-8, "reduced solve vs dense: {:.2e}", max_rel);
            println!("  reduced solve matches dense: {:.2e} max rel", max_rel);

            // full-route identity: reduce -> solve S -> backsub vs a
            // dense solve of the full mirrored system
            let hd = h.to_dense();
            let mut hfull = faer::Mat::<f64>::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    hfull[(i, j)] = if hd[(i, j)] != 0.0 { hd[(i, j)] } else { hd[(j, i)] };
                }
            }
            let dl = hfull.llt(faer::Side::Lower).expect("H SPD");
            let mut br = faer::Mat::<f64>::zeros(n, 1);
            for i in 0..n {
                br[(i, 0)] = grad[i];
            }
            let xd_full = dl.solve(&br);
            let mut max_full = 0.0f64;
            for i in 0..n {
                max_full = max_full.max(
                    (x_full[i] - xd_full[(i, 0)]).abs() / (1.0 + xd_full[(i, 0)].abs()),
                );
            }
            assert!(max_full < 1e-8, "full route vs dense: {:.2e}", max_full);
            println!("  full solution (kept + landmarks) matches dense: {:.2e} max rel", max_full);
        }
    }

    // -- transposed-orientation variant ---------------------------------
    // the same system with block ids permuted landmarks-first: every
    // coupling tile is then stored (elim, kept), the obs_trans = true
    // orientation. kept blocks keep their relative order, so S and the
    // reduced rhs must match the direct-orientation results (up to fp
    // association differences between the two kernel forms).
    {
        let nblk = partition.len() - 1;
        let ne = eliminated.len();
        let mut is_elim = vec![false; nblk];
        for &e in &eliminated {
            is_elim[e] = true;
        }
        let mut perm = vec![0usize; nblk]; // old block id -> new
        let mut next = 0;
        for b in 0..nblk {
            if is_elim[b] {
                perm[b] = next;
                next += 1;
            }
        }
        for b in 0..nblk {
            if !is_elim[b] {
                perm[b] = next;
                next += 1;
            }
        }
        let mut inv = vec![0usize; nblk];
        for b in 0..nblk {
            inv[perm[b]] = b;
        }
        let mut part2 = vec![0usize; nblk + 1];
        for nb in 0..nblk {
            part2[nb + 1] = part2[nb] + (partition[inv[nb] + 1] - partition[inv[nb]]);
        }

        let hsym = h.symbolic();
        let mut coords: Vec<(usize, usize)> = Vec::new();
        for c in 0..nblk {
            for b in hsym.col_range(c) {
                let (nr, nc) = (perm[hsym.blk_row(b)], perm[c]);
                let (lo, hi) = (nr.min(nc), nr.max(nc));
                coords.push((part2[lo], part2[hi]));
            }
        }
        let (sym2h, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            part2.clone(),
            part2.clone(),
            coords.len(),
            |k| coords[k],
        );
        let mut h2 = SparseBlockColMat::<usize, f64>::zeroed(sym2h);
        for c in 0..nblk {
            for b in hsym.col_range(c) {
                let r = hsym.blk_row(b);
                let (wr, wc) = (partition[r + 1] - partition[r], partition[c + 1] - partition[c]);
                let (nr, nc) = (perm[r], perm[c]);
                let src = h.vals()[hsym.val_range(b)].to_vec();
                let (tr, tc, flip) = if nr <= nc { (nr, nc, false) } else { (nc, nr, true) };
                let nb = h2.symbolic().col_range(tc)
                    .find(|&x| h2.symbolic().blk_row(x) == tr)
                    .unwrap();
                let drange = h2.symbolic().val_range(nb);
                let dstv = &mut h2.vals_mut()[drange];
                if !flip {
                    dstv.copy_from_slice(&src);
                } else {
                    for j in 0..wc {
                        for i in 0..wr {
                            dstv[j + i * wc] = src[i + j * wr];
                        }
                    }
                }
            }
        }
        let mut grad2 = vec![0.0; n];
        for b in 0..nblk {
            let w = partition[b + 1] - partition[b];
            grad2[part2[perm[b]]..part2[perm[b]] + w]
                .copy_from_slice(&grad[partition[b]..partition[b] + w]);
        }

        let elim2: Vec<usize> = (0..ne).collect();
        let (t_sym2, sym2) = min_ms(rounds, || schur_symbolic(h2.symbolic(), &elim2).unwrap());
        let mut s2 = sym2.alloc_s::<f64>();
        let mut ctx2 = SchurContext::new();
        let mut rk2 = vec![0.0; s2.symbolic().nrows()];
        let (t_red2, _) = min_ms(rounds, || {
            schur_reduce(&sym2, &h2, &grad2, &mut ctx2, &mut s2, &mut rk2).unwrap()
        });

        assert_eq!(s2.vals().len(), s.vals().len());
        let mut max_rel = 0.0f64;
        for (a, b) in std::iter::zip(s2.vals(), s.vals()) {
            max_rel = max_rel.max((a - b).abs() / (1.0 + b.abs()));
        }
        let mut max_rhs = 0.0f64;
        for (a, b) in std::iter::zip(&rk2, &rhs_out) {
            max_rhs = max_rhs.max((a - b).abs() / (1.0 + b.abs()));
        }
        assert!(max_rel < 1e-9 && max_rhs < 1e-9, "S {:.2e} rhs {:.2e}", max_rel, max_rhs);
        println!();
        println!("landmarks-first permutation (every coupling tile transposed):");
        println!("  schur_symbolic (one-time)                    {:7.3}", t_sym2);
        println!("  schur_reduce (S + rhs, per iteration)        {:7.3}", t_red2);
        println!("  S / rhs match direct orientation: {:.2e} / {:.2e} max rel", max_rel, max_rhs);
    }

    // -- dense verification at small sizes ------------------------------
    if n <= 2200 {
        let d = h.to_dense();
        let mut full = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                full[i + j * n] = if d[(i, j)] != 0.0 { d[(i, j)] } else { d[(j, i)] };
            }
        }
        let is_elim: Vec<bool> = {
            let mut v = vec![false; nblk];
            for &e in &eliminated {
                v[e] = true;
            }
            v
        };
        let keep_idx: Vec<usize> = (0..nblk)
            .filter(|&b| !is_elim[b])
            .flat_map(|b| partition[b]..partition[b + 1])
            .collect();
        let elim_idx: Vec<usize> = (0..nblk)
            .filter(|&b| is_elim[b])
            .flat_map(|b| partition[b]..partition[b + 1])
            .collect();
        let (nk, ne) = (keep_idx.len(), elim_idx.len());

        // dense reference S = Hkk - Hke Hee^-1 Hek via faer's LLT
        let mut hee = arael_faer::faer::Mat::<f64>::zeros(ne, ne);
        for (a, &i) in elim_idx.iter().enumerate() {
            for (b, &j) in elim_idx.iter().enumerate() {
                hee[(a, b)] = full[i + j * n];
            }
        }
        let mut hek = arael_faer::faer::Mat::<f64>::zeros(ne, nk + 1);
        for (c, &j) in keep_idx.iter().enumerate() {
            for (r, &i) in elim_idx.iter().enumerate() {
                hek[(r, c)] = full[i + j * n];
            }
        }
        for (r, &i) in elim_idx.iter().enumerate() {
            hek[(r, nk)] = grad[i];
        }
        use arael_faer::faer::prelude::Solve;
        let llt = hee.llt(arael_faer::faer::Side::Lower).expect("Hee SPD");
        let x = llt.solve(&hek);
        let sd = s.to_dense();
        let mut max_rel = 0.0f64;
        for b in 0..nk {
            for a in 0..=b {
                let mut want = full[keep_idx[a] + keep_idx[b] * n];
                for r in 0..ne {
                    want -= full[keep_idx[a] + elim_idx[r] * n] * x[(r, b)];
                }
                let rel = (sd[(a, b)] - want).abs() / (1.0 + want.abs());
                max_rel = max_rel.max(rel);
            }
        }
        let mut max_rhs = 0.0f64;
        for a in 0..nk {
            let mut want = grad[keep_idx[a]];
            for r in 0..ne {
                want -= full[keep_idx[a] + elim_idx[r] * n] * x[(r, nk)];
            }
            max_rhs = max_rhs.max((rhs_out[a] - want).abs() / (1.0 + want.abs()));
        }
        assert!(max_rel < 1e-9 && max_rhs < 1e-9, "S {:.2e} rhs {:.2e}", max_rel, max_rhs);
        println!("dense verification: OK (S max rel {:.2e}, rhs {:.2e})", max_rel, max_rhs);
    }
}
