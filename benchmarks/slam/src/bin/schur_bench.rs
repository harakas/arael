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
    path.serialize64(&mut params);
    let n = params.len();

    // block H via the two-scan assembly (as SparseFaer's fast path)
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

    // eliminated block ids: blocks fully inside the elimination hint's
    // scalar ranges (the landmark blocks)
    let hint = RootProblem::elimination_hint(&path);
    let nblk = partition.len() - 1;
    let eliminated: Vec<usize> = (0..nblk)
        .filter(|&b| {
            hint.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end)
        })
        .collect();
    assert!(!eliminated.is_empty(), "no eliminate_first hint on the model");

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
