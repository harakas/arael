// The block supernodal factor against faer's scalar Cholesky.
//
// Cholesky is unique for a given permutation, and faer's
// `factorize_symbolic_cholesky` takes a `Custom` permutation as given, so
// expanding `SupernodalSymbolic::order()` to scalars and forcing faer's
// simplicial route yields L over exactly our numbering. Every test here
// compares the panels entry by entry against that L, requires every panel
// entry outside faer's fill pattern (amalgamation and batch padding) to be
// exactly zero, and checks the solve against a dense twin.

use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use arael_faer::faer;
use arael_faer::faer::dyn_stack::MemStack;
use arael_faer::faer::perm::PermRef;
use arael_faer::faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholeskyRaw, SymmetricOrdering,
};
use arael_faer::faer::sparse::linalg::SupernodalThreshold;
use arael_faer::faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use arael_faer::faer::{Par, Side};
use arael_faer::schur::SchurReal;
use arael_faer::supernodal::{
    amd_block_order, nd_block_order, supernodal_factorize, supernodal_solve,
    supernodal_solve_multi, SupernodalContext, SupernodalParams, SupernodalSymbolic,
};
use arael_faer::SparseIndex;

/// Deterministic PRNG.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }
    fn unit(&mut self) -> f64 {
        (self.below(2001) as f64 - 1000.0) / 1000.0
    }
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            p.swap(i, self.below(i + 1));
        }
        p
    }
}

/// A block structure from a partition and scalar cell coordinates.
fn structure(part: Vec<SparseIndex>, cells: Vec<(usize, usize)>) -> SymbolicSparseBlockColMat<SparseIndex> {
    let (sym, _) =
        SymbolicSparseBlockColMat::from_scalar_coords(part.clone(), part, cells.len(), |k| cells[k]);
    sym
}

/// A random symmetric block structure: random widths, a chain for
/// connectivity, extra random couplings, full diagonal.
fn random_structure(nblk: usize, max_width: usize, extra: usize, seed: u64) -> SymbolicSparseBlockColMat<SparseIndex> {
    let mut rng = Lcg(seed);
    let mut part: Vec<SparseIndex> = vec![0];
    for _ in 0..nblk {
        part.push(part.last().unwrap() + (1 + rng.below(max_width)) as SparseIndex);
    }
    let mut cells = Vec::new();
    for j in 0..nblk {
        cells.push((part[j] as usize, part[j] as usize));
        if j > 0 {
            cells.push((part[j - 1] as usize, part[j] as usize));
        }
        for _ in 0..extra {
            cells.push((part[rng.below(j + 1)] as usize, part[j] as usize));
        }
    }
    structure(part, cells)
}

/// Random values on a structure, made SPD by adding `boost * n` to the
/// diagonal (2.0 is comfortably dominant, small values are ill
/// conditioned), mirrored into a dense twin, plus a right-hand side.
fn spd_on(
    sym: SymbolicSparseBlockColMat<SparseIndex>,
    seed: u64,
    boost: f64,
) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<f64>) {
    let n = sym.ncols();
    let mut rng = Lcg(seed ^ 0x9e3779b97f4a7c15);
    let mut a = SparseBlockColMat::<SparseIndex, f64>::zeroed(sym);
    let mut dense = vec![0.0f64; n * n];
    let asym = a.symbolic().clone();
    for j in 0..asym.nblk_cols() {
        let cj = asym.col_span(j).start;
        for b in asym.col_range(j) {
            let i = asym.blk_row(b);
            let ri = asym.row_span(i).start;
            let (wi, wj) = asym.block_dims(b);
            let base = asym.val_range(b).start;
            for cc in 0..wj {
                for rr in 0..wi {
                    if i == j && rr > cc {
                        continue;
                    }
                    let v = rng.unit();
                    a.vals_mut()[base + rr + cc * wi] = v;
                    dense[(ri + rr) + (cj + cc) * n] = v;
                    dense[(cj + cc) + (ri + rr) * n] = v;
                }
            }
        }
    }
    for j in 0..asym.nblk_cols() {
        let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
        let (w, _) = asym.block_dims(b);
        let base = asym.val_range(b).start;
        let cj = asym.col_span(j).start;
        for k in 0..w {
            a.vals_mut()[base + k * (w + 1)] += boost * n as f64;
            dense[(cj + k) * (n + 1)] += boost * n as f64;
        }
    }
    let rhs: Vec<f64> = (0..n).map(|_| 2.0 * rng.unit()).collect();
    (a, dense, rhs)
}

fn to_f32(a: &SparseBlockColMat<SparseIndex, f64>) -> SparseBlockColMat<SparseIndex, f32> {
    let mut b = SparseBlockColMat::<SparseIndex, f32>::zeroed(a.symbolic().clone());
    for (dst, &src) in b.vals_mut().iter_mut().zip(a.vals()) {
        *dst = src as f32;
    }
    b
}

fn rel_resid(dense: &[f64], n: usize, x: &[f64], rhs: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for r in 0..n {
        let mut ax = 0.0;
        for c in 0..n {
            ax += dense[r + c * n] * x[c];
        }
        num += (ax - rhs[r]) * (ax - rhs[r]);
        den += rhs[r] * rhs[r];
    }
    (num / den).sqrt()
}

/// faer's simplicial L L^T under our block order expanded to scalars:
/// per permuted column, the (permuted row, value) pairs sorted by row.
fn faer_factor<T: SchurReal>(
    a: &SparseBlockColMat<SparseIndex, T>,
    sn: &SupernodalSymbolic,
) -> Vec<Vec<(usize, T)>> {
    let asym = a.symbolic();
    let n = asym.ncols();
    let mut fwd: Vec<SparseIndex> = Vec::with_capacity(n);
    for &b in sn.order() {
        fwd.extend(asym.col_span(b as usize).map(|i| i as SparseIndex));
    }
    let mut inv = vec![0 as SparseIndex; n];
    for (new, &old) in fwd.iter().enumerate() {
        inv[old as usize] = new as SparseIndex;
    }
    let (col_ptr, row_idx) = asym.csc_pattern();
    let mut vals = vec![T::ZERO; asym.val_count()];
    a.csc_vals_into(&mut vals);
    let pattern = SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
    let symbolic = factorize_symbolic_cholesky(
        pattern,
        Side::Upper,
        SymmetricOrdering::Custom(PermRef::new_checked(&fwd, &inv, n)),
        CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        },
    )
    .expect("faer symbolic");
    let mut l = vec![T::ZERO; symbolic.len_val()];
    let need = symbolic.factorize_numeric_llt_scratch::<T>(Par::Seq, faer::Spec::default());
    let mut mem = vec![core::mem::MaybeUninit::<u8>::uninit(); need.unaligned_bytes_required()];
    symbolic
        .factorize_numeric_llt::<T>(
            &mut l,
            SparseColMatRef::new(pattern, &vals),
            Side::Upper,
            faer::linalg::cholesky::llt::factor::LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut mem),
            faer::Spec::default(),
        )
        .expect("faer numeric: the fixture must be positive definite");
    let SymbolicCholeskyRaw::Simplicial(s) = symbolic.raw() else {
        panic!("the simplicial route was forced");
    };
    let f = s.factor();
    let (cp, ri) = (f.col_ptr(), f.row_idx());
    (0..n)
        .map(|j| {
            let mut col: Vec<(usize, T)> =
                (cp[j] as usize..cp[j + 1] as usize).map(|k| (ri[k] as usize, l[k])).collect();
            col.sort_by_key(|&(r, _)| r);
            col
        })
        .collect()
}

/// faer's factor nnz (lower triangle, diagonal included) under our order.
fn faer_len_val(a: &SparseBlockColMat<SparseIndex, f64>, sn: &SupernodalSymbolic) -> u64 {
    faer_factor(a, sn).iter().map(|c| c.len() as u64).sum()
}

/// Every panel entry against faer's L: the largest
/// `|ours - faer| / (1 + |faer|)` over the entries faer has, and a count of
/// entries faer has NO structure for that are not exactly zero.
fn compare_with_faer<T: SchurReal>(
    a: &SparseBlockColMat<SparseIndex, T>,
    sn: &SupernodalSymbolic,
    factor: &[T],
) -> (f64, usize) {
    let l = faer_factor(a, sn);
    let asym = a.symbolic();
    let nblk = asym.nblk_cols();
    let mut sc = vec![0usize; nblk + 1];
    for k in 0..nblk {
        sc[k + 1] = sc[k] + asym.col_span(sn.order()[k] as usize).len();
    }
    let mut max_rel = 0.0f64;
    let mut nonzero_pads = 0usize;
    for s in 0..sn.n_supernodes() {
        let (q, h) = sn.supernode_dims(s);
        let c0 = sc[sn.supernode_cols(s).start];
        let base = sn.panel_range(s).start;
        // Permuted scalar row of every panel row.
        let mut prow = vec![0usize; h];
        for r in 0..q {
            prow[r] = c0 + r;
        }
        for (e, &blk) in sn.supernode_pattern(s).iter().enumerate() {
            let r0 = sn.supernode_pattern_rows(s)[e] as usize;
            let (lo, hi) = (sc[blk as usize], sc[blk as usize + 1]);
            for t in 0..hi - lo {
                prow[r0 + t] = lo + t;
            }
        }
        for c in 0..q {
            let col = &l[c0 + c];
            for r in c..h {
                let ours = factor[base + c * h + r].to_f64();
                match col.binary_search_by_key(&prow[r], |&(row, _)| row) {
                    Ok(k) => {
                        let v = col[k].1.to_f64();
                        max_rel = max_rel.max((ours - v).abs() / (1.0 + v.abs()));
                    }
                    Err(_) => {
                        if ours != 0.0 {
                            nonzero_pads += 1;
                        }
                    }
                }
            }
        }
    }
    (max_rel, nonzero_pads)
}

/// Symbolic + factorize (from a NaN-filled buffer) + solve; the factor
/// against faer's, the pads exactly zero, the residual against the dense
/// twin. Returns the symbolic for further assertions.
fn check(
    label: &str,
    a: &SparseBlockColMat<SparseIndex, f64>,
    dense: &[f64],
    rhs: &[f64],
    order: Option<&[usize]>,
    params: &SupernodalParams,
    par: Par,
) -> SupernodalSymbolic {
    let n = a.symbolic().ncols();
    let sn = SupernodalSymbolic::new(a.symbolic(), order, params).unwrap();
    let mut factor = vec![f64::NAN; sn.factor_val_count()];
    let mut ctx = SupernodalContext::new();
    supernodal_factorize(&sn, a, &mut factor, &mut ctx, par).unwrap();
    let (max_rel, pads) = compare_with_faer(a, &sn, &factor);
    assert!(max_rel < 1e-12, "{label}: factor differs from faer's by {max_rel:.3e}");
    assert_eq!(pads, 0, "{label}: {pads} padded entries are not exactly zero");
    let mut x = rhs.to_vec();
    supernodal_solve(&sn, &factor, &mut x, &mut ctx);
    let resid = rel_resid(dense, n, &x, rhs);
    assert!(resid < 1e-10, "{label}: residual {resid:.3e}");
    sn
}

/// The five parameter sets every structure runs under.
fn param_sets() -> Vec<(&'static str, SupernodalParams)> {
    vec![
        ("fundamental", SupernodalParams { relax: None, batch_ratio: None, ..Default::default() }),
        ("default", SupernodalParams::default()),
        ("lean", SupernodalParams::memory_lean()),
        ("no-postorder", SupernodalParams { postorder: false, ..Default::default() }),
        ("batch-3", SupernodalParams { batch_ratio: Some(3.0), ..Default::default() }),
    ]
}

/// The orders every structure runs under: natural, reversed, block AMD,
/// nested dissection, and a random permutation.
fn orders(sym: &SymbolicSparseBlockColMat<SparseIndex>, seed: u64) -> Vec<(&'static str, Vec<usize>)> {
    let nblk = sym.nblk_cols();
    vec![
        ("natural", (0..nblk).collect()),
        ("reversed", (0..nblk).rev().collect()),
        ("amd", amd_block_order(sym)),
        ("nd", nd_block_order(sym)),
        ("random", Lcg(seed ^ 0x5151).permutation(nblk)),
    ]
}

/// Random block structures (widths 1-6), well and ill conditioned,
/// under every order and parameter set; multi-rhs solves up to k = 131.
#[test]
fn factor_matches_faer_under_our_permutation() {
    for seed in [2u64, 21, 77] {
        for (nblk, boost) in [(40usize, 2.0), (40, 0.15), (7, 2.0)] {
            let sym = random_structure(nblk, 6, 2, seed);
            let (a, dense, rhs) = spd_on(sym.clone(), seed, boost);
            let n = a.symbolic().ncols();
            for (oname, order) in orders(&sym, seed) {
                for (pname, params) in param_sets() {
                    check(
                        &format!("seed {seed} nblk {nblk} boost {boost} {oname} {pname}"),
                        &a, &dense, &rhs, Some(&order), &params, Par::Seq,
                    );
                }
            }
            // Many right-hand sides at once, default order and parameters.
            let sn = SupernodalSymbolic::new(&sym, Some(&amd_block_order(&sym)), &SupernodalParams::default()).unwrap();
            let mut factor = vec![0.0; sn.factor_val_count()];
            let mut ctx = SupernodalContext::new();
            supernodal_factorize(&sn, &a, &mut factor, &mut ctx, Par::Seq).unwrap();
            let mut rng = Lcg(seed);
            for k in [1usize, 7, 131] {
                let many: Vec<f64> = (0..n * k).map(|_| rng.unit()).collect();
                let mut x = many.clone();
                supernodal_solve_multi(&sn, &factor, &mut x, k, &mut ctx);
                for c in 0..k {
                    let r = rel_resid(&dense, n, &x[c * n..(c + 1) * n], &many[c * n..(c + 1) * n]);
                    assert!(r < 1e-10, "seed {seed} k {k} column {c}: residual {r:.3e}");
                }
            }
        }
    }
}

/// Every block one wide, so the block analysis is a scalar one: the
/// factor matches faer's, and with amalgamation and postordering off the
/// fill count is faer's `len_val` exactly.
#[test]
fn scalar_limit_matches_faer() {
    for seed in [1u64, 5, 9] {
        let nblk = 150usize;
        let sym = random_structure(nblk, 1, 3, seed);
        let (a, dense, rhs) = spd_on(sym.clone(), seed, 0.05);
        for (oname, order) in orders(&sym, seed) {
            for (pname, params) in param_sets() {
                check(
                    &format!("scalar seed {seed} {oname} {pname}"),
                    &a, &dense, &rhs, Some(&order), &params, Par::Seq,
                );
            }
            let bare = SupernodalParams { relax: None, postorder: false, ..Default::default() };
            let sn = check(&format!("scalar seed {seed} {oname} bare"), &a, &dense, &rhs, Some(&order), &bare, Par::Seq);
            assert_eq!(sn.factor_scalar_nnz(), faer_len_val(&a, &sn), "scalar seed {seed} {oname}: fill");
        }
    }
}

/// A forest: two chains, an isolated block, a block hanging off the second
/// chain, and one long-range coupling in the first.
fn forest() -> SymbolicSparseBlockColMat<SparseIndex> {
    let widths = [3usize, 2, 3, 1, 4, 3, 3, 2, 2, 5, 1, 3];
    let mut part: Vec<SparseIndex> = vec![0];
    for &w in &widths {
        part.push(part.last().unwrap() + w as SparseIndex);
    }
    let at = |b: usize| part[b] as usize;
    let mut cells: Vec<(usize, usize)> = (0..widths.len()).map(|b| (at(b), at(b))).collect();
    for b in 1..5 {
        cells.push((at(b - 1), at(b)));
    }
    for b in 7..10 {
        cells.push((at(b - 1), at(b)));
    }
    cells.push((at(6), at(11)));
    cells.push((at(0), at(4)));
    structure(part, cells)
}

/// A disconnected block graph -- several elimination-tree roots, a
/// supernode with an empty pattern in the middle of the order.
#[test]
fn forest_and_isolated_blocks() {
    let sym = forest();
    let (a, dense, rhs) = spd_on(sym.clone(), 3, 2.0);
    for (oname, order) in orders(&sym, 3) {
        for (pname, params) in param_sets() {
            check(&format!("forest {oname} {pname}"), &a, &dense, &rhs, Some(&order), &params, Par::Seq);
        }
    }
}

/// The parallel path on a forest: two disconnected random components give
/// the cut two roots to work under.
#[cfg(feature = "rayon")]
#[test]
fn forest_and_isolated_blocks_threaded() {
    let comp = 300usize;
    let mut rng = Lcg(21);
    let mut part: Vec<SparseIndex> = vec![0];
    for _ in 0..2 * comp {
        part.push(part.last().unwrap() + (1 + rng.below(4)) as SparseIndex);
    }
    let mut cells = Vec::new();
    for c in 0..2 {
        let base = c * comp;
        for j in 0..comp {
            let bj = base + j;
            cells.push((part[bj] as usize, part[bj] as usize));
            if j > 0 {
                cells.push((part[bj - 1] as usize, part[bj] as usize));
            }
            cells.push((part[base + rng.below(j + 1)] as usize, part[bj] as usize));
        }
    }
    let sym = structure(part, cells);
    let (a, dense, rhs) = spd_on(sym.clone(), 21, 2.0);
    let nd = nd_block_order(&sym);
    for threads in [2usize, 4] {
        check(&format!("threaded forest {threads}"), &a, &dense, &rhs, Some(&nd), &SupernodalParams::default(), Par::rayon(threads));
    }
    let small = forest();
    let (a, dense, rhs) = spd_on(small.clone(), 3, 2.0);
    check("threaded small forest", &a, &dense, &rhs, None, &SupernodalParams::default(), Par::rayon(4));
}

/// A clique of wide blocks (one big dense panel) plus a tail of 9-wide
/// blocks each tied to three clique members.
fn wide() -> (SymbolicSparseBlockColMat<SparseIndex>, Vec<usize>) {
    let (nclique, ntail) = (6usize, 12usize);
    let mut part: Vec<SparseIndex> = vec![0];
    for _ in 0..nclique {
        part.push(part.last().unwrap() + 24);
    }
    for _ in 0..ntail {
        part.push(part.last().unwrap() + 9);
    }
    let mut cells = Vec::new();
    for j in 0..nclique {
        for i in 0..=j {
            cells.push((part[i] as usize, part[j] as usize));
        }
    }
    let mut rng = Lcg(77);
    for t in 0..ntail {
        let b = nclique + t;
        cells.push((part[b] as usize, part[b] as usize));
        for _ in 0..3 {
            cells.push((part[rng.below(nclique)] as usize, part[b] as usize));
        }
    }
    let tail_first: Vec<usize> = (nclique..nclique + ntail).chain(0..nclique).collect();
    (structure(part, cells), tail_first)
}

/// A 144-column panel (past faer's dense block size of 128, so the
/// blocked Cholesky and the panel solve run at size), in f64 and in f32 at
/// single precision's tolerance.
#[test]
fn wide_panels_match_faer() {
    let (sym, tail_first) = wide();
    let (a, dense, rhs) = spd_on(sym.clone(), 4, 0.5);
    let natural: Vec<usize> = (0..sym.nblk_cols()).collect();
    for (oname, order) in [("natural", &natural), ("tail-first", &tail_first)] {
        for (pname, params) in param_sets() {
            check(&format!("wide {oname} {pname}"), &a, &dense, &rhs, Some(order), &params, Par::Seq);
        }
    }

    let a32 = to_f32(&a);
    let n = a.symbolic().ncols();
    for (oname, order) in [("natural", &natural), ("tail-first", &tail_first)] {
        let sn = SupernodalSymbolic::new(&sym, Some(order), &SupernodalParams::default()).unwrap();
        let mut factor = vec![f32::NAN; sn.factor_val_count()];
        let mut ctx = SupernodalContext::new();
        supernodal_factorize(&sn, &a32, &mut factor, &mut ctx, Par::Seq).unwrap();
        let (max_rel, pads) = compare_with_faer(&a32, &sn, &factor);
        assert!(max_rel < 1e-4, "wide f32 {oname}: factor differs from faer's f32 by {max_rel:.3e}");
        assert_eq!(pads, 0, "wide f32 {oname}: padded entries not exactly zero");
        let mut x: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        supernodal_solve(&sn, &factor, &mut x, &mut ctx);
        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let resid = rel_resid(&dense, n, &x64, &rhs);
        assert!(resid < 1e-4, "wide f32 {oname}: residual {resid:.3e}");
    }
}

/// The packed-depth cap. 300 3-wide leaves on two 6-wide targets, the
/// 200 leaves seen by both targets first so their spans agree and the
/// default ratio joins them: one target's packed depth reaches 600, past
/// `BATCH_K_MAX`, and the bucket must split.
#[test]
fn batching_past_the_depth_cap_matches_faer() {
    let nlm = 300usize;
    let mut part: Vec<SparseIndex> = vec![0];
    for _ in 0..nlm {
        part.push(part.last().unwrap() + 3);
    }
    part.push(part.last().unwrap() + 6);
    part.push(part.last().unwrap() + 6);
    let mut cells: Vec<(usize, usize)> = (0..nlm + 2).map(|b| (part[b] as usize, part[b] as usize)).collect();
    for l in 0..nlm {
        cells.push((part[l] as usize, part[nlm] as usize));
        if l < 200 {
            cells.push((part[l] as usize, part[nlm + 1] as usize));
        }
    }
    cells.push((part[nlm] as usize, part[nlm + 1] as usize));
    let sym = structure(part, cells);
    let (a, dense, rhs) = spd_on(sym.clone(), 11, 1.0);
    for (pname, params) in param_sets() {
        let sn = check(&format!("depth cap {pname}"), &a, &dense, &rhs, None, &params, Par::Seq);
        if params.batch_ratio.is_some() {
            assert!(sn.batched_pairs() >= 250, "depth cap {pname}: only {} pairs batched", sn.batched_pairs());
        } else {
            assert_eq!(sn.batched_pairs(), 0);
        }
    }
}

/// Joinable and non-joinable descendants interleaved on one target:
/// runs of 3-wide leaves separated by 20- and 24-wide ones (past
/// `BATCH_DEPTH_MAX`), so buckets open, flush around the wide ones and
/// reopen.
#[test]
fn mixed_batch_buckets_match_faer() {
    let mut part: Vec<SparseIndex> = vec![0];
    let mut widths = Vec::new();
    for g in 0..8 {
        for _ in 0..5 {
            widths.push(3usize);
        }
        widths.push(if g % 2 == 0 { 20 } else { 24 });
    }
    widths.push(6);
    for &w in &widths {
        part.push(part.last().unwrap() + w as SparseIndex);
    }
    let target = widths.len() - 1;
    let mut cells: Vec<(usize, usize)> = (0..widths.len()).map(|b| (part[b] as usize, part[b] as usize)).collect();
    for b in 0..target {
        cells.push((part[b] as usize, part[target] as usize));
    }
    let sym = structure(part, cells);
    let (a, dense, rhs) = spd_on(sym.clone(), 13, 1.0);
    let no_relax = SupernodalParams { relax: None, ..Default::default() };
    let sn = check("mixed buckets", &a, &dense, &rhs, None, &no_relax, Par::Seq);
    assert!(sn.batched_pairs() >= 30, "mixed buckets: only {} pairs batched", sn.batched_pairs());
    for (pname, params) in param_sets() {
        check(&format!("mixed buckets {pname}"), &a, &dense, &rhs, None, &params, Par::Seq);
    }
}

/// The empty matrix is an empty factorization, not a panic.
#[test]
fn an_empty_matrix_factors_to_nothing() {
    let sym = structure(vec![0], vec![]);
    let a = SparseBlockColMat::<SparseIndex, f64>::zeroed(sym.clone());
    for (_, params) in param_sets() {
        let sn = SupernodalSymbolic::new(&sym, None, &params).unwrap();
        assert_eq!(sn.n_supernodes(), 0);
        assert_eq!(sn.factor_val_count(), 0);
        assert_eq!(sn.dim(), 0);
        let mut factor: Vec<f64> = Vec::new();
        let mut ctx = SupernodalContext::new();
        supernodal_factorize(&sn, &a, &mut factor, &mut ctx, Par::Seq).unwrap();
        let mut x: Vec<f64> = Vec::new();
        supernodal_solve(&sn, &factor, &mut x, &mut ctx);
        supernodal_solve_multi(&sn, &factor, &mut x, 3, &mut ctx);
        #[cfg(feature = "rayon")]
        supernodal_factorize(&sn, &a, &mut factor, &mut ctx, Par::rayon(4)).unwrap();
    }
    let sn = SupernodalSymbolic::new(&sym, Some(&[]), &SupernodalParams::default()).unwrap();
    assert_eq!(sn.n_supernodes(), 0);
}

/// One block is one panel: its factor is the dense Cholesky of the tile.
#[test]
fn a_single_block_matches_faer() {
    let sym = structure(vec![0, 4], vec![(0, 0)]);
    let (a, dense, rhs) = spd_on(sym.clone(), 6, 2.0);
    for (pname, params) in param_sets() {
        let sn = check(&format!("single block {pname}"), &a, &dense, &rhs, None, &params, Par::Seq);
        assert_eq!(sn.n_supernodes(), 1);
        assert_eq!(sn.supernode_dims(0), (4, 4));
    }
}

/// Zero right-hand sides leave the buffer alone.
#[test]
fn zero_right_hand_sides_are_a_no_op() {
    let sym = random_structure(10, 3, 1, 4);
    let (a, _, _) = spd_on(sym.clone(), 4, 2.0);
    let sn = SupernodalSymbolic::new(&sym, None, &SupernodalParams::default()).unwrap();
    let mut factor = vec![0.0; sn.factor_val_count()];
    let mut ctx = SupernodalContext::new();
    supernodal_factorize(&sn, &a, &mut factor, &mut ctx, Par::Seq).unwrap();
    let mut x: Vec<f64> = Vec::new();
    supernodal_solve_multi(&sn, &factor, &mut x, 0, &mut ctx);
    assert!(x.is_empty());
}

/// An order that names a block twice is rejected in every build.
#[test]
#[should_panic(expected = "order names block 2 twice")]
fn a_repeated_block_in_the_order_is_rejected() {
    let sym = random_structure(5, 2, 1, 4);
    let bad = [0usize, 1, 2, 2, 4];
    let _ = SupernodalSymbolic::new(&sym, Some(&bad), &SupernodalParams::default());
}
