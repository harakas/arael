//! How precisely each way of forming a BAL problem's reduced camera system
//! S gets it from the f32 model's Hessian, and whether S stays positive
//! definite: W = C D^-1 by Cholesky solves (the CPU's and the GPU's
//! reduction) against multiplying by D's explicit inverse, each emulated in
//! f32, df32 or f64 at each stage.
//!
//!     BAL_DATASET=372 BAL_LAMBDAS=1e-4,1e-6 cargo run -r --bin schur_precision
//!
//! The Hessians are taken at the initial point, their diagonals damped by
//! 1 + lambda (BAL_LAMBDAS, default 1e-4,1e-6,1e-8). Errors are against the
//! exact S of the f32 model's Hessian (formed in f64); the f64 model's S is
//! listed for scale. BAL_MODEL=f64 runs the recipes on the f64 model's
//! Hessian instead, against its S. Columns:
//!
//!   err/max   max |S - S_exact| over max |S_exact|
//!   fro       Frobenius norm of the difference over S_exact's
//!   tile      the worst tile's Frobenius error over that tile's norm
//!   llt64     S factorized in f64: ok, or the failure
//!   llt32     S rounded to f32 and factorized in f32 (the f32 route)
//!   step64    the step through the f64 factor against the exact S's
//!   step32    the step through the f32 factor against the exact S's

// The included modules are the benchmark's, and expose more than this binary
// calls.
#![allow(dead_code)]

#[path = "../bal.rs"]
mod bal;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::LmProblemInternals;
use arael_faer::SparseIndex;
use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use arael_faer::schur::{SchurContext, SchurSymbolic, schur_reduce, schur_symbolic};
use arael_faer::supernodal::{
    SupernodalContext, SupernodalParams, SupernodalSymbolic, amd_block_order, supernodal_factorize_as, supernodal_solve_as,
};

/// The precision of a stage: f32, df32 (emulated as f64 rounded to 48
/// significant bits after each operation), or f64.
#[derive(Clone, Copy, PartialEq)]
enum P {
    F32,
    Df32,
    F64,
}
use P::{Df32, F32, F64};

/// Where an emulated reduction rounds, and how it applies a point's block D.
#[derive(Clone, Copy)]
struct Recipe {
    name: &'static str,
    /// W = C D^-1 by multiplying with D's explicit inverse, else by two
    /// triangular solves with D's Cholesky factor.
    inverse: bool,
    /// D's Cholesky factor, and its inverse.
    d: P,
    /// W formed from operands in this precision.
    w: P,
    /// W rounded once to f32 after forming.
    w_round: bool,
    /// The products W_a C_b^T and their sum.
    acc: P,
    /// H_kk - sum.
    fin: P,
}

const RECIPES: [Recipe; 11] = [
    Recipe { name: "inverse, all f64", inverse: true, d: F64, w: F64, w_round: false, acc: F64, fin: F64 },
    Recipe { name: "solve, all f64", inverse: false, d: F64, w: F64, w_round: false, acc: F64, fin: F64 },
    Recipe { name: "inverse, all df32", inverse: true, d: Df32, w: Df32, w_round: false, acc: Df32, fin: Df32 },
    Recipe { name: "solve, all df32", inverse: false, d: Df32, w: Df32, w_round: false, acc: Df32, fin: Df32 },
    Recipe { name: "inverse df32, W rounded", inverse: true, d: Df32, w: Df32, w_round: true, acc: F32, fin: Df32 },
    Recipe { name: "solve df32, W rounded", inverse: false, d: Df32, w: Df32, w_round: true, acc: F32, fin: Df32 },
    Recipe { name: "inverse f64, W in f32", inverse: true, d: F64, w: F32, w_round: false, acc: F32, fin: F64 },
    Recipe { name: "inverse, all f32", inverse: true, d: F32, w: F32, w_round: false, acc: F32, fin: F32 },
    Recipe { name: "solve L df32, W in f32", inverse: false, d: Df32, w: F32, w_round: false, acc: F32, fin: Df32 },
    Recipe { name: "solve, W in f32, final df32", inverse: false, d: F32, w: F32, w_round: false, acc: F32, fin: Df32 },
    Recipe { name: "solve, all f32 (CPU emulated)", inverse: false, d: F32, w: F32, w_round: false, acc: F32, fin: F32 },
];

trait Zx {
    fn zx(self) -> usize;
}

impl Zx for SparseIndex {
    fn zx(self) -> usize {
        self as usize
    }
}

/// `x` rounded to `p`.
fn q(p: P, x: f64) -> f64 {
    match p {
        F32 => x as f32 as f64,
        // round to nearest at 48 significant bits: clear the low 5 of f64's 53
        Df32 => {
            let b = x.to_bits();
            f64::from_bits((b + (1 << 4)) & !((1u64 << 5) - 1))
        }
        F64 => x,
    }
}

/// `acc + x * y` with one rounding, in `p`.
fn fma(p: P, acc: f64, x: f64, y: f64) -> f64 {
    match p {
        F32 => (x as f32).mul_add(y as f32, acc as f32) as f64,
        _ => q(p, x.mul_add(y, acc)),
    }
}

/// S's values by `r`, from the Hessian values `h` (f32 values for the f32
/// model), in `sym`'s layout; `Err` names the eliminated block whose D is
/// not positive definite.
fn reduce(sym: &SchurSymbolic<SparseIndex>, hs: &SymbolicSparseBlockColMat<SparseIndex>, h: &[f64], r: &Recipe) -> Result<Vec<f64>, usize> {
    let s = &sym.s;
    let mut acc = vec![0.0f64; s.val_count()];
    let mut l = Vec::new();
    let mut li = Vec::new();
    let mut dinv = Vec::new();
    let mut w = Vec::new();
    let mut w_at = Vec::new();
    for slot in 0..sym.elim_diag().len() {
        let we = sym.elim_w()[slot].zx();
        let diag = hs.val_range(sym.elim_diag()[slot].zx()).start;
        let dval = |i: usize, j: usize| if i <= j { h[diag + i + j * we] } else { h[diag + j + i * we] };
        // L, lower, row-major
        let fd = r.d;
        l.clear();
        l.resize(we * we, 0.0);
        for j in 0..we {
            let mut d = q(fd, dval(j, j));
            for k in 0..j {
                d = fma(fd, d, -l[j * we + k], l[j * we + k]);
            }
            if !(d > 0.0) {
                return Err(slot);
            }
            let ljj = q(fd, d.sqrt());
            l[j * we + j] = ljj;
            for i in j + 1..we {
                let mut v = q(fd, dval(i, j));
                for k in 0..j {
                    v = fma(fd, v, -l[i * we + k], l[j * we + k]);
                }
                l[i * we + j] = q(fd, v / ljj);
            }
        }
        if r.inverse {
            // L^-1 by forward substitution, then D^-1 = L^-T L^-1, as k_schur_elim
            li.clear();
            li.resize(we * we, 0.0);
            for c in 0..we {
                for row in c..we {
                    let mut v = if row == c { 1.0 } else { 0.0 };
                    for k in c..row {
                        v = fma(fd, v, -l[row * we + k], li[k * we + c]);
                    }
                    li[row * we + c] = q(fd, v / l[row * we + row]);
                }
            }
            dinv.clear();
            dinv.resize(we * we, 0.0);
            for i in 0..we {
                for j in 0..we {
                    let mut v = 0.0;
                    for k in 0..we {
                        v = fma(fd, v, li[k * we + i], li[k * we + j]);
                    }
                    dinv[i * we + j] = v;
                }
            }
        }
        // W_o, row-major wo x we per observer
        let fw = r.w;
        let (o0, o1) = (sym.elim_obs_ptr()[slot].zx(), sym.elim_obs_ptr()[slot + 1].zx());
        let coupling = |o: usize, i: usize, k: usize| {
            let (wo, ca) = (sym.obs_w()[o].zx(), sym.obs_ca_off()[o].zx());
            if sym.obs_trans()[o] { h[ca + k + i * we] } else { h[ca + i + k * wo] }
        };
        w.clear();
        w_at.clear();
        let mut y = vec![0.0; we];
        for o in o0..o1 {
            w_at.push(w.len());
            for i in 0..sym.obs_w()[o].zx() {
                let at = w.len();
                if r.inverse {
                    for col in 0..we {
                        let mut v = 0.0;
                        for k in 0..we {
                            v = fma(fw, v, q(fw, coupling(o, i, k)), q(fw, dinv[k * we + col]));
                        }
                        w.push(v);
                    }
                } else {
                    // L y = c_i, then L^T x = y
                    for k in 0..we {
                        let mut v = q(fw, coupling(o, i, k));
                        for j in 0..k {
                            v = fma(fw, v, -q(fw, l[k * we + j]), y[j]);
                        }
                        y[k] = q(fw, v / q(fw, l[k * we + k]));
                    }
                    w.resize(at + we, 0.0);
                    for k in (0..we).rev() {
                        let mut v = y[k];
                        for j in k + 1..we {
                            v = fma(fw, v, -q(fw, l[j * we + k]), w[at + j]);
                        }
                        w[at + k] = q(fw, v / q(fw, l[k * we + k]));
                    }
                }
                if r.w_round {
                    for v in &mut w[at..] {
                        *v = q(F32, *v);
                    }
                }
            }
        }
        // the pairs' products into the tiles of S, a <= b
        let fa = r.acc;
        for (ia, oa) in (o0..o1).enumerate() {
            for ob in o0..o1 {
                let (ka, kb) = (sym.obs_kept()[oa].zx(), sym.obs_kept()[ob].zx());
                if ka > kb {
                    continue;
                }
                let tile = sym.s_tile(ka, kb).expect("a tile for every observer pair") as usize;
                let (wa, wb) = (sym.obs_w()[oa].zx(), sym.obs_w()[ob].zx());
                for j in 0..wb {
                    for i in 0..wa {
                        if ka == kb && i > j {
                            continue;
                        }
                        let mut v = acc[tile + i + j * wa];
                        for k in 0..we {
                            v = fma(fa, v, q(fa, w[w_at[ia] + i * we + k]), q(fa, coupling(ob, j, k)));
                        }
                        acc[tile + i + j * wa] = v;
                    }
                }
            }
        }
    }
    // S = H_kk - sum, diagonal tiles upper-only
    let mut hk = vec![0.0f64; s.val_count()];
    let (src, dst) = sym.copy_tiles();
    for (&a, &b) in src.iter().zip(dst) {
        let from = hs.val_range(a.zx());
        let to = s.val_range(b.zx());
        hk[to].copy_from_slice(&h[from]);
    }
    let mut out = vec![0.0f64; s.val_count()];
    for j in 0..s.nblk_cols() {
        for x in s.col_range(j) {
            let i = s.blk_row(x);
            let wa = s.row_span(i).len();
            for (e, v) in s.val_range(x).enumerate() {
                out[v] = if i == j && e % wa > e / wa { 0.0 } else { q(r.fin, hk[v] - acc[v]) };
            }
        }
    }
    Ok(out)
}

/// err/max, Frobenius and worst-tile relative errors of `v` against `x`.
fn errors(s: &SymbolicSparseBlockColMat<SparseIndex>, v: &[f64], x: &[f64]) -> (f64, f64, f64) {
    let (mut dmax, mut xmax, mut d2, mut x2, mut tile) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in 0..s.nblk_cols() {
        for b in s.col_range(j) {
            let (mut td, mut tx) = (0.0, 0.0);
            for k in s.val_range(b) {
                let d = v[k] - x[k];
                dmax = dmax.max(d.abs());
                xmax = xmax.max(x[k].abs());
                td += d * d;
                tx += x[k] * x[k];
            }
            d2 += td;
            x2 += tx;
            if tx > 0.0 {
                tile = tile.max((td / tx).sqrt());
            }
        }
    }
    (dmax / xmax, (d2 / x2).sqrt(), tile)
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let n: f64 = b.iter().map(|y| y * y).sum();
    (d / n).sqrt()
}

/// The step `S^-1 rhs` through a factor in `T`, or the factorization's error.
fn step<T: arael_faer::schur::SchurReal>(sn: &SupernodalSymbolic, s: &SparseBlockColMat<SparseIndex, f64>, rhs: &[f64], threads: usize) -> Result<Vec<f64>, String> {
    let mut factor = vec![T::ZERO; sn.factor_val_count()];
    let mut ctx = SupernodalContext::<T>::new();
    supernodal_factorize_as(sn, s, &mut factor, &mut ctx, threads).map_err(|e| e.to_string())?;
    let mut x = rhs.to_vec();
    supernodal_solve_as(sn, &factor, &mut x, &mut ctx, threads);
    Ok(x)
}

fn main() {
    let cams = std::env::var("BAL_DATASET").unwrap_or_else(|_| "372".to_string());
    let lambdas: Vec<f64> = std::env::var("BAL_LAMBDAS")
        .unwrap_or_else(|_| "1e-4,1e-6,1e-8".to_string())
        .split(',')
        .map(|v| v.trim().parse().expect("BAL_LAMBDAS: comma-separated numbers"))
        .collect();
    let threads: usize = std::env::var("BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let f64_model = match std::env::var("BAL_MODEL").as_deref() {
        Err(_) | Ok("f32") => false,
        Ok("f64") => true,
        Ok(other) => panic!("BAL_MODEL={other}: f32 or f64"),
    };
    let prefix = format!("problem-{cams}-");
    let path = std::fs::read_dir("datasets")
        .expect("the datasets directory (run from benchmarks/bal)")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix) && n.ends_with("-pre.txt")))
        .unwrap_or_else(|| panic!("BAL_DATASET={cams}: no datasets/{prefix}*-pre.txt"));
    let ds = bal::load(path.to_str().unwrap());
    let mut scene = arael_runner::build_f64(&ds);
    let (cameras, points, observations) = arael_runner::build_parts::<f32>(&ds);
    let mut scene32 = arael_runner::SceneF { cameras, points, observations };
    let (h64_0, grad64, partition) = bench_harness::arael::hessian::<SparseIndex, _>(&mut scene);
    let (h32_0, grad32, partition32) = bench_harness::arael::hessian_in::<SparseIndex, f32, _>(&mut scene32);
    assert_eq!(partition, partition32);
    assert_eq!(h64_0.symbolic().val_count(), h32_0.symbolic().val_count());
    let nblk = partition.len() - 1;
    let hs = h64_0.symbolic().clone();
    // what the solver marginalizes, as gpu_factor_bench picks it
    let blocks_in = |ranges: &[std::ops::Range<usize>]| -> Vec<usize> {
        (0..nblk).filter(|&b| ranges.iter().any(|r| r.start <= partition[b] && partition[b + 1] <= r.end)).collect()
    };
    let hint = LmProblemInternals::marginalize_hint(&scene);
    let candidates: Vec<Vec<usize>> =
        if hint.is_empty() { LmProblemInternals::marginalize_candidates(&scene).iter().map(|r| blocks_in(r)).collect() } else { vec![blocks_in(&hint)] };
    let eliminated: Vec<usize> = candidates
        .into_iter()
        .filter(|ids| !ids.is_empty())
        .max_by_key(|ids| ids.iter().map(|&b| partition[b + 1] - partition[b]).sum::<usize>())
        .expect("nothing marginalizable in a BAL scene");
    let sym = schur_symbolic(&hs, &eliminated).unwrap();
    let order = amd_block_order(&sym.s);
    let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
    let damp_diag = |vals: &mut [f64], lambda: f64| {
        for b in 0..nblk {
            let w = partition[b + 1] - partition[b];
            let d = hs.col_range(b).find(|&x| hs.blk_row(x) == b).unwrap();
            let base = hs.val_range(d).start;
            for k in 0..w {
                vals[base + k * (w + 1)] *= 1.0 + lambda;
            }
        }
    };
    for &lambda in &lambdas {
        // the f64 model's, and the f32 model's damped in f32 as its solve does
        let mut h64 = h64_0.clone();
        damp_diag(h64.vals_mut(), lambda);
        let mut h32 = h32_0.clone();
        let d32 = 1.0f32 + lambda as f32;
        for b in 0..nblk {
            let w = partition[b + 1] - partition[b];
            let d = hs.col_range(b).find(|&x| hs.blk_row(x) == b).unwrap();
            let base = hs.val_range(d).start;
            for k in 0..w {
                h32.vals_mut()[base + k * (w + 1)] *= d32;
            }
        }
        // the recipes' input: the f32 model's Hessian, or the f64 model's
        let (h32w, g32w): (Vec<f64>, Vec<f64>) = if f64_model {
            (h64.vals().to_vec(), grad64.clone())
        } else {
            (h32.vals().iter().map(|&v| v as f64).collect(), grad32.iter().map(|&v| v as f64).collect())
        };
        let h32w_mat = SparseBlockColMat::new(hs.clone(), h32w.clone());

        let mut ctx = SchurContext::<f64>::new();
        let mut ctx32 = SchurContext::<f32>::new();
        let nk = sym.s.nrows();
        let mut s_ref = sym.alloc_s::<f64>();
        let mut rhs_ref = vec![0.0; nk];
        schur_reduce(&sym, &h64, &grad64, &mut ctx, &mut s_ref, &mut rhs_ref).unwrap();
        let mut s_x = sym.alloc_s::<f64>();
        let mut rhs_x = vec![0.0; nk];
        schur_reduce(&sym, &h32w_mat, &g32w, &mut ctx, &mut s_x, &mut rhs_x).unwrap();
        let mut s_c = sym.alloc_s::<f32>();
        let mut rhs_c = vec![0.0f32; nk];
        let cpu32 = schur_reduce(&sym, &h32, &grad32, &mut ctx32, &mut s_c, &mut rhs_c).map(|_| s_c.vals().iter().map(|&v| v as f64).collect::<Vec<f64>>());

        println!(
            "Ladybug-{cams}, the {} model, lambda {lambda:e}: S {nk} x {nk}, {} pairs, {} eliminated blocks",
            if f64_model { "f64" } else { "f32" },
            sym.pair_count(),
            sym.elim_diag().len()
        );
        println!(
            "{:<36} {:>9} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9}",
            "recipe", "err/max", "fro", "tile", "llt64", "llt32", "step64", "step32"
        );
        // 1 + lambda rounds to 1 in f32 below about 6e-8: the f32 model's
        // Hessian is then undamped and its exact S can be singular
        let x64 = step::<f64>(&sn, &s_x, &rhs_x, threads).ok();
        let row = |name: &str, vals: Result<Vec<f64>, String>, rhs: &[f64]| {
            let vals = match vals {
                Ok(v) => v,
                Err(e) => {
                    println!("{name:<36} {e}");
                    return;
                }
            };
            let (em, fro, tile) = errors(&sym.s, &vals, s_x.vals());
            let m = SparseBlockColMat::new(sym.s.clone(), vals);
            let s64 = step::<f64>(&sn, &m, rhs, threads);
            let s32 = step::<f32>(&sn, &m, rhs, threads);
            let cell = |r: &Result<Vec<f64>, String>| match (r, &x64) {
                (Ok(x), Some(x64)) => (String::from("ok"), format!("{:.2e}", rel(x, x64))),
                (Ok(_), None) => (String::from("ok"), String::from("-")),
                (Err(_), _) => (String::from("notPD"), String::from("-")),
            };
            let ((l64, e64), (l32, e32)) = (cell(&s64), cell(&s32));
            println!("{name:<36} {em:>9.2e} {fro:>9.2e} {tile:>9.2e} {l64:>8} {l32:>8} {e64:>9} {e32:>9}");
        };
        row("f64 model (for scale)", Ok(s_ref.vals().to_vec()), &rhs_ref);
        row("exact (schur_reduce f64)", Ok(s_x.vals().to_vec()), &rhs_x);
        row("CPU f32 (schur_reduce f32)", cpu32.map_err(|e| e.to_string()), &rhs_x);
        for r in &RECIPES {
            row(r.name, reduce(&sym, &hs, &h32w, r).map_err(|slot| format!("D of eliminated block {slot} not positive definite")), &rhs_x);
        }
        println!();
    }
}
