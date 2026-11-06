//! A block matrix's supernodal factorization and solve on a GPU against
//! the CPU's (features `gpu-cuda`, `gpu-hip`): ms per factorization and per
//! solve on the device (the matrix uploaded once, timed to the device's
//! wait), in f64 and with the f32 factor, the solutions checked against
//! the CPU's. What an Nsight Compute session profiles. A factorization
//! not positive definite in a precision says so and is timed all the same.
//!
//! The environment: ROUNDS (20), BENCH_THREADS (the CPU's threads, 8),
//! ARAEL_GPU (`cuda` or `hip`); GPU_SMALL_Q, GPU_SMALL_FLOPS,
//! GPU_BIG_PAIR_FLOPS, GPU_WIDE_PAIR_K, GPU_WIDE_PAIR_FLOPS,
//! GPU_CHAIN_FLOPS_F32, GPU_CHAIN_FLOPS_F64, GPU_CHAIN_ITEMS,
//! GPU_SOLVE_CHAIN_WORK, GPU_SOLVE_WORK and GPU_BLOCKS_PER_SM override the
//! device's
//! `GpuSupernodalParams`; GPU_CHECK_FACTOR=1 compares the f64 factor with
//! the CPU's, supernode by supernode, and checks 50 more factorizations;
//! GPU_ORDER picks the elimination order, `amd` (the default) or `nd`
//! (nested dissection); GPU_LEVELS=1 prints the elimination tree's levels (as the device runs
//! them, leaves first): supernodes, the widest and the tallest, the
//! descendant pairs and the multiply-adds of the updates and the factor.
//!
//! ```ignore
//! bench_harness::gpu_factor::compare("S of slam 1000", &s, &rhs);
//! ```

use std::sync::Arc;
use std::time::Instant;

use arael::__gpu::{Device, FactorizeError, GpuFactor, GpuPrecision, GpuSupernodalParams, Wide, bytemuck};
use arael_faer::SparseIndex;
use arael_faer::bsc::SparseBlockColMat;
use arael_faer::supernodal::{
    SupernodalContext, SupernodalParams, SupernodalSymbolic, amd_block_order, nd_block_order, supernodal_factorize, supernodal_solve,
};

/// The median of `v`, ms.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `f` timed `rounds` times, ms each.
fn times(rounds: usize, mut f: impl FnMut()) -> Vec<f64> {
    (0..rounds)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect()
}

/// The largest difference of `a` from `b`, relative to `b`'s largest entry.
fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let max = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / max
}

/// The device's parameters, the defaults overridden by the environment.
fn params() -> GpuSupernodalParams {
    let var = |name: &str| std::env::var(name).ok().map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("{name}={v}: a number")));
    let mut p = GpuSupernodalParams::default();
    if let Some(v) = var("GPU_SMALL_Q") {
        p.small_q = v as u32;
    }
    if let Some(v) = var("GPU_SMALL_FLOPS") {
        p.small_flops = v;
    }
    if let Some(v) = var("GPU_BIG_PAIR_FLOPS") {
        p.big_pair_flops = v;
    }
    if let Some(v) = var("GPU_WIDE_PAIR_K") {
        p.wide_pair_k = v as u32;
    }
    if let Some(v) = var("GPU_WIDE_PAIR_FLOPS") {
        p.wide_pair_flops = v;
    }
    if let Some(v) = var("GPU_CHAIN_FLOPS_F32") {
        p.chain_flops_f32 = v;
    }
    if let Some(v) = var("GPU_CHAIN_FLOPS_F64") {
        p.chain_flops_f64 = v;
    }
    if let Some(v) = var("GPU_CHAIN_ITEMS") {
        p.chain_items = v as u32;
    }
    if let Some(v) = var("GPU_SOLVE_CHAIN_WORK") {
        p.solve_chain_work = v;
    }
    if let Some(v) = var("GPU_SOLVE_WORK") {
        p.solve_work = v as usize;
    }
    if let Some(v) = var("GPU_BLOCKS_PER_SM") {
        p.blocks_per_sm = v as u32;
    }
    p
}

/// The supernodes whose factor values differ from the CPU's by more than
/// 1e-9 of the panel's largest entry, with the first such entry.
fn check_factor(sn: &SupernodalSymbolic, got: &[f64], cpu: &[f64]) {
    let mut bad = 0;
    for s in 0..sn.n_supernodes() {
        let (q, h) = sn.supernode_dims(s);
        let base = sn.panel_range(s).start;
        let max = (0..q).flat_map(|c| (c..h).map(move |r| (c, r))).fold(0.0f64, |m, (c, r)| m.max(cpu[base + c * h + r].abs()));
        let first = (0..q).flat_map(|c| (c..h).map(move |r| (c, r))).find(|&(c, r)| (got[base + c * h + r] - cpu[base + c * h + r]).abs() > 1e-9 * max);
        if let Some((c, r)) = first {
            bad += 1;
            if bad <= 8 {
                println!("  supernode {s} ({q} x {h}): first difference at column {c}, row {r}: {} against {}", got[base + c * h + r], cpu[base + c * h + r]);
            }
        }
    }
    println!("  {bad} of {} supernodes differ", sn.n_supernodes());
}

fn run<D: Device>(dev: Arc<D>, a: &SparseBlockColMat<SparseIndex, f64>, rhs: &[f64], sn: &SupernodalSymbolic, cpu_factor: &[f64], x_cpu: &[f64], rounds: usize) {
    let n = rhs.len();
    let mut vals = dev.alloc(8 * a.vals().len()).unwrap();
    dev.write(&mut vals, 0, bytemuck::cast_slice(a.vals())).unwrap();
    let fail = dev.alloc(4).unwrap();
    let mut xb = dev.alloc(8 * n).unwrap();
    for precision in [GpuPrecision::F64, GpuPrecision::F32] {
        let mut f = GpuFactor::new(&dev, precision, sn, a.symbolic(), &params(), Wide::F64).unwrap();
        // through the host API once (compiles the kernels), checked; a matrix
        // not positive definite in this precision is reported and still timed:
        // a bad pivot continues with 1, the kernels are the same
        match f.factorize(a.vals()) {
            Ok(()) => {}
            Err(FactorizeError::NotPositiveDefinite) => println!("{:<4} {precision:?}: not positive definite in this precision", D::RUNTIME),
            Err(e) => panic!("{precision:?} factorization: {e:?}"),
        }
        if std::env::var_os("GPU_CHECK_FACTOR").is_some() {
            if let GpuFactor::F64(g) = &f {
                check_factor(sn, &g.factor_values().unwrap(), cpu_factor);
            }
        }
        let mut x = rhs.to_vec();
        f.solve(&mut x).unwrap();
        let diff = rel_diff(&x, x_cpu);
        if std::env::var_os("GPU_CHECK_FACTOR").is_some() && precision == GpuPrecision::F64 {
            // factorize and solve again and again: an intermittent fault shows
            let (mut worst, mut off) = (0.0f64, 0);
            for _ in 0..50 {
                f.factorize(a.vals()).unwrap();
                let mut y = rhs.to_vec();
                f.solve(&mut y).unwrap();
                let d = rel_diff(&y, x_cpu);
                worst = worst.max(d);
                off += (d > 1e-8) as usize;
            }
            println!("  50 more factorizations and solves: {off} off, the worst {worst:.1e}");
        }
        let factor = times(rounds, || {
            f.factorize_device((&vals, 0), (&fail, 0)).unwrap();
            dev.sync().unwrap();
        });
        dev.write(&mut xb, 0, bytemuck::cast_slice(rhs)).unwrap();
        let solve = times(rounds, || {
            f.solve_device((&xb, 0)).unwrap();
            dev.sync().unwrap();
        });
        println!(
            "{:<4} {:?}: factor {:8.2} ms (min {:8.2}), solve {:7.2} ms (min {:7.2}); solution against the CPU's {:.1e}",
            D::RUNTIME,
            precision,
            median(factor.clone()),
            factor.iter().cloned().fold(f64::INFINITY, f64::min),
            median(solve.clone()),
            solve.iter().cloned().fold(f64::INFINITY, f64::min),
            diff
        );
    }
}

/// Each supernode's level in `sn`'s tree, one above its highest child.
fn levels_of(sn: &SupernodalSymbolic) -> Vec<usize> {
    let ns = sn.n_supernodes();
    let mut level = vec![0usize; ns];
    for s in 0..ns {
        if let Some(p) = sn.supernode_parent(s) {
            assert!(p > s, "a supernode's parent comes after it");
            level[p] = level[p].max(level[s] + 1);
        }
    }
    level
}

/// The levels of `sn`'s tree, a supernode one above its highest child.
fn print_levels(sn: &SupernodalSymbolic) {
    let ns = sn.n_supernodes();
    let level = levels_of(sn);
    // the multiply-adds of s's descendant updates: per descendant, its rows
    // in s's columns (kw) against its rows from there on (m)
    let update_work = |s: usize| -> (usize, f64) {
        let cols = sn.supernode_cols(s);
        let mut work = 0.0;
        for &d in sn.descendants(s) {
            let d = d as usize;
            let (dq, dh) = sn.supernode_dims(d);
            let (pat, rows) = (sn.supernode_pattern(d), sn.supernode_pattern_rows(d));
            let e0 = pat.partition_point(|&b| (b as usize) < cols.start);
            let e1 = pat.partition_point(|&b| (b as usize) < cols.end);
            let row = |e: usize| if e < rows.len() { rows[e] as f64 } else { dh as f64 };
            let (kw, m) = (row(e1) - row(e0), dh as f64 - row(e0));
            work += dq as f64 * (kw * m - kw * (kw - 1.0) / 2.0);
        }
        (sn.descendants(s).len(), work)
    };
    let nl = level.iter().max().map_or(0, |&l| l + 1);
    // over the levels: the widest supernode's columns, the largest
    // supernode's factor and the level's multiply-adds
    let (mut sum_q, mut sum_fac, mut sum_work) = (0.0, 0.0, 0.0);
    for l in 0..nl {
        let sns: Vec<usize> = (0..ns).filter(|&s| level[s] == l).collect();
        let (mut q, mut h, mut pairs, mut upd, mut fac, mut big) = (0, 0, 0, 0.0, 0.0, 0.0f64);
        for &s in &sns {
            let (a, b) = sn.supernode_dims(s);
            let (np, w) = update_work(s);
            (q, h, pairs, upd) = (q.max(a), h.max(b), pairs + np, upd + w);
            let (a, b) = (a as f64, b as f64);
            let f = a * a * a / 6.0 + (b - a) * a * a / 2.0;
            fac += f;
            big = big.max(f);
        }
        (sum_q, sum_fac, sum_work) = (sum_q + q as f64, sum_fac + big, sum_work + upd + fac);
        println!(
            "  level {l:3}: {:5} supernodes, widest {q:4}, tallest {h:5}, {pairs:6} descendant pairs, updates {:8.0}k, factor {:8.0}k multiply-adds",
            sns.len(),
            upd / 1e3,
            fac / 1e3
        );
    }
    println!("  over the levels: widest columns {sum_q:.0}, largest factor {sum_fac:.4e}, multiply-adds {sum_work:.4e}");
}

/// `a`'s factorization (in its AMD block order, as the solver's) and the
/// solve of `rhs`, on the CPU and on the GPU ARAEL_GPU names; `what` heads
/// the report.
pub fn compare(what: &str, a: &SparseBlockColMat<SparseIndex, f64>, rhs: &[f64]) {
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let threads: usize = std::env::var("BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let ordering = std::env::var("GPU_ORDER").unwrap_or_else(|_| "amd".to_string());
    let order = match ordering.as_str() {
        "amd" => amd_block_order(a.symbolic()),
        "nd" => nd_block_order(a.symbolic()),
        other => panic!("GPU_ORDER={other}: amd or nd"),
    };
    let sn = SupernodalSymbolic::new(a.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
    println!(
        "{what}, {ordering} order: {} params, {} supernodes in {} levels, {} factor values, {:.3e} flops; {rounds} rounds, the CPU on {threads} threads",
        rhs.len(),
        sn.n_supernodes(),
        levels_of(&sn).iter().max().map_or(0, |&l| l + 1),
        sn.factor_val_count(),
        sn.flops()
    );
    if std::env::var_os("GPU_LEVELS").is_some() {
        print_levels(&sn);
    }
    let mut factor = vec![0.0f64; sn.factor_val_count()];
    let mut ctx = SupernodalContext::new();
    supernodal_factorize(&sn, a, &mut factor, &mut ctx, threads).unwrap();
    let mut x_cpu = rhs.to_vec();
    supernodal_solve(&sn, &factor, &mut x_cpu, &mut ctx, threads);
    let cpu = times(rounds, || supernodal_factorize(&sn, a, &mut factor, &mut ctx, threads).unwrap());
    let cpu_solve = times(rounds, || {
        let mut x = rhs.to_vec();
        supernodal_solve(&sn, &factor, &mut x, &mut ctx, threads);
    });
    println!("cpu  F64: factor {:8.2} ms, solve {:7.2} ms", median(cpu), median(cpu_solve));
    match std::env::var("ARAEL_GPU").as_deref().unwrap_or("cuda") {
        #[cfg(feature = "gpu-cuda")]
        "cuda" => run(Arc::new(arael::__gpu::cuda::CudaDevice::new(0).unwrap()), a, rhs, &sn, &factor, &x_cpu, rounds),
        #[cfg(feature = "gpu-hip")]
        "hip" => run(Arc::new(arael::__gpu::hip::HipDevice::new(0).unwrap()), a, rhs, &sn, &factor, &x_cpu, rounds),
        other => panic!("ARAEL_GPU={other}: not a runtime this binary was built with (features gpu-cuda, gpu-hip)"),
    }
}
