//! The reduced system's block supernodal factorization and solve on a GPU
//! against the CPU's, on the slam problem's S: ms per factorization and
//! per solve on the device (S uploaded once, timed to the device's wait),
//! in f64 and with the f32 factor, the solutions checked against the
//! CPU's. The device run is what an Nsight Compute session profiles.
//!
//!     SLAM_POSES=1000 ROUNDS=20 cargo run -r --features gpu-cuda --bin gpu_factor_bench
//!     ARAEL_GPU=hip SLAM_POSES=300 cargo run -r --features gpu-hip --bin gpu_factor_bench
//!
//! BENCH_THREADS sets the CPU factorization's threads (8); GPU_SMALL_Q,
//! GPU_SMALL_FLOPS, GPU_BIG_PAIR_FLOPS, GPU_WIDE_PAIR_K, GPU_WIDE_PAIR_FLOPS,
//! GPU_ELEM_CAP, GPU_SOLVE_WORK and GPU_BLOCKS_PER_SM override the device's
//! `GpuSupernodalParams`. GPU_CHECK_FACTOR=1 compares the f64 factor with
//! the CPU's, supernode by supernode, and prints where they differ.

#[path = "../scene.rs"]
mod scene;
#[path = "../arael_runner.rs"]
mod arael_runner;
#[path = "../reduced.rs"]
mod reduced;

use std::sync::Arc;
use std::time::Instant;

use arael::__gpu::{Device, GpuFactor, GpuPrecision, GpuSupernodalParams, Wide, bytemuck};
use arael_faer::SparseIndex;
use arael_faer::supernodal::{SupernodalContext, SupernodalParams, SupernodalSymbolic, amd_block_order, supernodal_factorize, supernodal_solve};
use scene::SceneConfig;

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
    if let Some(v) = var("GPU_ELEM_CAP") {
        p.elem_cap = v as usize;
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

fn run<D: Device>(dev: Arc<D>, r: &reduced::Reduced<SparseIndex>, sn: &SupernodalSymbolic, cpu_factor: &[f64], x_cpu: &[f64], rounds: usize) {
    let n = r.rhs.len();
    let mut vals = dev.alloc(8 * r.s.vals().len()).unwrap();
    dev.write(&mut vals, 0, bytemuck::cast_slice(r.s.vals())).unwrap();
    let fail = dev.alloc(4).unwrap();
    let mut xb = dev.alloc(8 * n).unwrap();
    for precision in [GpuPrecision::F64, GpuPrecision::F32] {
        let mut f = GpuFactor::new(&dev, precision, sn, r.s.symbolic(), &params(), Wide::F64).unwrap();
        // through the host API once (compiles the kernels), checked
        f.factorize(r.s.vals()).unwrap();
        if std::env::var_os("GPU_CHECK_FACTOR").is_some() {
            if let GpuFactor::F64(g) = &f {
                check_factor(sn, &g.factor_values().unwrap(), cpu_factor);
            }
        }
        let mut x = r.rhs.clone();
        f.solve(&mut x).unwrap();
        let diff = rel_diff(&x, x_cpu);
        if std::env::var_os("GPU_CHECK_FACTOR").is_some() && precision == GpuPrecision::F64 {
            // factorize and solve again and again: an intermittent fault shows
            let (mut worst, mut off) = (0.0f64, 0);
            for _ in 0..50 {
                f.factorize(r.s.vals()).unwrap();
                let mut y = r.rhs.clone();
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
        dev.write(&mut xb, 0, bytemuck::cast_slice(&r.rhs)).unwrap();
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

fn main() {
    let mut cfg = SceneConfig::default();
    if let Some(n) = std::env::var("SLAM_POSES").ok().and_then(|v| v.parse().ok()) {
        cfg.num_poses = n;
        cfg.num_landmarks = 4 * n;
    }
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let threads: usize = std::env::var("BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let r = reduced::reduced_system::<SparseIndex>(&cfg);
    let order = amd_block_order(&r.sym.s);
    let sn = SupernodalSymbolic::new(&r.sym.s, Some(&order), &SupernodalParams::default()).unwrap();
    let n = r.rhs.len();
    println!(
        "S: {} poses, {} params, {} supernodes, {} factor values; {} rounds, the CPU on {} threads",
        cfg.num_poses,
        n,
        sn.n_supernodes(),
        sn.factor_val_count(),
        rounds,
        threads
    );
    let mut factor = vec![0.0f64; sn.factor_val_count()];
    let mut ctx = SupernodalContext::new();
    supernodal_factorize(&sn, &r.s, &mut factor, &mut ctx, threads).unwrap();
    let mut x_cpu = r.rhs.clone();
    supernodal_solve(&sn, &factor, &mut x_cpu, &mut ctx, threads);
    let cpu = times(rounds, || supernodal_factorize(&sn, &r.s, &mut factor, &mut ctx, threads).unwrap());
    let cpu_solve = times(rounds, || {
        let mut x = r.rhs.clone();
        supernodal_solve(&sn, &factor, &mut x, &mut ctx, threads);
    });
    println!("cpu  F64: factor {:8.2} ms, solve {:7.2} ms", median(cpu), median(cpu_solve));
    match std::env::var("ARAEL_GPU").as_deref().unwrap_or("cuda") {
        #[cfg(feature = "gpu-cuda")]
        "cuda" => run(Arc::new(arael::__gpu::cuda::CudaDevice::new(0).unwrap()), &r, &sn, &factor, &x_cpu, rounds),
        #[cfg(feature = "gpu-hip")]
        "hip" => run(Arc::new(arael::__gpu::hip::HipDevice::new(0).unwrap()), &r, &sn, &factor, &x_cpu, rounds),
        other => panic!("ARAEL_GPU={other}: not a runtime this binary was built with (features gpu-cuda, gpu-hip)"),
    }
}
