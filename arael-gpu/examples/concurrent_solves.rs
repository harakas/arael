//! Many small solves at once, the way a caller with a thread pool runs
//! them: a fixed set of solves (small bundle-adjustment structures, a fresh
//! reduction and factor per solve, several damped attempts each) spread
//! over N threads, each with a `GpuContext` of its own and so a stream of
//! its own, against the same set on one thread. Prints the wall time and
//! the solves per second for each N.
//!
//!   cargo run -r -p arael-gpu --features cuda --example concurrent_solves -- \
//!       [--threads 1,2,4,8] [--solves 48] [--attempts 10] [--no-graphs] [--f64]

use std::time::Instant;

use arael_faer::schur::schur_symbolic;
use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};
use arael_gpu::test_util::{ba_like, rel_residual};
use arael_gpu::{GpuBackend, GpuContext, GpuFactor, GpuPrecision, GpuSchur};

fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].parse().ok().unwrap_or_else(|| panic!("{name}: a value")))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: Vec<usize> = arg::<String>(&args, "--threads").map_or(vec![1, 2, 4, 8], |v| v.split(',').map(|t| t.parse().expect("--threads 1,2,4")).collect());
    let total: usize = arg(&args, "--solves").unwrap_or(48);
    let attempts: usize = arg(&args, "--attempts").unwrap_or(10);
    let graphs = !args.iter().any(|a| a == "--no-graphs");
    let f64_factor = args.iter().any(|a| a == "--f64");
    let precision = if f64_factor { GpuPrecision::F64 } else { GpuPrecision::F32 };
    let refine = (!f64_factor).then_some((4usize, 1e-10f64));

    // (poses, landmarks, seed): small structures, as most COLMAP solves are
    let problems: Vec<_> = [(20usize, 250usize, 3u64), (40, 500, 1), (80, 1200, 2)]
        .iter()
        .map(|&(np, nl, seed)| {
            let (h, dense, rhs, elim) = ba_like(np, nl, seed);
            let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
            let order = nd_block_order(&sym.s);
            let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
            (h, dense, rhs, sym, sn)
        })
        .collect();
    let problems = &problems;
    let max_threads = threads.iter().copied().max().unwrap_or(1);
    let contexts: Vec<GpuContext> = (0..max_threads).map(|_| GpuContext::new(GpuBackend::Cuda, 0).unwrap()).collect();

    // one solve of problem `p` on `ctx`, checked
    let solve = |ctx: &GpuContext, p: usize| {
        let (h, dense, rhs, sym, sn) = &problems[p];
        let n = h.symbolic().ncols();
        let mut g = GpuSchur::new(ctx, sym, h.symbolic());
        g.use_graphs(graphs);
        let mut f = GpuFactor::new(ctx, precision, sn, &sym.s, &Default::default());
        let mut delta = vec![0.0f64; n];
        for _ in 0..attempts {
            g.solve(&mut f, h.vals(), rhs, refine, &mut delta).unwrap();
        }
        let res = rel_residual(dense, n, &delta, rhs);
        assert!(res < 1e-8, "residual {res}");
    };
    // warm-up: every context compiles nothing new and grows its arena to the
    // largest problem before anything is timed
    std::thread::scope(|s| {
        for ctx in &contexts {
            s.spawn(|| (0..problems.len()).for_each(|p| solve(ctx, p)));
        }
    });

    println!(
        "{total} solves of {} structures (n {:?}), {attempts} attempts each, {precision:?} factor, {}",
        problems.len(),
        problems.iter().map(|p| p.0.symbolic().ncols()).collect::<Vec<_>>(),
        if graphs { "graphs" } else { "launches" }
    );
    let mut base = None;
    for &nt in &threads {
        let t = Instant::now();
        std::thread::scope(|s| {
            for (k, ctx) in contexts.iter().take(nt).enumerate() {
                s.spawn(move || {
                    for i in (k..total).step_by(nt) {
                        solve(ctx, i % problems.len());
                    }
                });
            }
        });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let base_ms = *base.get_or_insert(ms);
        println!("{nt} threads, {nt} contexts: {ms:8.1} ms, {:7.1} solves/s, {:.2}x", total as f64 / ms * 1e3, base_ms / ms);
    }
}
