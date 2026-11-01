//! Repeated solves through the device route, the way a COLMAP-style caller
//! runs them: several structures of different sizes, each solved several
//! times by a fresh reduction and factor, each solve a few damped attempts
//! (reduce, factorize, solve, refine, back-substitute). All solves share one
//! `GpuContext`, so the arena's buffers serve them all; `--context-per-solve`
//! gives every solve a context of its own instead; `--no-graphs` launches
//! every attempt's kernels instead of replaying the first one's graph;
//! `--attempts N` sets the attempts per solve (4); `--hip` runs on HIP
//! device 0 (built with `--features hip`) instead of CUDA.
//!
//! Between phases it writes a marker copy of a unique size, so an API trace
//! can be cut into phases: `scripts/gpu_trace.py` reports allocations,
//! synchronizations, copies and launches per phase.
//!
//!   cargo run -r -p arael-gpu --features cuda --example repeat_solves
//!   scripts/gpu_trace.py -- target/release/examples/repeat_solves

use arael_faer::schur::schur_symbolic;
use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};
use arael_gpu::test_util::ba_like;
use arael_gpu::{GpuBackend, GpuContext, GpuFactor, GpuPrecision, GpuSchur, sync};
use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Marker copies are `MARKER + id` bytes; the trace script knows the range.
const MARKER: usize = 1_000_003;

/// A copy of `MARKER + id` bytes from pinned memory staged once, so the
/// marker itself leaves no trace in the pinned pool.
fn marker(c: &Client, buf: &Handle, pinned: &Bytes, id: usize) {
    c.write(buf, pinned.view(0, MARKER + id).unwrap());
    sync(c);
}

fn main() {
    let f64_factor = std::env::args().any(|a| a == "--f64");
    let backend = if std::env::args().any(|a| a == "--hip") { GpuBackend::Hip } else { GpuBackend::Cuda };
    let per_solve = std::env::args().any(|a| a == "--context-per-solve");
    let graphs = !std::env::args().any(|a| a == "--no-graphs");
    let args: Vec<String> = std::env::args().collect();
    let attempts: usize = args.iter().position(|a| a == "--attempts").map_or(4, |i| args[i + 1].parse().expect("--attempts N"));
    let shared = GpuContext::new(backend, 0).unwrap();
    let c = shared.client().clone();
    let mbuf = c.empty(MARKER + 4096);
    let mut pinned = Bytes::from_bytes_vec(vec![0u8; MARKER + 4096]);
    c.staging(std::iter::once(&mut pinned), false);
    let pinned = pinned.shared();
    // (poses, landmarks, seed): sizes going up and down, the last a repeat of
    // the second's size with another structure
    let structures = [(40usize, 500usize, 1u64), (80, 1200, 2), (20, 250, 3), (80, 1200, 4)];
    let solves = 3usize;
    let precision = if f64_factor { GpuPrecision::F64 } else { GpuPrecision::F32 };
    let refine = (!f64_factor).then_some((4usize, 1e-10f64));
    println!(
        "{} structures x {solves} solves x {attempts} attempts, {precision:?} factor, {}, {}",
        structures.len(),
        if per_solve { "a context per solve" } else { "one shared context" },
        if graphs { "graphs" } else { "launches" }
    );
    for (si, &(np, nl, seed)) in structures.iter().enumerate() {
        let (h, _, rhs, elim) = ba_like(np, nl, seed);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let n = h.symbolic().ncols();
        // wall time per attempt, the first of each solve apart (it captures)
        let (mut first, mut later) = (Vec::new(), Vec::new());
        for r in 0..solves {
            let phase = si * 16 + r;
            marker(&c, &mbuf, &pinned, 2 * phase + 1);
            // a fresh solver's device state, leased from the context
            let ctx = if per_solve { GpuContext::new(backend, 0).unwrap() } else { shared.clone() };
            let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
            g.use_graphs(graphs);
            let mut f = GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
            marker(&c, &mbuf, &pinned, 2 * phase + 2);
            let mut delta = vec![0.0f64; n];
            for a in 0..attempts {
                let t = std::time::Instant::now();
                g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if a == 0 { first.push(ms) } else { later.push(ms) }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        println!(
            "structure {si}: {} scalars, reduced {}, {} supernodes; ms per attempt, median: first {:.2}, later {:.2}",
            n,
            sym.kept_size(),
            sn.n_supernodes(),
            median(&mut first),
            median(&mut later)
        );
    }
    marker(&c, &mbuf, &pinned, 4095);
    println!("arena: {} buffers created, {:.1} MB reserved", shared.buffers_created(), shared.reserved_bytes() as f64 / 1e6);
}
