//! Repeated solves through the device route, the way a COLMAP-style caller
//! runs them: several structures of different sizes, each solved several
//! times by a fresh reduction and factor, each solve a few damped attempts
//! (reduce, factorize, solve, refine, back-substitute). All solves share one
//! device, so its pool's blocks serve them all: after each structure the
//! device's allocations and reserved bytes are printed, and a structure no
//! larger than about 1.25 times one seen before allocates nothing.
//! `--device-per-solve` gives every solve a device of its own instead;
//! `--no-graphs` launches every attempt's kernels instead of replaying the
//! first one's graph; `--attempts N` sets the attempts per solve (4);
//! `--hip` runs on HIP device 0 (built with `--features hip`) instead of
//! CUDA.
//!
//! Between phases it writes a marker copy of a unique size, so an API trace
//! can be cut into phases: `scripts/gpu_trace.py` reports allocations,
//! synchronizations, copies and launches per phase.
//!
//!   cargo run -r -p arael-gpu --features cuda --example repeat_solves
//!   scripts/gpu_trace.py -- target/release/examples/repeat_solves

// Without a runtime feature there is no device to run on.
#![cfg_attr(not(any(feature = "cuda", feature = "hip")), allow(dead_code))]

use std::sync::Arc;

use arael_faer::schur::schur_symbolic;
use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};
use arael_gpu::test_util::ba_like;
use arael_gpu::{Device, GpuFactor, GpuPrecision, GpuSchur, Wide};

/// Marker copies are `MARKER + id` bytes; the trace script knows the range.
const MARKER: usize = 1_000_003;

/// A copy of `MARKER + id` bytes into `block`, then a wait.
fn marker<D: Device>(dev: &D, block: &mut D::Block, staged: &[u8], id: usize) {
    dev.write(block, 0, &staged[..MARKER + id]).unwrap();
    dev.sync().unwrap();
}

fn run<D: Device>(new: impl Fn() -> D) {
    let f64_factor = std::env::args().any(|a| a == "--f64");
    let per_solve = std::env::args().any(|a| a == "--device-per-solve");
    let graphs = !std::env::args().any(|a| a == "--no-graphs");
    let args: Vec<String> = std::env::args().collect();
    let attempts: usize = args.iter().position(|a| a == "--attempts").map_or(4, |i| args[i + 1].parse().expect("--attempts N"));
    let shared = Arc::new(new());
    // the markers' block and bytes, made before the first phase
    let staged = vec![0u8; MARKER + 4096];
    let mut mblock = shared.alloc(MARKER + 4096).unwrap();
    // (poses, landmarks, seed): sizes going up and down, then a repeat of
    // the second's size with another structure, then one 5% larger
    let structures = [(40usize, 500usize, 1u64), (80, 1200, 2), (20, 250, 3), (80, 1200, 4), (84, 1260, 5)];
    let solves = 3usize;
    let precision = if f64_factor { GpuPrecision::F64 } else { GpuPrecision::F32 };
    let refine = (!f64_factor).then_some((4usize, 1e-10f64));
    println!(
        "{} structures x {solves} solves x {attempts} attempts, {precision:?} factor, {}, {}",
        structures.len(),
        if per_solve { "a device per solve" } else { "one shared device" },
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
            marker(&*shared, &mut mblock, &staged, 2 * phase + 1);
            // a fresh solver's device state, from the device's pool
            let dev = if per_solve { Arc::new(new()) } else { shared.clone() };
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), Wide::F64).unwrap();
            g.use_graphs(graphs);
            let mut f = GpuFactor::new(&dev, precision, &sn, &sym.s, &Default::default(), Wide::F64).unwrap();
            marker(&*shared, &mut mblock, &staged, 2 * phase + 2);
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
            "structure {si}: {} scalars, reduced {}, {} supernodes; ms per attempt, median: first {:.2}, later {:.2}; \
             device: {} allocations, {:.1} MB reserved",
            n,
            sym.kept_size(),
            sn.n_supernodes(),
            median(&mut first),
            median(&mut later),
            shared.allocations(),
            shared.reserved_bytes() as f64 / 1e6
        );
    }
    marker(&*shared, &mut mblock, &staged, 4095);
}

fn main() {
    if std::env::args().any(|a| a == "--hip") {
        #[cfg(feature = "hip")]
        return run(|| arael_gpu::hip::HipDevice::new(0).unwrap());
        #[cfg(not(feature = "hip"))]
        panic!("--hip needs the hip feature");
    }
    #[cfg(feature = "cuda")]
    run(|| arael_gpu::cuda::CudaDevice::new(0).unwrap());
    #[cfg(not(feature = "cuda"))]
    panic!("CUDA needs the cuda feature (or --hip with the hip feature)");
}
