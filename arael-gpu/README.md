# arael-gpu

GPU kernels for arael's sparse solvers, written once in CubeCL and run on
CUDA or HIP: the block supernodal Cholesky in arael's panel layout and its
level-set solve, and the Schur reduction that feeds it.

Everything runs on a `GpuContext`: a stream of its own and an arena that
every device buffer is leased from and returned to. Solving problem after
problem through one context stops creating buffers once the largest has
been seen (`reserved_bytes`, `buffers_created`); separate contexts run on
separate streams.

The symbolic analysis stays on the CPU in arael-faer. `GpuSupernodal<F>` is
built from a `SupernodalSymbolic` and the matrix's block structure once per
sparsity structure; it then factorizes any matrix of that structure from
the matrix's tile values and solves against the factor it keeps on the
device, in `f64` or in `f32` (which arael refines against the f64 matrix).

```rust,ignore
use arael_gpu::{GpuBackend, GpuContext, GpuFactor, GpuPrecision};

let ctx = GpuContext::new(GpuBackend::Cuda, 0)?;
let mut g = GpuFactor::new(&ctx, GpuPrecision::F32, &sn, a.symbolic(), &Default::default());
g.factorize(a.vals())?;      // Err when a diagonal pivot is not positive
g.solve(&mut rhs);           // rhs becomes x, in the matrix's ordering
```

`GpuSchur` is built from a `SchurSymbolic` and the Hessian's block structure.
It forms the reduced system on the device from the Hessian's tile values and
the gradient, in `f64` (an `f32` reduction loses the cancelling sums'
digits), and then factorizes it with a `GpuFactor`, solves, refines the
solution against it when the factor is `f32`, and recovers the eliminated
blocks, all on the device, waiting for it once, for the result:

```rust,ignore
let mut g = GpuSchur::new(&ctx, &schur_sym, h.symbolic());
let mut f = GpuFactor::new(&ctx, GpuPrecision::F32, &sn_of_s, &schur_sym.s, &Default::default());
// the full step; Err when an eliminated block or the reduced system is not definite
let stats = g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta)?;
```

The refinement decides on the device whether the next step is due; the
steps launched after it has converged do nothing. The first `solve` with a
factor and a refinement setting runs its kernels and then captures them as
a graph (CUDA or HIP), and the solves after it replay that graph, one
launch instead of hundreds (`use_graphs(false)` turns this off). Each
structure's integer data goes up in one copy; the matrix values are read
from the caller's slices in place (`host::write_borrowed`).

Through arael, the same is `SparseFaerOptions::default().with_gpu(Some(GpuDevice::cuda(0)))`
with arael's `gpu-cuda` (or `gpu-hip`) feature; see docs/SOLVERS.md.

## Features

- `cuda`, `hip`: the runtimes. Without one the kernels compile (CubeCL's
  frontend needs no device) and `client` reports the backend unavailable.
  The first build of a runtime fetches CubeCL's LLVM bundle and takes a
  few minutes.

## Testing

```sh
cargo test -p arael-gpu                   # the plan: every update element covered once, every solve row and column
cargo test -p arael-gpu --features cuda   # the device against arael-faer's CPU factorization and Schur reduction, f64 and f32, bad pivots
```

`examples/repeat_solves.rs` solves several structures several times the way
arael does, prints the wall time per attempt (`--no-graphs` for launched
kernels), and writes marker copies between phases; `scripts/gpu_trace.py`
traces any command with nsys and counts, per phase, launches, blocking
syncs, copies, device and pinned allocations, and graph instantiations:

```sh
cargo build -r -p arael-gpu --features cuda --example repeat_solves
scripts/gpu_trace.py -- target/release/examples/repeat_solves
```

`cargo run -r -p arael-gpu --features cuda --example async_copies` times
CubeCL's copies: enqueue against completion, several in flight, and an
upload and a download on two streams.

## Knobs

`GpuSupernodalParams`: the small-supernode width (96 columns), the
multiply-adds from which a descendant update takes the GEMM-core kernel
(2e6), the elements per cube of the small-update kernel (16384), the
multiply-adds per cube of the solve's gather and GEMV (65536), and the
loads in flight of the backward dot products (8 in f64, 4 in f32). The
defaults are the measured ones (docs/dev/GPU.md, E3 and E5 step 7).
