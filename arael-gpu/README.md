# arael-gpu

GPU kernels for arael's sparse solvers, written once in CubeCL and run on
CUDA or HIP: the block supernodal Cholesky in arael's panel layout and its
level-set solve.

The symbolic analysis stays on the CPU in arael-faer. `GpuSupernodal<F>` is
built from a `SupernodalSymbolic` and the matrix's block structure once per
sparsity structure; it then factorizes any matrix of that structure from
the matrix's tile values and solves against the factor it keeps on the
device, in `f64` or in `f32` (which arael refines against the f64 matrix).

```rust,ignore
use arael_gpu::{GpuBackend, GpuFactor, GpuPrecision, client};

let client = client(GpuBackend::Cuda, 0)?;
let mut g = GpuFactor::new(client, GpuPrecision::F32, &sn, a.symbolic(), &Default::default());
g.factorize(a.vals())?;      // Err when a diagonal pivot is not positive
g.solve(&mut rhs);           // rhs becomes x, in the matrix's ordering
```

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
cargo test -p arael-gpu --features cuda   # the device against arael-faer's CPU factorization, f64 and f32, and a bad pivot
```

## Knobs

`GpuSupernodalParams`: the small-supernode width (96 columns), the
multiply-adds from which a descendant update takes the GEMM-core kernel
(2e6), the elements per cube of the small-update kernel (16384), the
multiply-adds per cube of the solve's gather and GEMV (65536), and the
loads in flight of the backward dot products (8 in f64, 4 in f32). The
defaults are the measured ones (docs/dev/GPU.md, E3 and E5 step 7).
