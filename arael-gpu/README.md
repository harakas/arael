# arael-gpu

The GPU runtime and the sparse solvers of [arael](https://crates.io/crates/arael)
on a GPU: the Schur reduction, the block supernodal Cholesky factorization,
preconditioned conjugate gradients and the vector passes of the
Levenberg-Marquardt (LM) loop.

Most programs use it through arael: turn on one of arael's features
`gpu-cuda`, `gpu-hip`, `gpu-metal` or `gpu-cpu` and give the solver a GPU.

```rust,ignore
let opts = SparseFaerOptions::default().with_gpu(GpuDevice::detect());
```

The section "Solving on a GPU" of `docs/SOLVERS.md` in the arael repository
describes solving on a GPU through arael. Use this crate directly to run the
solvers, or kernels of your own, on a device.

## Runtimes

| Feature | Runtime | GPUs | Needed at run time |
|---|---|---|---|
| `cuda` | CUDA | NVIDIA | the CUDA driver library (`libcuda`) and NVRTC, CUDA's run-time compiler (`libnvrtc`) |
| `hip` | HIP | AMD | the HIP runtime (`libamdhip64`) and hipRTC, HIP's run-time compiler (`libhiprtc`), from ROCm |
| `metal` | Metal | Apple silicon (M1 on) | nothing: Metal ships with macOS |
| `cpu` | the CPU device | none | a C++17 compiler (GCC or Clang) on Linux |

The libraries are loaded at run time, so a build needs neither the CUDA
toolkit nor ROCm, and a Metal build needs no Xcode. Without a runtime
feature there is no device. The CPU device runs the GPU kernels on the host,
compiled by `ARAEL_CPU_CXX`, else `CXX`, else `c++`: it is slow, for tests
on machines without a GPU.

## Example

```rust,ignore
use std::sync::Arc;
use arael_gpu::{cuda::CudaDevice, GpuFactor, GpuPrecision, GpuSchur, Wide};

let dev = Arc::new(CudaDevice::new(0)?);
// built once per sparsity structure, from arael-faer's symbolic analysis
let mut schur = GpuSchur::new(&dev, &schur_sym, h.symbolic(), Wide::F64)?;
let mut factor = GpuFactor::new(&dev, GpuPrecision::F32, &sn_of_s, &schur_sym.s, &Default::default(), Wide::F64)?;
// one damped solve: reduce, factorize, refine up to 4 steps, back-substitute; one wait
let stats = schur.solve(&mut factor, h.vals(), &rhs, Some((4, 1e-10)), &mut delta)?;
```

`GpuCg` solves the reduced system by conjugate gradients in place of the
factor, and `LmVectors` keeps the LM loop's vectors on the device. The crate
docs describe each type.

## Testing

```sh
cargo test -p arael-gpu --features cuda    # on an NVIDIA GPU
cargo test -p arael-gpu --features hip     # on an AMD GPU
cargo test -p arael-gpu --features metal   # on an Apple silicon Mac
cargo test -p arael-gpu --features cpu     # without a GPU
```

The tests compare the device's results with arael-faer's host
factorization and reduction, in f64, f32 and df32.

## Kernel cache

Compiled kernels are kept on disk between processes, so only the first run
on a machine compiles them: in `~/.cache/arael/kernels` on Linux, or the
directory `ARAEL_KERNEL_CACHE` names (`off` keeps nothing), or the one
`set_kernel_cache` sets. Metal keeps its own cache.

## Tracing

`ARAEL_GPU_TRACE=<file>` (or `trace::start`) writes every operation of a
traced device into the file, one JSON object per line: allocations, copies,
launches with their arguments, graph captures and replays. arael, built
with its feature `gpu-trace`, traces its devices; a direct user wraps a
device with `trace::traced`. `ARAEL_GPU_TRACE_TIME=1` adds each launch's
time on the device. `tools/devmap` in the arael repository turns a trace
into a browsable map of a solve.

## Settings

The solvers take their settings in `GpuSupernodalParams`, `GpuSchurParams`
and `GpuCgParams`. Each field's rustdoc gives its unit and allowed values;
the rustdoc of `GpuSupernodalParams` defines the terms used here
(supernode, update, level, panel steps). In the defaults, discrete means a
discrete GPU and integrated an integrated GPU or the CPU device.

`GpuSupernodalParams` (the factorization):

| Field | Meaning | Default |
|---|---|---|
| `small_q` | supernodes narrower than this many columns are factored by one block each | 96 |
| `small_flops` | the most multiply-adds of a supernode one block factors | 3e5 |
| `big_pair_flops` | an update of at least this many multiply-adds is tiled over many blocks | 2e6 |
| `wide_pair_k`, `wide_pair_flops` | so is an update from a supernode at least this wide, from this many multiply-adds on | 64, 1e5 |
| `chain_flops_f32`, `chain_flops_f64` | the most multiply-adds of a level in a chain (consecutive levels one block factors in one launch), by factor scalar; 0 is off | 3e5, 0 |
| `chain_items` | the most update pieces of a level in a chain | 32 |
| `solve_chain_work` | the most multiply-adds of a level's solve in a solve chain | 1e5 |
| `blocks_per_sm` | blocks per multiprocessor a level's updates are cut to make | 4 |
| `gather_work` | multiply-adds per block of the forward solve's gather | 65536 |
| `gemv_work` | multiply-adds per block of the backward solve's product | 2048 discrete, 16384 integrated |
| `lane_unroll` | loads per loop pass of the backward solve's dot products | 8 in f64, 4 in f32 |
| `syrk_block` | the column group of the panel steps' trailing update | 32 discrete, 256 Apple, other integrated 256 in f32 and 128 in f64 |
| `syrk_tile` | the tile of the trailing update: 32 or 64 | 64 in f32 integrated, else 32 |
| `trsm_rows` | rows per block of the panel steps' solve below a diagonal tile | the warp width discrete, 64 integrated |
| `potf2_unroll` | whether a diagonal tile's pivot steps are unrolled | on discrete, off integrated |
| `diag_warps` | warps per diagonal tile: 1 or 4 | 4 NVIDIA and AMD, 1 Apple and the CPU device |
| `trsv_wide` | the narrowest supernode whose triangular solves run over many blocks | 512 discrete, 1024 integrated |
| `trsv_block` | columns per launch of those solves | 128 discrete, 256 integrated |
| `pair_tiles` | whether a big update is tiled in its own rows and columns | off in f32 discrete, else on |
| `lookahead` | whether a column group's trailing update overlaps the next group's steps, on the side stream | off |

`GpuSchurParams` (the reduction):

| Field | Meaning | Default |
|---|---|---|
| `tile_warps` | units per block forming S, a tile each | 4 discrete, 1 integrated |
| `split_pairs` | a tile of more pairs takes a whole block | 128 |
| `tile_groups` | lane groups per unit: 1, 2, 4 or 8 | 1 or 2, by the tiles |
| `f32_reduction` | whether the reduction computes in f32 | off |
| `kept_warps` | warps per kept block in the reduced right-hand side and the residual | by the structure |

`GpuCgParams` (conjugate gradients):

| Field | Meaning | Default |
|---|---|---|
| `graph_iters` | iterations per captured graph | 8 |
| `first_iters` | iterations the first round of the first solve queues | 16 |
| `matvec_warps` | warps per kept block in the product with S | by the structure |
