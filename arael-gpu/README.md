# arael-gpu

arael's GPU runtime and the kernels of its sparse solvers, on CUDA and
HIP. The kernels are written once in a C++ subset (`kernels/`) and
compiled at run time for the device at hand: NVRTC on CUDA
(`cuda::CudaDevice`, feature `cuda`), hipRTC on HIP (`hip::HipDevice`,
feature `hip`). Both runtimes load the vendor's libraries at run time, so
a build needs neither the CUDA toolkit nor ROCm. The CPU device
(`cpu::CpuDevice`, feature `cpu`) runs the same kernels on the CPU,
compiled by the host's C++ compiler, for tests and CI without a GPU.

On a device:

- `GpuSupernodal` (and `GpuFactor`, either scalar behind one type): the
  block supernodal Cholesky in arael's panel layout and its level-set
  solve, in `f64` or `f32` (which arael refines against the f64 matrix);
- `GpuSchur`: the Schur reduction from the Hessian's tile values and the
  gradient, then the factorization, the solve, the refinement and the
  back-substitution, waiting for the device once, for the result;
- `LmVectors`: the LM loop's vector passes (the trial point, the norms,
  the damping, the predicted reduction).

The symbolic analysis stays on the CPU in arael-faer. Each solver object
is built once per sparsity structure.

```rust,ignore
use arael_gpu::{cuda::CudaDevice, GpuFactor, GpuPrecision, GpuSchur, Wide};

let dev = Arc::new(CudaDevice::new(0)?);
let mut g = GpuSchur::new(&dev, &schur_sym, h.symbolic(), Wide::F64)?;
let mut f = GpuFactor::new(&dev, GpuPrecision::F32, &sn_of_s, &schur_sym.s, &Default::default(), Wide::F64)?;
// the full step; Err when an eliminated block or the reduced system is not definite
let stats = g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta)?;
```

`Wide` is the scalar the device keeps and sums H, the gradient, the cost,
the reduction and the LM vectors in: `f64`, or `df32` (two f32, about 48
bits, f32 arithmetic only) for cards whose f64 is slow.

Every device has a stream of its own and a pool of device memory: a block
that is dropped goes back to the pool and is handed out again, and a
fresh block has room for a quarter more, so solving a structure after a
slightly larger or smaller one allocates nothing (`Device::allocations`,
`Device::reserved_bytes`). Separate devices run on separate streams, so
solves on several threads, a device each, run at the same time.

The refinement decides on the device whether the next step is due; the
steps launched after it has converged do nothing. The first `solve` with a
factor and a refinement setting runs its kernels and then captures them as
a graph, and the solves after it replay that graph, one launch instead of
hundreds (`use_graphs(false)` turns this off). Captures are thread-local:
other threads' work goes on while one is open.

Compiled kernels are kept in a disk cache (`KernelCache`,
`ARAEL_KERNEL_CACHE`), found again by their source, the target and the
compiler's version.

Through arael, the same is `SparseFaerOptions::default().with_gpu(Some(GpuDevice::cuda(0)))`
with arael's `gpu-cuda` (or `gpu-hip`) feature; see docs/SOLVERS.md.

## Features

- `cuda`, `hip`: the runtimes. Without one there is no device, and arael
  reports the backend unavailable.
- `cpu`: the CPU device. The compiler is `ARAEL_CPU_CXX`, else `CXX`,
  else `c++` (GCC or Clang, on Linux); kernels are compiled with
  contraction on and FMA, as on a GPU, and a block's threads run as
  fibers that switch at barriers. Slow: a test target.

The kernels work in warps of 32 lanes: an NVIDIA warp, an AMD RDNA
wavefront, or half of a 64-lane AMD wavefront (GCN, CDNA). They are not
tuned for 64-lane GPUs: a kernel launched with 32 threads per block fills
half a wavefront there.

## Testing

```sh
cargo test -p arael-gpu --features cuda   # the device against arael-faer's CPU factorization and Schur reduction, f64, f32 and df32, bad pivots, the pool, graph capture
cargo test -p arael-gpu --features hip    # the same on HIP
cargo test -p arael-gpu --features cpu    # the same on the CPU device, without a GPU
```

`examples/repeat_solves.rs` solves several structures several times the way
arael does, prints the wall time per attempt and the device's allocations
after each structure (`--no-graphs` for launched kernels, `--hip` for HIP),
and writes marker copies between phases; `scripts/gpu_trace.py` traces any
command with nsys and counts, per phase, launches, blocking syncs, copies,
device and pinned allocations, and graph instantiations:

```sh
cargo build -r -p arael-gpu --features cuda --example repeat_solves
scripts/gpu_trace.py -- target/release/examples/repeat_solves
```

`examples/concurrent_solves.rs` spreads a fixed set of small solves over
1, 2, 4 and 8 threads, a device each, and prints the solves per second.

## Knobs

`GpuSupernodalParams`:

- `small_q` (96 columns) and `small_flops` (3e5): a supernode narrower
  than `small_q` is factored by the one-block small kernel unless the
  solve of its rows below takes more than `small_flops` multiply-adds; the
  others take the panel steps, which spread over many blocks.
- `big_pair_flops` (2e6): a descendant update of at least this many
  multiply-adds takes the GEMM-core kernel; `wide_pair_k` (64) and
  `wide_pair_flops` (1e5): an update from a descendant at least that wide
  takes it from `wide_pair_flops` on, since it reuses each k slab across
  its tile.
- `chain_flops_f32` (3e5), `chain_flops_f64` (0, off) and `chain_items`
  (32): a run of levels without big updates or panel steps, each of at most
  that many multiply-adds and `chain_items` update pieces and factors, is
  factored by one block in one launch (the top of a tree, one small level
  above another). In f64 one block's multiply-add rate is the limit where
  f64 is slow.
- `solve_chain_work` (1e5): a run of levels of one supernode each, each of
  at most that many multiply-adds of solve, is solved by one block in one
  launch forward and one backward.
- `blocks_per_sm` (4): a level whose big updates make fewer blocks than
  this per multiprocessor has its tiles' updates cut across blocks (by
  rectangles, then along k), the shared tiles subtracted atomically; a
  level's small updates are cut into pieces of its share of the work.
- `gather_work` (65536): the multiply-adds per block of the solve's
  gather, which also takes at most 256 rows per block.
- `gemv_work`: the multiply-adds per block of the solve's GEMV (the rows
  below a supernode times its solved columns). Unset, 16384 on an
  integrated GPU (and the CPU device) and 2048 on a discrete one: on SLAM
  and pose graph systems an NVIDIA discrete GPU runs the GEMV fastest in
  blocks of 1024 to 4096, an AMD integrated GPU in blocks of 16384.
  Discrete AMD GPUs are not measured.
- `lane_unroll`: the loads in flight of the backward dot products (8 in
  f64, 4 in f32).
- `syrk_block`: the columns per block of the panel steps' trailing update,
  a multiple of 32. In blocks of 32 every 32-column step updates
  everything past it; in wider blocks the trailing part is read and
  written once per block. Unset, 128 on an integrated GPU (and the CPU
  device) and 32 on a discrete one: on bundle adjustment and SLAM reduced
  systems blocks of 128 are faster on AMD integrated GPUs, whose memory
  bandwidth bounds the update, and slower on an NVIDIA discrete GPU.
  Discrete AMD GPUs are not measured.
- `syrk_tile`: the rows and columns of the panel steps' trailing update
  tiles, 32 or 64, a block each. The block of a step's first tile also
  factors the next diagonal tile, so smaller tiles shorten each step and
  spread it over more blocks. Unset, 32, except 64 for an `f32` factor on
  an integrated GPU (and the CPU device): on SLAM and pose graph systems
  tiles of 32 are faster in both precisions on an NVIDIA discrete GPU and
  in `f64` on an AMD integrated GPU, tiles of 64 in `f32` there.
  Discrete AMD GPUs are not measured.
- `trsm_rows`: the rows per block of the panel steps' triangular solve,
  at least 32 and a divisor of 1024. Smaller blocks spread a step's rows
  over more blocks, and every block copies the step's 32 x 32 diagonal
  tile. Unset, 64 on an integrated GPU (and the CPU device) and a warp on
  a discrete one: on bundle adjustment and SLAM reduced systems a warp
  per block is faster on an NVIDIA discrete GPU, whose solve launches
  mostly have fewer blocks than multiprocessors, and 64 rows on an AMD
  integrated GPU, which they fill. Discrete AMD GPUs are not measured.
