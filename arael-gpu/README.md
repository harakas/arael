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
  back-substitution, waiting for the device once, for the result. Each
  eliminated block is applied through its Cholesky factor, as on the CPU,
  not through an explicit inverse, which loses digits on ill-conditioned
  blocks (points seen along nearly parallel rays);
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

A device also has a side stream (`Lane::Side`): launches queued on it
between `Device::fork` and `Device::join` run alongside the main stream's.
A fork orders the side's launches after the main stream's work so far, a
join the main stream's next work after the side's; reads and writes follow
the main stream. On CUDA and HIP the side stream runs at the highest
priority; on Metal it is a second queue; the CPU device runs its launches
in order.

```rust,ignore
dev.launch(&update_next, launch, &args)?;
dev.fork()?;
dev.launch(&update_rest, big, &args)?;
dev.launch_on(Lane::Side, &factor_next, small, &args)?; // alongside update_rest
dev.join()?;
```

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
  written once per block. Unset, 256 on an Apple GPU; on another
  integrated GPU (and the CPU device) 256 for an `f32` factor and 128 for
  an `f64` one; 32 on a discrete one. On bundle adjustment and SLAM
  reduced systems wide blocks are faster on AMD integrated GPUs, whose
  memory bandwidth bounds the update (256 against 128: `f32` 0.4 to 3%
  faster, `f64` the same), and slower on an NVIDIA discrete GPU; on an
  Apple M4 Pro blocks of 256 are faster on the large bundle adjustment
  systems and the same on small ones. Discrete AMD GPUs are not measured.
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
- `potf2_unroll`: whether the panel steps factor a 32-column diagonal tile
  with its 32 pivots unrolled and each column broadcast through shared
  memory, instead of a loop with shuffles; the factor is the same. Unset,
  on a discrete GPU and not on an integrated one (or the CPU device):
  unrolled is faster on an NVIDIA discrete GPU and slower on an AMD
  integrated GPU, where its registers lower the occupancy of the whole
  trailing update. Discrete AMD GPUs are not measured.
- `trsv_wide`: the narrowest supernode (columns) whose triangular solves
  run over many blocks, a launch per `trsv_block` columns: one block
  solves the launch's diagonal block, the others take the block before it
  off the rows (forward) or columns (backward) past it. Narrower
  supernodes are solved by one block each; one too wide for a block's
  shared memory always takes the many. Unset, 512 on a discrete GPU, 1024
  on an integrated one (and the CPU device): on an NVIDIA discrete GPU the
  solve of a bundle adjustment's reduced system about halves; an AMD
  integrated GPU, with fewer units, gains only on wider supernodes.
- `trsv_block`: the columns per launch of those solves, a multiple of 32.
  Unset, 128 on a discrete GPU, 256 on an integrated one, where fewer
  launches measured faster. Discrete AMD GPUs are not measured.
- `pair_tiles`: whether the big descendant updates are tiled in each
  pair's own rows and columns, every result added into the target
  atomically, instead of in the target's 64 x 64 tiles, which sum the
  rectangles of several pairs in shared memory but compute each rectangle
  as a whole tile. Pair tiles execute about half the multiply-adds. Unset,
  on an integrated GPU (and the CPU device), and on a discrete one for an
  `f64` factor: an AMD integrated GPU factors 13 to 29% faster with them in
  both precisions, an NVIDIA discrete GPU 5 to 32% faster in `f64`, but
  up to 6% slower in `f32`, where the multiply-adds are cheap and the
  atomic adds are the cost. Discrete AMD GPUs are not measured.
- `lookahead`: whether the panel steps look ahead. A block's last step
  updates the next block's columns first; the rest of its update then runs
  on the main stream while the next block's steps run on the side stream,
  so their chain of small launches overlaps it. Unset, off: slower on an
  Apple M4 Pro, where the two queues' synchronization costs more than the
  overlap gains. On an NVIDIA discrete GPU it pays only on the widest
  supernodes: with `syrk_block` 256, Ladybug-1723's reduced system factors
  8% faster in `f32` and 1% in `f64`, while the smaller bundle adjustment
  systems factor 3 to 10% slower and SLAM systems up to 1.5% slower; in
  blocks of 32, the default there, it is always slower. On an AMD
  integrated GPU, with 2 compute units to fill, it is up to 1% slower.

`GpuSchurParams` (`GpuSchur::with_params`), for the forming of S, which
takes a tile of S per lockstep unit (a warp or a wavefront):

- `tile_warps`: units per block. Unset, 4 on a discrete GPU, where a tile
  of more than `split_pairs` (128) pairs takes a whole block, its units
  summing every fourth pair; 1 on an integrated GPU (and the CPU device),
  a block per tile in storage order. On an NVIDIA discrete GPU 4 units cut
  the reduction's tail; an AMD integrated GPU has none to fill. Discrete
  AMD GPUs are not measured.
- `tile_groups`: lane groups per unit, 1, 2, 4 or 8, each on a different
  pair of the tile, the groups' sums added at the tile's end. More groups
  leave fewer lanes idle on small tiles. Unset, two when they issue more
  than 10% fewer products over the plan's tiles than one, else one. On an
  NVIDIA discrete and an AMD integrated GPU two groups form SLAM's 6 x 6
  tiles in 20% and 11% less time than one, and bundle adjustment's 9 x 9
  tiles in 3% and 15% more. Four groups are slower than two on both.
- `f32_reduction`: whether the reduction computes in f32, as the CPU's f32
  route (the eliminated blocks' factors, W, S, the reduced right-hand side
  and the eliminated blocks' step), for a solve with an f32 factor that
  does not refine against S. Off: in the wide scalar. arael turns it on
  for an f32 solve, unless `GpuDevice::with_wide_schur` keeps the wide
  scalar.
