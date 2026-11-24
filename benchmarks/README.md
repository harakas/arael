# Benchmarks

Each benchmark times arael against other nonlinear least-squares solvers on
one problem family, every system minimizing the same cost:

| directory | problem |
|---|---|
| [bal](bal/README.md) | bundle adjustment on the BAL datasets |
| [slam](slam/README.md) | monocular SLAM, six factor types |
| [pgo](pgo/README.md) | 2D and 3D pose graphs |
| [plane](plane/README.md) | SLAM with plane landmarks |
| [loc](loc/README.md) | localization against a fixed landmark map |

They share [harness](harness): the probes, the timing rules, the table, the
core pin and the environment variables below. Run a benchmark from its
directory with `cargo run -r`; its README says what else it needs (datasets,
the C++ runners) and lists its own variables.

## Environment variables

These mean the same in every benchmark. On/off variables take `0` or `1`.
A value that does not parse stops the run, and so does a variable the
benchmark does not read, or a name the benchmarks no longer read (the
message names the one to use; the list is `RENAMED` in
[harness/src/env.rs](harness/src/env.rs)).

| variable | meaning | default |
|---|---|---|
| `ROUNDS` | interleaved rounds; each cell is the minimum over them | per benchmark |
| `BENCH_THREADS` | threads for every system, and the number of cores the run is pinned to (the last ones); `0` is every core | 1 |
| `BENCH_PIN` | `0`: keep the thread caps, do not pin | pinned |
| `BENCH_SYSTEMS` | comma-separated patterns: only the rows whose label contains one run; a pattern ending in `$` must be the whole label. A filtered run validates only against what ran | every row |
| `BENCH_MAX_ITERS` | every row's full solve stops after this many attempts; a diagnostic, not a table | unset |
| `BENCH_SOLVE_ONLY` | `1`: each row runs its full solve alone, without the warmup and the probes | 0 |
| `BENCH_NO_MEM` | `1`: skip the peak-memory pass | 0 |
| `BENCH_MEM_ITERS` | iterations of the memory pass's solve (not pgo, whose pass runs the full solve) | 3 |
| `BENCH_MEM_EXE` | the binary the memory pass runs instead of this one | this binary |
| `BENCH_COV` | `1`: the covariance benchmark instead of the solves (bal, slam, loc) | 0 |
| `COV_BUDGET_S`, `COV_CAP`, `COV_CELL_CAP_S` | covariance benchmark: seconds spent on one cell, the most repetitions in it, and the seconds over which a cell is not run (`*`) | 5, per benchmark, 120 |
| `VERBOSE` | `1`: arael logs every iteration; the external runners' messages are printed | 0 |
| `TIMING` | `1`: every arael solve prints its phase times (below) | 0 |
| `REPORT` | `1`: each arael row's solve report is printed under the row | 0 |
| `DRIVER` | arael's lambda driver: `nielsen` (gain ratio) or `fixed` | per benchmark |
| `ARAEL_LAMBDA0` | arael's first lambda, for every row and dataset | per benchmark |
| `ARAEL_GPU` | the GPU for the arael rows: `cuda`, `hip` or `metal`, with an optional device index (`cuda:1`), or `cpu`, the CPU device | none |
| `ARAEL_GPU_WIDE` | `f64` or `df32` (two f32s): the device's wide scalar | the device's |
| `ARAEL_GPU_STAGING` | `pinned` or `pageable`: the host memory uploads go through | pinned |
| `ARAEL_F32_FACTOR` | n: factorize in f32 and refine each solve by at most n steps | off |
| `ARAEL_REFINE_TOL` | the relative residual the refinement stops at; needs `ARAEL_F32_FACTOR` | arael's |
| `ARAEL_ORDERING` | `auto`, `amd`, `nd` (nested dissection), `natural` or `marginalize-first`: the elimination ordering of the rows that set one (not plane, loc) | per benchmark |
| `ARAEL_SCHUR` | `auto`, `force` or `never`: whether the rows that may marginalize do (not loc) | per benchmark |
| `ARAEL_ENVELOPE` | `auto`, `always` or `never`: how a reduced system is factored (slam, plane) | auto |
| `ARAEL_CG_TOL`, `ARAEL_CG_MAXITER`, `ARAEL_CG_RESTART` | the conjugate gradient rows' tolerance, iteration cap and true-residual interval (bal, slam) | per benchmark, arael's, arael's |
| `ARAEL_COV_ORDERING` | `auto`, `amd`, `nd` or `natural`: the covariance factorization's ordering (bal, slam) | per benchmark |
| `ARAEL_BLOCK_SUPERNODAL` | `auto`, `1`/`always` or `0`/`off`/`scalar`: when the block supernodal Cholesky factorizes | auto |
| `ARAEL_BLOCK_SUPERNODAL_BATCH` | a ratio (`1.5`), or `0`/`off`: its update batching | arael's |
| `ARAEL_BLOCK_SUPERNODAL_LEAN` | `1`: its memory-lean amalgamation | 0 |

The thread caps of OpenBLAS and OpenMP must be in the environment before the
process starts (OpenBLAS sizes its pool when its library loads):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # the same number as BENCH_THREADS
```

`TIMING=1` prints, under every arael solve (the warmup and the probes
included): `[timing]` -- total, assembly, analysis, linear solve and cost
evaluation; `[setup]` -- the parts of the first solve's setup; `[gpu]` -- the
device time of the assemblies, the attempts and the conjugate gradients; and
`[par]` -- the threaded sweeps.

## On a GPU

A cargo feature builds a GPU runtime into a benchmark, and `ARAEL_GPU` gives
the GPU to its arael rows. Every benchmark has the same features:

| runtime | build | the machine needs | run |
|---|---|---|---|
| CUDA | `--features gpu-cuda` | an NVIDIA GPU, the CUDA driver library and NVRTC | `ARAEL_GPU=cuda cargo run -r --features gpu-cuda` |
| HIP | `--features gpu-hip` | an AMD GPU, the HIP runtime and hipRTC from ROCm | `ARAEL_GPU=hip cargo run -r --features gpu-hip` |
| Metal | `--features gpu-metal` | Apple silicon, macOS | `ARAEL_GPU=metal ARAEL_F32_FACTOR=4 cargo run -r --features gpu-metal` |
| CPU device | `--features gpu-cpu` | a C++17 compiler | `ARAEL_GPU=cpu cargo run -r --features gpu-cpu` |

Metal has no f64: an f64 row that factorizes on it needs `ARAEL_F32_FACTOR`,
or it fails. The CPU device runs the GPU's kernels on the host; it is slow,
for tests and [tools/devmap](../tools/devmap/README.md). `gpu-trace` records
the device's work (`ARAEL_GPU_TRACE`), for tools/devmap.

The header names the GPU the rows are given:

```
=== slam-bench configuration ===
...
arael gpu         : cuda:0 NVIDIA GeForce RTX 4090, wide f64, staging pinned [ARAEL_GPU, ARAEL_GPU_WIDE, ARAEL_GPU_STAGING]
arael f32 factor  : off [ARAEL_F32_FACTOR]
```

`arael gpu` is the device as its runtime lists it, its wide scalar and its
staging; a GPU the build does not find on the machine reads `NOT FOUND`, and
its rows fail. `arael f32 factor` is `off`, or the refinement that
`ARAEL_F32_FACTOR` and `ARAEL_REFINE_TOL` set. Under each arael row given a
GPU, a `gpu:` line says what its full solve did on it:

```
system         total ms     iters    ms/iter  full-iter  full-norm  1st-iter ms     final cost
arael LM f64       3.63      3(3)       1.21       0.81      1.000         1.91      3062.0482
    gpu: cuda:0 NVIDIA GeForce RTX 4090 (sm_89), wide f64 -- assembly, damped solves and the LM loop on the device, f64 factor
```

- `cuda:0 NVIDIA GeForce RTX 4090 (sm_89)`: the device, its name and target
  (`, integrated` follows the target on an integrated GPU).
- `wide f64`: the scalar the device keeps its wide values in (`df32` on Metal).
- After `--`, where the work ran: `assembly, damped solves and the LM loop on
  the device`; or `damped solves on the device, assembly on the host (why)`;
  or `not used: the solve took a route that does not run on a device` (arael
  logs a warning that names the route).
- Last, how the device solves: `f64 factor`, `f32 factor` (an f32 row, or
  `ARAEL_F32_FACTOR`) or `conjugate gradients`.

The external systems always run on the host.

## gpu_factor_bench

`gpu_factor_bench` (in bal, slam, pgo and plane) times one matrix of the
benchmark's problem: its block supernodal factorization and solve on the
GPU, in f64 and with the f32 factor, against the CPU's, the solutions
checked against the CPU's. It prints ms per factorization and per solve.

```sh
ARAEL_GPU=cuda cargo run -r --features gpu-cuda --bin gpu_factor_bench
```

| benchmark | matrix | chosen by |
|---|---|---|
| bal | the reduced camera system S of a Ladybug dataset | `BAL_DATASET` (camera count, 372 default), damped by 1 + `ARAEL_LAMBDA0` (1e-4) |
| slam | the reduced pose system S | `SLAM_POSES` |
| pgo | the whole Hessian | `PGO_DATASET`: `M3500` (default), `city10000`, `sphere2500`, `parking-garage` |
| plane | the whole Hessian | `PLANE_POSES` |

It reads `ROUNDS` (20), `BENCH_THREADS` (the CPU's threads, 8; not pinned)
and `ARAEL_GPU` (`cuda`, the default, `hip` or `metal`; device 0), and needs
`gpu-cuda`, `gpu-hip` or `gpu-metal`. `GPU_ORDER` picks the ordering (`amd`
or `nd`), `GPU_CHECK_FACTOR=1` compares the f64 factor with the CPU's,
`GPU_LEVELS=1` prints the elimination tree's levels, `GPU_PROFILE=1` (Metal)
times each kernel; the `GPU_*` kernel parameters are listed in
[harness/src/gpu_factor.rs](harness/src/gpu_factor.rs).

## Reading a table

One cell per system, the minimum over the rounds:

| column | meaning |
|---|---|
| total ms | the whole solve: setup and every iteration, retries included |
| iters | `accepted(attempts)`. An attempt is one linear solve; a rejected step raises the damping and costs a factorization. One number where a system reports no attempts |
| ms/iter | total ms over attempts, the setup included |
| full-iter | one complete iteration (linearize, assemble, factorize, solve), measured as t(2 iterations) - t(1 iteration) so the setup cancels |
| full-norm | full-iter over the reference row's (named in each README) |
| 1st-iter ms | one iteration and the setup the later ones do not pay again |
| peak MB | the process's peak resident memory (`VmHWM`), each system in a process of its own |
| final cost | the one reference cost function of the benchmark, evaluated on each system's result |

full-iter is the number that compares across systems. Total time and the
iteration count depend on each system's damping schedule and termination
rule, so they are no ranking.

A `-` means the number cannot be measured cleanly:

- full-iter, full-norm and 1st-iter are `-` for a row whose first
  iteration was not one accepted step: it holds wasted factorizations.
- full-iter and full-norm are `-` for the conjugate gradient rows (marked
  `*` in the READMEs): their work per iteration changes as the solve converges, so one
  iteration does not stand for the others. They are read on total ms and
  ms/iter.

t(1) and t(2) are each the minimum over the rounds, so 1st-iter minus
full-iter is the setup plus the difference between the first and the second
iteration, not the setup alone.

A row is at the common optimum when its cost is within 1% of the best and its
solution within the benchmark's distance of the best; the line under the
table counts them.
