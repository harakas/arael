# Localization benchmark

A trajectory estimated against a **known, fixed** landmark map from bearing
observations, wheel/visual odometry, and pose priors -- the localization
problem, as distinct from SLAM. It is ported from `examples/loc_demo.rs` and
shares the methodology of [`benchmarks/slam`](../slam) and
[`benchmarks/pgo`](../pgo): one scene generator, one reference cost function
every system is validated against, a cross-checked initial cost, single-core
pinning, and min-of-N interleaved timing.

## Why localization is its own benchmark

In SLAM the landmarks are optimized, so the Hessian fills in (each landmark
couples every pose that sees it) -- a bundle-adjustment structure. In
**localization the landmarks are constants**: the bearing factors hit fixed
points, so they contribute no pose-to-pose coupling, and the only thing that
couples poses is the **consecutive odometry**. The Hessian is therefore
**block-tridiagonal** -- a band. With 6-DOF poses laid out consecutively the
half-bandwidth is `kd = 2*6 - 1 = 11`, and arael solves it with its band
Cholesky (`solve_band`), matching `loc_demo`. A fixed map also removes the
gauge freedom, so absolute pose errors are meaningful and no GPS is needed.

The factor graph is deliberately heterogeneous -- bearings (atan2), euler
odometry (rotation composition + euler extraction), and pose drift + tilt
priors -- which stresses per-factor code generation, not just a pose chain.

## The systems

Every system optimizes the identical problem (the exported scene) and is
scored by the one `scene::reference_cost`. The bearing residuals are plain
Gaussian: this is the outlier-free scenario, so `loc_demo`'s robust
`gamma*atan` kernel is omitted (with no outliers it has no benefit, and its
saturation manufactures spurious minima different solvers fall into).

- **arael** (f64 + f32) -- band Cholesky (`kd=11`); `LOC_ARAEL_SOLVER=faer`
  switches to the general sparse solver for comparison. Both roots use
  `#[arael(root, fast_atan)]`: the bearing residuals go through the fast
  polynomial atan2 (max error < 1e-6 rad), which is why the shared
  initial-cost cross-check tolerance is 1e-5 (arael measures ~4e-7
  relative, the exact systems ~1e-16). The validation gates (cost
  within 1%, pose RMSE < 5 cm) are orders of magnitude above the
  approximation.
- **tiny-solver**, **factrs** -- Rust, dual-number autodiff, their own LM.
- **Ceres** (3 linear solvers), **g2o**, **GTSAM**, **SymForce** (f64 + f32) --
  C++: analytic Jacobians (g2o/GTSAM), autodiff (Ceres), or code-generated
  linearization (SymForce, from `symforce_gen.py`). Each is cross-checked to
  the reference cost; g2o/GTSAM are additionally Jacobian-verified against
  finite differences (`G2O_VERIFY_JAC=1`, `GTSAM_VERIFY_JAC=1`).

The bearing factor is **single-variable** (the fixed landmark carries no
derivatives): a remote block on the pose in arael, a unary edge/factor in
g2o/GTSAM.

## Results

Apple M4 Pro, single core, min of N interleaved rounds; each run's
`config:` banner records ROUNDS, solver, and damping. All systems converge
in 3 steps to the same optimum; **`full-it ms` -- the cost of one full
steady-state iteration (defined under Running) -- is the headline metric**,
with the one-time first iteration shown apart. `LOC_POSES=N` sets the pose
count (landmarks scale as `4N`). The arael rows use `fast_atan` (above).

### 60 poses (240 landmarks, 5,357 frines, 360 parameters; ROUNDS=50, LOC_LAMBDA0=1e-5)

| system                       | total ms | iters | ms/iter | full-it ms | 1st-iter ms | peak MB | final cost |
|------------------------------|---------:|------:|--------:|-----------:|------------:|--------:|-----------:|
| **arael LM f64 (band)**      |      0.9 |  3(3) | **0.31** |   **0.31** |         0.4 |     4.7 |  3274.6025 |
| **arael LM f32 (band)**      |      0.9 |  3(3) | **0.31** |   **0.31** |         0.3 |     4.1 |  3274.6025 |
| symforce LM f32              |      4.6 |  3(4) |    1.15 |          - |         4.8 |    16.9 |  3274.6025 |
| symforce LM f64              |      5.2 |  3(4) |    1.29 |          - |         5.3 |    19.2 |  3274.6025 |
| ceres sparse_normal_cholesky |      4.6 |  3(3) |    1.52 |       0.80 |         3.0 |    13.0 |  3274.6025 |
| g2o LM                       |      4.6 |  3(3) |    1.52 |       1.25 |         2.1 |    11.3 |  3274.6025 |
| ceres sparse_schur           |      4.7 |  3(3) |    1.58 |       0.85 |         3.0 |    12.9 |  3274.6025 |
| ceres iterative_schur        |      4.9 |  3(3) |    1.64 |       1.05 |         2.8 |    12.1 |  3274.6025 |
| gtsam LM                     |     10.5 |  3(3) |    3.51 |       3.15 |         4.2 |    14.6 |  3274.6025 |
| factrs LM                    |     14.3 |     3 |    4.75 |          - |         6.2 |    11.4 |  3274.6025 |
| tiny-solver LM               |     69.2 |     3 |   23.07 |          - |        25.3 |    12.5 |  3274.6025 |

### 300 poses (1,200 landmarks, 41,323 frines, 1,800 parameters; ROUNDS=5, default damping)

| system                       | total ms | iters | ms/iter | full-it ms | 1st-iter ms | peak MB | final cost |
|------------------------------|---------:|------:|--------:|-----------:|------------:|--------:|------------:|
| **arael LM f64 (band)**      |      7.9 |  3(3) | **2.64** |   **2.69** |         2.7 |    13.4 | 25269.0409 |
| **arael LM f32 (band)**      |      7.7 |  2(3) | **2.55** |   **2.56** |         2.5 |     9.7 | 25269.0410 |
| symforce LM f32              |     41.6 |  2(4) |   10.41 |          - |        41.5 |    90.1 | 25269.0410 |
| symforce LM f64              |     45.5 |  3(4) |   11.36 |          - |        44.9 |   104.3 | 25269.0409 |
| ceres sparse_normal_cholesky |     36.4 |  3(3) |   12.12 |       5.90 |        24.6 |    42.0 | 25269.0409 |
| ceres sparse_schur           |     37.1 |  3(3) |   12.38 |       6.85 |        23.4 |    37.2 | 25269.0409 |
| ceres iterative_schur        |     38.9 |  3(3) |   12.97 |       8.30 |        22.3 |    36.6 | 25269.0409 |
| g2o LM                       |     38.9 |  3(3) |   12.98 |      10.20 |        18.5 |    37.8 | 25269.0409 |
| gtsam LM                     |     89.9 |  3(3) |   29.98 |      28.05 |        33.8 |    58.8 | 25269.0409 |
| factrs LM                    |    104.8 |     3 |   34.94 |          - |        47.0 |    64.4 | 25269.0409 |
| tiny-solver LM               |    550.9 |     3 |  183.62 |          - |       205.1 |    68.3 | 25269.0409 |

### Raspberry Pi 5 (Cortex-A76, single core, 60 poses)

The full field on a Raspberry Pi 5. The ordering is unchanged -- arael leads
~4x, everything is ~3x slower than the dev machine as expected for the smaller
core, and all 11 rows still reach the common optimum. (The Pi tables predate
the arael rows' fast_atan adoption and the total/iters/full-it columns.)

| system                       | ms/iter | 1st-iter ms | peak MB | final cost |
|------------------------------|--------:|------------:|--------:|-----------:|
| **arael LM f64 (band)**      |  **1.28** |       1.3 |     4.5 |  3274.6025 |
| **arael LM f32 (band)**      |  **1.35** |       1.4 |     3.9 |  3274.6025 |
| symforce LM f32              |    4.99 |        17.7 |    16.9 |  3274.6025 |
| ceres sparse_normal_cholesky |    5.16 |        10.2 |    11.1 |  3274.6025 |
| ceres iterative_schur        |    5.57 |         9.8 |    10.6 |  3274.6025 |
| g2o LM                       |    5.58 |         8.4 |    10.0 |  3274.6025 |
| ceres sparse_schur           |    5.65 |        10.6 |    11.0 |  3274.6025 |
| symforce LM f64              |    5.73 |        20.1 |    19.2 |  3274.6025 |
| gtsam LM                     |   15.61 |        16.5 |    15.4 |  3274.6025 |
| factrs LM                    |   18.41 |        23.8 |    11.5 |  3274.6025 |
| tiny-solver LM               |   89.13 |        97.2 |    11.4 |  3274.6025 |

### Raspberry Pi Zero (ARM1176, ARMv6, single core, 100 poses)

The cross-compiled static-musl binary on a real Pi Zero, arael + factrs only --
the C++ solvers are not cross-built, and tiny-solver is omitted as too slow. arael dominates by ~18x, and the initial cost still matches
the reference (5e-15), confirming the ARMv6 build is bit-faithful. Note **f32 is
markedly faster than f64 here** -- the ARM1176 VFPv2 FPU favors single precision
-- the reverse of the Cortex-A76 above.

| system              | ms/iter | 1st-iter ms | peak MB | final cost |
|---------------------|--------:|------------:|--------:|-----------:|
| **arael LM f32 (band)** |  **47.74** |      48.4 |     2.9 |  6050.6032 |
| **arael LM f64 (band)** |  **65.03** |      65.9 |     3.9 |  6050.6032 |
| factrs LM               |  873.33 |     1069.5 |    11.5 |  6050.6032 |

## Analysis

arael leads by **~4x per attempt** (ms/iter) at both sizes -- 0.31 vs
the fastest competitor's 1.15 (SymForce f32) and ~1.5 (Ceres/g2o) at 60
poses; 2.5/2.6 vs 10-13 at 300 -- and its memory is a fraction of the field
at scale (10-13 MB vs 37-104 MB at 300 poses; SymForce, the per-attempt
runner-up, is the heaviest).

The `full-it ms` column splits that lead in two. On the pure steady-state
iteration arael's closest exact competitor is Ceres sparse_normal_cholesky
at **~2.2-2.6x** (0.80 vs 0.31 at 60 poses; 5.90 vs 2.69 at 300). The rest
of the headline gap is the field's much heavier FIRST iteration -- symbolic
factorization and structure setup that arael largely avoids (1st-iter 24.6
ms for Ceres vs 2.7 for arael at 300 poses) -- amortized over a solve that
only takes 3 iterations.

**The steady-state lead is not the band solver.** Running the same problem
with `LOC_ARAEL_SOLVER=faer` (arael's *general* sparse Cholesky -- the same
approach the others take) measured only ~7% slower than band at 300 poses
and identical at 60 before the fast_atan adoption; with the bearing
residuals now cheaper the solver's share grows, and the pair needs a
re-measure. The advantage is arael's **assembly**: it code-generates the
residual + Jacobian + Hessian block at compile time (flat, CSE'd,
accumulated directly), where Ceres/factrs/tiny pay a runtime autodiff tax
and g2o/GTSAM pay their general-purpose graph frameworks' per-edge dispatch
and sparse-assembly overhead. SymForce, which also code-generates, is the
closest per-attempt -- confirming the assembly is most of the gap; the rest
is solver/framework overhead.

**What the band solver buys is memory and scaling, not the headline:** on
this block-tridiagonal structure it holds ~10% less peak memory than
arael's general sparse path at 300 poses (13.4 vs 14.6 MB), and band
Cholesky is O(n * kd^2) with contiguous storage, while a general sparse
factorization carries a fill-reducing ordering and supernodal machinery the
band does not need. It is the right tool for localization, and on a long
real trajectory (thousands of poses) the gap widens; at these sizes the
code-gen assembly dominates.

f32 matches f64 to the final cost here (the problem is well-conditioned) at
lower memory -- the band factor and its Cholesky halve in width.

The localization panel of the bar chart embedded in the top-level
README (`../chart-slam-loc-light.svg` / `-dark.svg`) is generated by
`../make_slam_loc_chart.py` (stdlib only) from the Raspberry Pi 5
ms/iter column above; after re-running the benchmark, update its
`PANELS` table from the results and re-run it.

## Running

```sh
ROUNDS=10 cargo run --release                    # 60 poses (default)
LOC_POSES=300 ROUNDS=5 cargo run --release       # larger tier
LOC_ARAEL_SOLVER=faer cargo run --release        # arael on general sparse
LOC_NO_MEM=1 cargo run --release                 # skip peak-memory subprocess
LOC_TIMING=1 ROUNDS=100 cargo run --release      # arael f64-vs-f32 per-phase
                                                 # steady-state timing table
```

The `full-it ms` column is the cost of one FULL steady-state iteration,
free of the two distortions in `ms/iter` = total/attempts: rejected
damping attempts (which skip the re-linearization) deflate it, and the
first iteration's one-time costs (sparsity discovery, symbolic
factorization) inflate it. arael reports the sum of the solver's
steady-state per-phase means (`gather_timing`; first calls excluded).
External systems report `(total - 1st-iter) / (attempts - 1)` computed
from the min-of-rounds totals -- exact when the run is rejection-free,
since every steady iteration is then a full one -- and `-` when the run
had rejected attempts or no attempt accounting (tiny-solver, factrs). g2o's `iters` reads
accepted(attempts) with the attempts from an untimed statistics pass,
never gathered in the timed solve.

The C++ runners execute as subprocesses over an exported copy of the scene;
they are skipped with a warning if their binary is absent.

```sh
cmake -B cpp/build cpp && cmake --build cpp/build   # Ceres, g2o, GTSAM
# g2o needs libg2o-dev + cholmod; GTSAM needs libgtsam-dev (+ libtbb-dev)
```

Verification (never in the timed solve): `G2O_VERIFY_JAC=1` and
`GTSAM_VERIFY_JAC=1` check each analytic Jacobian against finite differences.

### Cross-compiling for Raspberry Pi

`.cargo/config.toml` defines two dependency-free static-musl targets, linked by
the bundled `rust-lld` -- no external cross toolchain. The C++ solvers are not
built; on the device the harness detects they are missing and runs the Rust
solvers only (arael f64/f32, tiny-solver, factrs).

```sh
# Pi Zero / Zero W / Pi 1 (BCM2835, ARM1176 = ARMv6)
rustup target add arm-unknown-linux-musleabihf
cargo build --release --target arm-unknown-linux-musleabihf

# Pi 5 / any aarch64 board
rustup target add aarch64-unknown-linux-musl
cargo build --release --target aarch64-unknown-linux-musl
```

Copy `target/<triple>/release/loc-bench` to the device and run it there.

### SymForce

The generated factor headers are committed under `cpp/symforce_gen/`, so the
runner builds directly against a SymForce source checkout:

```sh
cmake -B cpp/build cpp -DSYMFORCE_DIR=<checkout> && cmake --build cpp/build
```

To regenerate the headers after changing a residual, run `symforce_gen.py` with
the SymForce Python toolchain on its path -- the sympy backend suffices, no
symengine build needed:

```sh
SYMFORCE_SYMBOLIC_API=sympy PYTHONPATH=<checkout>:<checkout>/gen/python \
    python symforce_gen.py cpp/symforce_gen
```
