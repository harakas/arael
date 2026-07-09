# Heterogeneous visual-inertial SLAM benchmark

Batch optimization of a synthetic visual-inertial SLAM problem,
comparing arael against six other solvers:
[tiny-solver](https://crates.io/crates/tiny-solver) (Rust, dual-number
autodiff), [factrs](https://crates.io/crates/factrs) (Rust, dual-number
ForwardProp autodiff), [Ceres](http://ceres-solver.org) (C++ 2.2,
template autodiff; three linear-solver configurations),
[SymForce](https://symforce.org) (Skydio's symbolic code-generation
path; its own `sym::Optimizer`, templated over f64/f32),
[g2o](https://github.com/RainerKuemmerle/g2o) (C++; custom vertices and
six custom edges with hand-derived analytic Jacobians, CHOLMOD, landmarks
marginalized via Schur), and [GTSAM](https://gtsam.org) (C++ 4.2; six
custom `NoiseModelFactorN` factors with hand-derived analytic Jacobians,
multifrontal Cholesky). Unlike the pose-graph
([benchmarks/pgo](../pgo)) and bundle-adjustment ([benchmarks/bal](../bal))
benchmarks -- each a single factor type -- this problem is
heterogeneous: six factor types, several nonlinear.

## Problem

A robot drives an S-curve (60/125/250 poses); 3D landmarks are observed
by 5 cameras giving 360-degree coverage. The factor graph:

| factor | arity | residuals | content |
|--------|-------|----------:|---------|
| GPS | pose | 3 | position vs a biased GPS reading, whitened |
| pose drift | pose | 6 | soft prior to the initialization |
| tilt | pose | 2 | roll/pitch from an accelerometer |
| landmark drift | landmark | 3 | soft prior to the initialization |
| bearing | landmark + pose | 2 | `atan2` bearing residual in the feature frame |
| odometry | pose + pose | 6 | full 6-DOF relative motion (rotation composition + euler extraction) |

Poses are `(x, y, z, roll, pitch, yaw)` -- 6 raw parameters, rotation
built from the euler angles inside each residual (arael's
`SimpleEulerAngleParam`; matched by plain 6-vectors in the other four
systems). Landmarks are 3 parameters. The scene is generated from a
fixed seed; `SLAM_POSES=N` sets the pose count (landmarks scale as `4N`).
"Wide" landmarks (15%) are visible across up to `N/4` poses, so the
observation count grows faster than linearly (5.4k / 12.8k / 31.8k at
60 / 125 / 250 poses).

This is the **outlier-free** scenario: the bearing and GPS residuals are
plain Gaussian (max-likelihood when there are no outliers). The
`slam_demo` example this is ported from adds wrong feature associations
and suppresses them with a robust `gamma*atan(r/gamma)` kernel plus
graduated optimization; that machinery is omitted here because it only
earns its place with outliers present -- and the kernel's saturation
otherwise manufactures a spurious landmark-depth minimum that different
solvers fall into, which would make "final cost" a basin lottery rather
than a clean comparison. The bearing factors carry an information scale
(`frine_isigma_scale`, fixed at 1.0) that is the hook for reintroducing
graduation with outliers. The drift regularizers pull toward an explicit
stored prior (the init), so every system computes the same residual.

The problem is optimized as a single least-squares solve from a good
initialization; all five systems reach one unambiguous optimum.

## Methodology

Same as the pgo/bal benchmarks:

- **One cost function.** `src/scene.rs::reference_cost` evaluates every
  system's final parameters. Each system's own residual code is
  cross-checked against it at the initial estimate to 1e-9 relative --
  all seven implementations (arael, tiny-solver, factrs, Ceres, SymForce,
  g2o, GTSAM) and the reference agree bit-for-bit.
- **One validation.** A row is "at the common optimum" when its cost is
  within 1% of the best AND its per-pose translation RMSE to the best
  solution is under 5 cm (the gauge is fixed by GPS + priors, so
  absolute positions are comparable without alignment).
- **Full performance, no shortcuts.** Every system runs at parity: its
  own autodiff/codegen/analytic Jacobians, no numeric differentiation
  anywhere. g2o's six custom edges carry hand-derived analytic
  `linearizeOplus` Jacobians, checked against finite differences
  (`G2O_VERIFY_JAC=1`; verification only, never in the timed solve).
- **Timing.** Total time is the min over N interleaved rounds; per-step
  (ms/iter) is total / total iterations; first-iteration time is a fresh
  solve capped at one iteration. arael, Ceres, and SymForce report
  `accepted(total)` steps.
- **Peak memory** (peak MB) is the process high-water mark (`VmHWM`),
  each solver measured in a fresh process: arael, tiny-solver, and
  factrs via a single-solver subprocess of this binary; Ceres, SymForce,
  and g2o as their own subprocesses.
- **Single core**, verified: threading env vars forced to 1 (Ceres's
  SPARSE_NORMAL_CHOLESKY BLAS runs at 757% CPU otherwise -- its
  `num_threads=1` option does not cover its BLAS), the process pinned to a
  fixed core, and each subprocess's `Cpus_allowed_list` asserted equal to
  that core.

**Problem-appropriate initial damping for every LM.** Damping-schedule
strategy is a per-problem tuning knob, not an intrinsic property of a
solver -- so each LM gets an initial damping suited to this
well-initialized graph rather than its shipped default (the same policy
as [benchmarks/pgo](../pgo)): arael `initial_lambda = 1e-8` with the
plain fixed schedule (no adaptive driver -- no step is rejected here, so
a gain-ratio driver would only over-damp and inflate the step count;
`SLAM_DRIVER=nielsen` opts into it, `SLAM_LAMBDA0` overrides; f32 uses
`1e-7` at the 60-pose size, where a hair more damping stops it cleanly at
the f32 precision floor instead of grinding), tiny-solver
and Ceres `initial_trust_region_radius = 1e12`, SymForce
`initial_lambda = 1e-10` (it ships 1.0), factrs its default (`lambda`
starts at 1e-10 -- already near-Gauss-Newton), g2o `setUserLambdaInit(1e-9)`
(`G2O_LAMBDA_INIT` overrides), GTSAM `lambdaInitial = 1e-9`
(`GTSAM_LAMBDA0` overrides). All stop at a shared termination class
(1e-5 absolute or relative). With this policy every exact-factorization
solver converges in 3 steps, so **ms/iter -- the per-step pipeline cost
(linearize + assemble + factorize + solve) -- is the durable cross-system
comparison**; total time and iteration count reflect the damping
strategy, which is tuning, not implementation.

arael runs f64 and f32; its residuals and SymForce's are code-generated
(SymForce from `symforce_gen.py`, flattened C++ under `cpp/symforce_gen/`).
Ceres runs three linear solvers: the two exact factorizations
(`sparse_normal_cholesky`, `sparse_schur`) are the like-for-like per-step
rows; `iterative_schur` is matrix-free preconditioned CG -- cheaper per
step but inexact, and at near-Gauss-Newton damping it does not reach the
validation gate here. g2o factorizes with CHOLMOD (GPL); arael ships the
permissive pure-Rust faer.

## Results (Apple M4 Pro, single core, min of rounds)

### 60 poses (240 landmarks, 5,370 observations, 1,080 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     15.0 |   3(3) |    5.01 |         7.1 |    12.6 |  3062.0482 |
| arael LM f32                  |     13.6 |   3(3) |    4.53 |         6.2 |    10.1 |  3062.0482 |
| tiny-solver LM               |    102.4 |      3 |   34.13 |        40.4 |    26.1 |  3062.0482 |
| factrs LM                     |     40.0 |      3 |   13.32 |        17.7 |    23.3 |  3062.0482 |
| ceres sparse_normal_cholesky  |     18.4 |   3(3) |    6.13 |        10.5 |    17.0 |  3062.0482 |
| ceres sparse_schur            |     19.3 |   3(3) |    6.42 |        11.5 |    16.5 |  3062.0482 |
| ceres iterative_schur         |     25.6 |   6(6) |    4.27 |         6.3 |    12.9 |  3067.3849 (RMSE 0.745 m) |
| symforce LM f64               |     28.1 |   3(4) |    7.03 |        18.5 |    27.1 |  3062.0482 |
| symforce LM f32               |     32.7 |   4(5) |    6.53 |        17.3 |    23.3 |  3062.0500 |
| g2o LM                        |     14.5 |   3(3) |    4.82 |         7.4 |    16.4 |  3062.0482 |
| gtsam LM                      |     32.3 |   3(3) |   10.77 |        15.9 |    47.4 |  3062.0482 |

10/11 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 125 poses (500 landmarks, 12,830 observations, 2,250 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     44.1 |   3(3) |   14.69 |        20.2 |    26.1 |  7424.1484 |
| arael LM f32                  |     39.6 |   3(3) |   13.19 |        17.6 |    20.3 |  7424.1506 |
| tiny-solver LM               |    269.3 |      3 |   89.77 |       104.8 |    55.1 |  7424.1484 |
| factrs LM                     |    101.9 |      3 |   33.95 |        43.2 |    50.9 |  7424.1484 |
| ceres sparse_normal_cholesky  |     49.9 |   3(3) |   16.65 |        30.1 |    28.4 |  7424.1485 |
| ceres sparse_schur            |     54.9 |   3(3) |   18.32 |        31.7 |    28.3 |  7424.1485 |
| ceres iterative_schur         |    585.7 | 30(30) |   19.52 |        15.4 |    18.8 |  7424.7966 (RMSE 0.179 m) |
| symforce LM f64               |     89.1 |   3(4) |   22.27 |        52.5 |    55.0 |  7424.1484 |
| symforce LM f32               |    124.1 |   5(6) |   20.69 |        49.2 |    46.3 |  7424.1592 |
| g2o LM                        |     40.0 |   3(3) |   13.33 |        20.1 |    32.1 |  7424.1484 |
| gtsam LM                      |     88.0 |   3(3) |   29.32 |        44.4 |   116.2 |  7424.1484 |

10/11 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 250 poses (1,000 landmarks, 31,823 observations, 4,500 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |    201.1 |   3(3) |   67.05 |        77.6 |    65.2 | 18581.8936 |
| arael LM f32                  |    157.2 |   2(3) |   52.40 |        64.6 |    46.7 | 18581.8975 |
| tiny-solver LM               |    708.9 |      3 |  236.29 |       279.0 |   152.1 | 18581.8936 |
| factrs LM                     |    341.2 |      3 |  113.75 |       136.2 |   139.5 | 18581.8936 |
| ceres sparse_normal_cholesky  |    192.7 |   3(3) |   64.24 |       109.0 |    65.1 | 18581.8937 |
| ceres sparse_schur            |    200.4 |   3(3) |   66.80 |       107.7 |    65.9 | 18581.8937 |
| ceres iterative_schur         |    414.8 | 10(10) |   41.48 |        40.0 |    33.8 | 18583.3216 (RMSE 0.185 m) |
| symforce LM f64               |    334.3 |   3(4) |   83.58 |       170.2 |   137.0 | 18581.8936 |
| symforce LM f32               |    572.3 |   6(7) |   81.75 |       164.2 |   108.0 | 18581.9541 |
| g2o LM                        |    155.2 |   3(3) |   51.75 |        74.6 |    84.8 | 18581.8936 |
| gtsam LM                      |    312.1 |   3(3) |  104.02 |       144.9 |   390.6 | 18581.8936 |

10/11 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

Every exact solver converges in 3 steps, so the tables compare per-step
cost (ms/iter) directly. g2o edges ahead at the largest size -- possibly
its supernodal CHOLMOD factorization (arael ships permissive faer),
though this was not verified by a backend swap. GTSAM reaches the same
optimum but is the slowest exact solver per-step and, by a wide margin,
the heaviest on memory (391 MB at 250 poses vs 65 for arael).

### SLAM on the edge: Raspberry Pi

**Raspberry Pi 5** -- Cortex-A76 @ 2.4 GHz, aarch64, Debian Bookworm --
runs the 60-pose problem -- 240 landmarks, 5,370 observations, 1,080
parameters -- via `SLAM_POSES=60 ROUNDS=10 cargo run --release`:

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     59.0 |   3(3) |   19.66 |        26.4 |    12.0 |  3062.0482 |
| arael LM f32                  |     49.5 |   3(3) |   16.50 |        22.9 |     9.6 |  3062.0482 |
| tiny-solver LM                |    385.2 |      3 |  128.39 |       148.5 |    25.0 |  3062.0482 |
| factrs LM                     |    144.9 |      3 |   48.30 |        61.8 |    27.5 |  3062.0482 |
| ceres sparse_normal_cholesky  |     92.5 |   3(3) |   30.83 |        41.6 |    15.5 |  3062.0482 |
| ceres sparse_schur            |     83.5 |   3(3) |   27.83 |        42.6 |    15.1 |  3062.0482 |
| ceres iterative_schur         |     84.1 |   6(6) |   14.02 |        21.3 |    11.3 |  3067.3849 (RMSE 0.745 m) |
| symforce LM f64               |     89.0 |   3(4) |   22.25 |        57.5 |    27.1 |  3062.0482 |
| symforce LM f32               |     98.3 |   4(5) |   19.66 |        50.5 |    23.3 |  3062.0500 |
| g2o LM                        |     63.6 |   3(3) |   21.19 |        28.6 |    15.2 |  3062.0482 |
| gtsam LM                      |    143.5 |   3(3) |   47.84 |        55.4 |    48.2 |  3062.0482 |

**Raspberry Pi Zero W** -- ARM1176JZF-S @ ~1 GHz, ARMv6, no NEON,
in-order, 512 MB -- is too small to compile natively, so it runs a
cross-compiled, dependency-free static musl binary (flags in
`.cargo/config.toml`). Only the Rust solvers run here -- the C++ stack
(Ceres/g2o/GTSAM) is not cross-compiled for ARMv6:

```sh
rustup target add arm-unknown-linux-musleabihf
cargo build --release --target arm-unknown-linux-musleabihf
```

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |   2217.6 |   3(3) |  739.21 |       858.5 |     8.9 |  3062.0484 |
| arael LM f32                  |   1660.2 |   3(3) |  553.40 |       665.5 |     6.5 |  3062.0484 |
| tiny-solver LM                |  11293.2 |      3 | 3764.38 |      4061.2 |    13.1 |  3062.0483 |
| factrs LM                     |   4601.8 |      3 | 1533.92 |      1737.1 |    14.3 |  3062.0483 |

Both reach the same optimum. The Pi 5's per-step slowdown vs the M4 Pro is
not uniform: ~3.6-3.9x for the Rust/faer solvers (arael, tiny, factrs),
~3.1x for SymForce, and 4.3-5.0x for the CHOLMOD-based C++ solvers (Ceres,
g2o, GTSAM). So arael's per-step lead over those widens on the Pi 5 --
g2o, a hair faster than arael f64 on the M4 Pro (4.82 vs 5.01), lands
behind it here (21.19 vs 19.66) -- while its gap to SymForce narrows.
60 poses in 50-385 ms; usable.

The Zero (Rust solvers only), on its in-order ARMv6 core, is ~100x
slower, and there arael's lead over tiny/factrs narrows (6.8x -> 5.1x
over tiny). It was measured with the static-musl binary (the Pi 5 is
native glibc); musl's simpler malloc inflates the allocation-heavy
tiny/factrs there by ~30-50%.

## Rotation parameterization: simple vs euler vs quaternion

A standalone binary, `rot_compare`, isolates the cost of arael's three
SO(3) parameterizations on this scene and solver: `SimpleEulerAngleParam`
(Euler angles optimized directly), `EulerAngleParam` (an Euler-angle delta
re-centered onto a matrix reference each step), and `QuaternionParam` (a
rotation-vector / exp-map delta re-centered onto a unit-quaternion
reference). Only the pose rotation parameter differs -- all three feed the
identical GPS / drift / tilt / bearing / odometry factors (the cross-solver
tables above use `SimpleEulerAngleParam`). Each solve reports where its time
goes via `LmConfig::gather_timing`; the three run interleaved, median of 30
rounds.

All three reach the same optimum in the same 3 steps. The breakdown (Apple
M4 Pro, 60 poses / 240 landmarks, per-iteration mean with the first iteration
excluded):

| parameterization        | assembly | linear solve | cost eval | advance | total ms | vs simple |
|-------------------------|---------:|-------------:|----------:|--------:|---------:|----------:|
| `SimpleEulerAngleParam` |    0.402 |         3.57 |      0.14 |   0.000 |     15.0 |        -- |
| `EulerAngleParam`       |    0.466 |         3.57 |      0.14 |   0.001 |     15.2 |       +2% |
| `QuaternionParam`       |    0.552 |         3.57 |      0.14 |   0.001 |     15.5 |       +3% |

(ms/iter, except the last two columns: whole median solve, and total slowdown
vs simple.) The linear solve -- factorize + back-substitute the damped normal
equations -- is ~3.57 ms/iter for all three: they produce identical sparsity,
and it dominates the solve. The parameterization shows up only in **assembly**
(residual + Jacobian + Hessian): the naive variant is cheapest -- its rotation
is a division-free polynomial in cached sincos values; `EulerAngleParam` adds
the reference composition; `QuaternionParam`'s rotation-vector delta is a
rational expression with the largest Jacobian. Assembly is **+16% (euler) and
+37% (quaternion)** slower per iteration than simple; because the shared linear
solve dominates, that narrows to roughly **+2% and +3%** on total solve time
(small enough to sit near the run-to-run noise, unlike the robust assembly
gap). Re-centering (`advance`) is free for the naive variant (nothing to
re-center) and a rounding sliver for the other two.

The first iteration is far heavier and identical across all three: it
establishes the structure -- ~1.5 ms to discover the sparsity pattern in the
first assembly, ~5.2 ms for the one-time symbolic factorization in the first
solve -- neither of which a warm re-solve pays.

**Net: `SimpleEulerAngleParam` is fastest and `QuaternionParam` slowest** --
up to +37% per iteration on the rotation assembly, but only +3% on total time,
because the shared linear solve dwarfs the assembly gap. The total gap is a
small-scene effect: the linear solve scales worse than assembly, so at 300
poses / 1,200 landmarks it reaches ~90 ms/iter and the assembly difference --
still ~+33% per iteration -- vanishes into under 1% of total (below run-to-run
noise). The durable statement is the per-iteration assembly cost; choose the
parameterization for its geometry (gimbal behaviour, large-step isotropy), not
for these milliseconds. The flip side -- how that geometry drives the
*iteration count* when rotations are large and the initialization is bad -- is
measured in [benchmarks/aerobatics](../aerobatics/README.md), where the ranking
inverts (quaternion fewest iterations, naive euler the most).

Run it (a separate binary from the cross-solver `cargo run`):

```sh
cargo run --release --bin rot_compare              # 60 poses (default), median of 30 rounds
POSES=300 ROUNDS=10 cargo run --release --bin rot_compare   # larger scene, fewer rounds
```

## Running

```sh
cmake -B cpp/build cpp && cmake --build cpp/build   # Ceres, g2o, GTSAM
# g2o needs libg2o-dev + cholmod; GTSAM needs libgtsam-dev (plus its
#   undeclared deps libboost-all-dev and libtbb-dev)
# add -DSYMFORCE_DIR=/path/to/symforce for the SymForce runner
# (regenerate its factor headers with:
#  symforce-venv/bin/python3 symforce_gen.py cpp/symforce_gen)
ROUNDS=3 cargo run --release                        # 60 poses (default)
SLAM_POSES=250 ROUNDS=3 cargo run --release
SLAM_SKIP_TINY=1 cargo run --release                # skip tiny-solver
SLAM_VERBOSE=1 cargo run --release                  # arael per-iteration trace
G2O_VERIFY_JAC=1 cpp/build/g2o_slam scene.txt lm out.txt     # check g2o Jacobians
GTSAM_VERIFY_JAC=1 cpp/build/gtsam_slam scene.txt lm out.txt # check GTSAM Jacobians
```

Env knobs: `SLAM_POSES`, `ROUNDS`, `SLAM_SKIP_TINY`, `SLAM_NO_MEM`,
`SLAM_DRIVER=nielsen`, `SLAM_LAMBDA0`, `CERES_SOLVERS=<comma list>`,
`CERES_RADIUS0`, `TINY_MAXITER`, `TINY_RADIUS0`, `SYMFORCE_LAMBDA0`,
`SYMFORCE_MAXITER`, `G2O_LAMBDA_INIT`, `G2O_GAIN`, `G2O_VERIFY_JAC`,
`GTSAM_LAMBDA0`, `GTSAM_VERIFY_JAC`.
