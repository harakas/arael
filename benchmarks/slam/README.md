# Heterogeneous monocular SLAM benchmark

Batch optimization of a synthetic monocular SLAM problem,
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

A robot drives an S-curve (60/120/300 poses); 3D landmarks are observed
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
observation count grows faster than linearly (5.4k / 12.2k / 41.4k at
60 / 120 / 300 poses).

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
- **Single core**, verified: threading env vars exported to 1 before the
  process starts -- openblas-pthread sizes its pool at library load,
  before `main()` can set the vars, and those early threads escape the
  main-thread CPU pin (Ceres's SPARSE_NORMAL_CHOLESKY BLAS runs at 757%
  CPU without the caps; its `num_threads=1` option does not cover its
  BLAS). The process is pinned to a fixed core and each subprocess's
  `Cpus_allowed_list` is asserted equal to that core.

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
The arael models mark `eliminate_first(landmarks)`, so the faer rows
factorize with the landmarks ordered first instead of AMD.
Ceres runs three linear solvers: the two exact factorizations
(`sparse_normal_cholesky`, `sparse_schur`) are the like-for-like per-step
rows; `iterative_schur` is matrix-free preconditioned CG -- cheaper per
step but inexact, and at near-Gauss-Newton damping it does not reach the
validation gate here. g2o factorizes with CHOLMOD (GPL); arael ships the
permissive pure-Rust faer, and a `--features cholmod-gpl` build adds an
`arael LM f64 cholmod-gpl` row (CHOLMOD supernodal -- see the sparse-backend
comparison below).

## Results (Apple M4 Pro, single core, min of rounds)

### 60 poses (240 landmarks, 5,370 observations, 1,080 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     13.7 |   3(3) |    4.57 |         5.8 |    14.3 |  3062.0482 |
| arael LM f64 cholmod-gpl      |     12.7 |   3(3) |    4.22 |         5.9 |    17.8 |  3062.0482 |
| arael LM f32                  |     12.5 |   3(3) |    4.18 |         5.0 |    11.0 |  3062.1407 |
| tiny-solver LM               |    101.9 |      3 |   33.97 |        40.0 |    27.0 |  3062.0482 |
| factrs LM                     |     37.0 |      3 |   12.33 |        16.7 |    24.2 |  3062.0482 |
| ceres sparse_normal_cholesky  |     18.5 |   3(3) |    6.15 |        10.6 |    16.9 |  3062.0482 |
| ceres sparse_schur            |     19.0 |   3(3) |    6.32 |        11.3 |    16.4 |  3062.0482 |
| ceres iterative_schur         |     25.4 |   6(6) |    4.23 |         6.3 |    12.8 |  3067.3849 (RMSE 0.745 m) |
| symforce LM f64               |     27.6 |   3(4) |    6.90 |        18.2 |    27.1 |  3062.0482 |
| symforce LM f32               |     32.3 |   4(5) |    6.47 |        17.2 |    23.3 |  3062.0500 |
| g2o LM                        |     14.4 |   3(3) |    4.81 |         7.4 |    16.4 |  3062.0482 |
| gtsam LM                      |     32.0 |   3(3) |   10.67 |        15.6 |    47.3 |  3062.0482 |

11/12 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 120 poses (480 landmarks, 12,229 observations, 2,160 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     36.3 |   3(3) |   12.11 |        14.7 |    26.9 |  7065.8806 |
| arael LM f64 cholmod-gpl      |     34.5 |   3(3) |   11.51 |        15.9 |    30.1 |  7065.8806 |
| arael LM f32                  |     31.9 |   3(3) |   10.65 |        12.9 |    19.3 |  7065.8816 |
| tiny-solver LM               |    242.5 |      3 |   80.84 |        95.3 |    52.7 |  7065.8806 |
| factrs LM                     |     92.9 |      3 |   30.96 |        41.0 |    48.9 |  7065.8806 |
| ceres sparse_normal_cholesky  |     47.8 |   3(3) |   15.92 |        29.1 |    28.5 |  7065.8809 |
| ceres sparse_schur            |     54.3 |   3(3) |   18.11 |        31.3 |    27.3 |  7065.8809 |
| ceres iterative_schur         |    236.8 | 15(15) |   15.79 |        15.1 |    18.3 |  7066.3200 (RMSE 0.150 m) |
| symforce LM f64               |     84.6 |   3(4) |   21.15 |        51.9 |    52.7 |  7065.8806 |
| symforce LM f32               |    119.9 |   5(6) |   19.98 |        48.2 |    44.5 |  7065.8881 |
| g2o LM                        |     38.5 |   3(3) |   12.83 |        19.5 |    30.2 |  7065.8806 |
| gtsam LM                      |     87.5 |   3(3) |   29.16 |        41.2 |   107.2 |  7065.8806 |

11/12 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 300 poses (1,200 landmarks, 41,433 observations, 5,400 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |    225.6 |   3(3) |   75.19 |        83.9 |    87.6 | 24243.9094 |
| arael LM f64 cholmod-gpl      |    269.8 |   3(3) |   89.92 |       105.5 |   102.4 | 24243.9094 |
| arael LM f32                  |    183.5 |   3(3) |   61.15 |        70.7 |    64.8 | 24243.9107 |
| tiny-solver LM               |    990.6 |      3 |  330.21 |       377.7 |   172.0 | 24243.9094 |
| factrs LM                     |    482.8 |      3 |  160.94 |       195.1 |   186.2 | 24243.9094 |
| ceres sparse_normal_cholesky  |    312.5 |   3(3) |  104.17 |       141.1 |   101.8 | 24243.9095 |
| ceres sparse_schur            |    316.9 |   3(3) |  105.65 |       168.1 |    87.2 | 24243.9095 |
| ceres iterative_schur         |    515.7 |   9(9) |   57.30 |        56.3 |    41.1 | 24244.9863 (RMSE 0.137 m) |
| symforce LM f64               |    495.3 |   3(4) |  123.83 |       249.2 |   177.1 | 24243.9094 |
| symforce LM f32               |    854.7 |   6(7) |  122.10 |       235.7 |   136.1 | 24244.0803 (RMSE 0.060 m) |
| g2o LM                        |    240.2 |   3(3) |   80.07 |       114.8 |   114.4 | 24243.9094 |
| gtsam LM                      |    521.3 |   3(3) |  173.75 |       237.4 |   607.7 | 24243.9094 |

10/12 at the common optimum; ceres iterative_schur (inexact CG) and
symforce f32 do not reach the gate.

Every exact solver converges in 3 steps, so the tables compare per-step
cost (ms/iter) directly. arael f64 leads the f64 field at every size --
the landmarks-first elimination ordering keeps the faer factor small
where AMD's pose/landmark interleaving fills in (see the sparse-backend
comparison below), and it beats g2o's Schur-reduced supernodal CHOLMOD
75.2 vs 80.1 ms/iter at 300 poses. The cholmod-gpl row (CHOLMOD's own
AMD, no hint) no longer pays off at scale. GTSAM reaches the same
optimum but is the slowest exact solver per-step and, by a wide margin,
the heaviest on memory (608 MB at 300 poses vs 88 for arael).

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
goes via `LmConfig::gather_timing`; the three run interleaved, median of 1000
rounds.

All three reach the same optimum in the same 3 steps. The breakdown (Apple
M4 Pro, 60 poses / 240 landmarks, per-iteration mean with the first iteration
excluded):

| parameterization        | assembly | linear solve | cost eval | advance | total ms | vs simple |
|-------------------------|---------:|-------------:|----------:|--------:|---------:|----------:|
| `SimpleEulerAngleParam` |    0.395 |         3.53 |      0.13 |   0.000 |    15.56 |        -- |
| `EulerAngleParam`       |    0.399 |         3.53 |      0.14 |   0.001 |    15.60 |     +0.3% |
| `QuaternionParam`       |    0.408 |         3.53 |      0.14 |   0.001 |    15.61 |     +0.3% |

(ms/iter, except the last two columns: whole median solve, and total slowdown
vs simple.) The linear solve -- factorize + back-substitute the damped normal
equations -- is ~3.53 ms/iter for all three: they produce identical sparsity,
and it dominates the solve. The parameterization shows up only in **assembly**
(residual + Jacobian + Hessian). Each pose computes its rotation matrix and its
rotation Jacobian (dR/dparam) once, when its parameters update, and the
per-bearing assembly reads both rather than rebuilding them; with ~90 bearings
per pose that spreads the parameterization-specific cost -- the naive variant's
cached-sincos polynomial, `EulerAngleParam`'s reference composition,
`QuaternionParam`'s rational rotation-vector Jacobian -- across all of a pose's
observations. Assembly is then only **+1% (euler) and +3% (quaternion)** slower
per iteration than simple, and the shared linear solve shrinks that to **+0.3%**
on total solve time, at the run-to-run noise floor. Re-centering (`advance`) is
free for the naive variant (nothing to re-center) and a rounding sliver for the
other two.

The first iteration is far heavier and identical across all three: it
establishes the structure -- the first assembly runs ~2.2 ms (one-time
sparsity-pattern discovery over the ~0.4 ms steady state) and the first solve
runs ~5.1 ms (the one-time symbolic factorization over the ~3.5 ms steady
numeric solve) -- neither of which a warm re-solve pays.

**Net: the three parameterizations sit within ~3% per iteration on assembly and
~0.3% on total solve time.** The per-pose rotation-Jacobian precompute keeps the
exact SO(3) parameterizations (euler delta, quaternion) essentially as cheap per
step as the naive angles, so per-iteration cost is not a reason to prefer one
over another. Because the linear solve scales worse than assembly, at
larger scenes it dominates total time even more completely and the small
assembly gap disappears into it. Choose the parameterization for its geometry
(gimbal behaviour, large-step isotropy), not for these milliseconds. The flip
side -- how that geometry drives the *iteration count* when rotations are large
and the initialization is bad -- is measured in
[benchmarks/aerobatics](../aerobatics/README.md), where the ranking inverts
(quaternion fewest iterations, naive euler the most).

Run it (a separate binary from the cross-solver `cargo run`):

```sh
ROUNDS=1000 cargo run --release --bin rot_compare           # 60 poses, median of 1000 rounds (table above)
POSES=300 ROUNDS=10 cargo run --release --bin rot_compare   # larger scene, fewer rounds
```

## arael sparse-backend comparison

The default arael rows use faer with the model's `eliminate_first
(landmarks)` ordering. Building with `--features cholmod-gpl` adds an
`arael LM f64 cholmod-gpl` row (CHOLMOD supernodal, its own AMD -- the
hint cannot be injected through the Eigen wrapper) next to the faer row.
`SLAM_ARAEL_SOLVER` additionally swaps the default row's f64 backend
(each needs its cargo feature). Measured at 300 poses (5,400 parameters,
770k Hessian nonzeros), same optimum and iteration count on every
backend:

| backend (`SLAM_ARAEL_SOLVER=`)        | feature        | ms/iter |
|---------------------------------------|----------------|--------:|
| faer (default, landmarks-first)       | --             |     ~75 |
| CHOLMOD supernodal (`cholmod_gpl`)    | `cholmod-gpl`  |     ~90 |
| Eigen SimplicialLLT (`eigen`)         | `eigen`        |    ~411 |
| CHOLMOD simplicial (`cholmod`)        | `cholmod`      |    ~420 |

The elimination ordering is most of faer's lead: with plain AMD the same
faer solve measures ~101 ms/iter -- AMD interleaves poses and landmarks
and its factor carries 3.05M nonzeros against landmarks-first's 2.25M.

**License warning:** CHOLMOD's Supernodal module is GPL -- building with
`cholmod-gpl` makes the binary subject to the GPL (the `cholmod` feature
binds only the LGPL Simplicial module). That is why it is a separate,
explicit opt-in and not the default.

A `cholmod-gpl` build links CHOLMOD's dependency stack (OpenBLAS, LAPACK,
gfortran, gomp, AMD) into the benchmark binary, inflating every Rust row's
VmHWM by a few MB of shared-library baseline that a faer-only deployment
would not carry. The tables therefore report the non-gpl Rust rows' memory
from a default build -- in a cholmod-gpl run, point `SLAM_MEM_EXE` at a
default-build binary to source them cleanly. The cholmod-gpl row always
self-measures: its linked stack is part of that backend's real cost.

`G2O_STATS=1` prints g2o's own per-iteration timing breakdown (assembly,
Schur complement, factorization) for comparison.

The SLAM panel of the bar chart embedded in the top-level README
(`../chart-slam-loc-light.svg` / `-dark.svg`) is generated by
`../make_slam_loc_chart.py` (stdlib only) from the 300-pose ms/iter
column above; after re-running the benchmark, update its `PANELS`
table from the results and re-run it.

## Running

```sh
cmake -B cpp/build cpp && cmake --build cpp/build   # Ceres, g2o, GTSAM
# g2o needs libg2o-dev + cholmod; GTSAM needs libgtsam-dev (plus its
#   undeclared deps libboost-all-dev and libtbb-dev)
# add -DSYMFORCE_DIR=/path/to/symforce for the SymForce runner
# (regenerate its factor headers with:
#  symforce-venv/bin/python3 symforce_gen.py cpp/symforce_gen)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1     # before load; see Methodology
ROUNDS=3 cargo run --release                        # 60 poses (default)
SLAM_POSES=300 ROUNDS=3 cargo run --release
SLAM_SKIP_TINY=1 cargo run --release                # skip tiny-solver
SLAM_VERBOSE=1 cargo run --release                  # arael per-iteration trace
G2O_VERIFY_JAC=1 cpp/build/g2o_slam scene.txt lm out.txt     # check g2o Jacobians
GTSAM_VERIFY_JAC=1 cpp/build/gtsam_slam scene.txt lm out.txt # check GTSAM Jacobians
```

Env knobs: `SLAM_POSES`, `ROUNDS`, `SLAM_SKIP_TINY`, `SLAM_NO_MEM`, `SLAM_TIMING`,
`SLAM_ARAEL_SOLVER=eigen|cholmod|cholmod_gpl`, `SLAM_HESSIAN_BITMAP=<png>`,
`SLAM_MEM_EXE=<default-build binary for clean Rust-row memory>`,
`SLAM_DRIVER=nielsen`, `SLAM_LAMBDA0`, `CERES_SOLVERS=<comma list>`,
`CERES_RADIUS0`, `TINY_MAXITER`, `TINY_RADIUS0`, `SYMFORCE_LAMBDA0`,
`SYMFORCE_MAXITER`, `G2O_LAMBDA_INIT`, `G2O_GAIN`, `G2O_VERIFY_JAC`,
`GTSAM_LAMBDA0`, `GTSAM_VERIFY_JAC`.
