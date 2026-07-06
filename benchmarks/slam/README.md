# Heterogeneous visual-inertial SLAM benchmark

Batch optimization of a synthetic visual-inertial SLAM problem,
comparing arael against four other solvers:
[tiny-solver](https://crates.io/crates/tiny-solver) (Rust, dual-number
autodiff), [factrs](https://crates.io/crates/factrs) (Rust, dual-number
ForwardProp autodiff), [Ceres](http://ceres-solver.org) (C++ 2.2,
template autodiff; three linear-solver configurations),
[SymForce](https://symforce.org) (Skydio's symbolic code-generation
path; its own `sym::Optimizer`, templated over f64/f32), and
[g2o](https://github.com/RainerKuemmerle/g2o) (C++; custom vertices and
six custom edges with hand-derived analytic Jacobians, supernodal
CHOLMOD, landmarks marginalized via Schur). Unlike the pose-graph
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
  all six implementations (arael, tiny-solver, factrs, Ceres, SymForce,
  g2o) and the reference agree bit-for-bit.
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
`SLAM_DRIVER=nielsen` opts into it, `SLAM_LAMBDA0` overrides), tiny-solver
and Ceres `initial_trust_region_radius = 1e12`, SymForce
`initial_lambda = 1e-10` (it ships 1.0), factrs its default (`lambda`
starts at 1e-10 -- already near-Gauss-Newton), g2o `setUserLambdaInit(1e-9)`
(`G2O_LAMBDA_INIT` overrides). All stop at a shared termination class
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

## Results (aarch64 VM, single core, min of rounds)

### 60 poses (240 landmarks, 5,370 observations, 1,080 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     14.9 |   3(3) |    4.98 |         7.0 |    12.8 |  3062.0482 |
| arael LM f32                  |     40.3 | 3(11)* |    3.67 |         6.4 |    10.2 |  3062.0483 |
| tiny-solver LM                |    103.7 |      3 |   34.58 |        40.1 |    26.4 |  3062.0482 |
| factrs LM                     |     38.4 |      3 |   12.81 |        16.9 |    23.3 |  3062.0482 |
| ceres sparse_normal_cholesky  |     18.3 |   3(3) |    6.10 |        10.8 |    16.9 |  3062.0482 |
| ceres sparse_schur            |     19.5 |   3(3) |    6.49 |        11.6 |    16.5 |  3062.0482 |
| ceres iterative_schur         |     25.7 |   6(6) |    4.28 |         6.5 |    12.9 |  3067.3849 (RMSE 0.745 m) |
| symforce LM f64               |     27.4 |   3(4) |    6.84 |        18.0 |    27.1 |  3062.0482 |
| symforce LM f32               |     32.1 |   4(5) |    6.41 |        17.3 |    23.3 |  3062.0500 |
| g2o LM                        |     14.4 |   3(3) |    4.80 |         7.6 |    16.4 |  3062.0482 |

9/10 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

\* f32 hits the optimum by step 3, then the remaining steps can't improve
at the f32 noise floor and are rejected -- inflating total time, not
ms/iter.

### 125 poses (500 landmarks, 12,830 observations, 2,250 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     46.1 |   3(3) |   15.37 |        20.6 |    26.1 |  7424.1484 |
| arael LM f32                  |     40.4 |   3(3) |   13.47 |        17.8 |    20.4 |  7424.1506 |
| tiny-solver LM                |    263.3 |      3 |   87.77 |       106.9 |    56.1 |  7424.1484 |
| factrs LM                     |    100.1 |      3 |   33.36 |        43.4 |    50.9 |  7424.1484 |
| ceres sparse_normal_cholesky  |     51.3 |   3(3) |   17.10 |        30.2 |    28.4 |  7424.1485 |
| ceres sparse_schur            |     55.1 |   3(3) |   18.35 |        32.2 |    28.3 |  7424.1485 |
| ceres iterative_schur         |    598.2 | 30(30) |   19.94 |        15.6 |    18.8 |  7424.7966 (RMSE 0.179 m) |
| symforce LM f64               |     89.2 |   3(4) |   22.30 |        56.3 |    55.0 |  7424.1484 |
| symforce LM f32               |    126.1 |   5(6) |   21.01 |        49.9 |    46.3 |  7424.1592 |
| g2o LM                        |     43.0 |   3(3) |   14.32 |        20.9 |    32.1 |  7424.1484 |

9/10 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 250 poses (1,000 landmarks, 31,823 observations, 4,500 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |    200.6 |   3(3) |   66.87 |        83.8 |    65.4 | 18581.8936 |
| arael LM f32                  |    159.0 |   2(3) |   53.00 |        65.5 |    46.8 | 18581.8975 |
| tiny-solver LM                |    755.9 |      3 |  251.97 |       293.0 |   136.5 | 18581.8936 |
| factrs LM                     |    341.2 |      3 |  113.73 |       144.7 |   139.4 | 18581.8936 |
| ceres sparse_normal_cholesky  |    196.8 |   3(3) |   65.62 |       109.1 |    65.1 | 18581.8937 |
| ceres sparse_schur            |    207.1 |   3(3) |   69.03 |       110.9 |    65.9 | 18581.8937 |
| ceres iterative_schur         |    436.6 | 10(10) |   43.66 |        43.6 |    33.8 | 18583.3216 (RMSE 0.185 m) |
| symforce LM f64               |    344.1 |   3(4) |   86.02 |       174.1 |   137.0 | 18581.8936 |
| symforce LM f32               |    582.1 |   6(7) |   83.16 |       164.4 |   108.0 | 18581.9541 |
| g2o LM                        |    155.1 |   3(3) |   51.69 |        77.0 |    84.8 | 18581.8936 |

9/10 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

Every exact solver converges in 3 steps, so the tables compare per-step
cost (ms/iter) directly. g2o edges ahead at the largest size -- possibly
its supernodal CHOLMOD factorization (arael ships permissive faer),
though this was not verified by a backend swap.

## Running

```sh
cmake -B cpp/build cpp && cmake --build cpp/build   # Ceres + g2o
# add -DSYMFORCE_DIR=/path/to/symforce for the SymForce runner
# (regenerate its factor headers with:
#  symforce-venv/bin/python3 symforce_gen.py cpp/symforce_gen)
ROUNDS=3 cargo run --release                        # 60 poses (default)
SLAM_POSES=250 ROUNDS=3 cargo run --release
SLAM_SKIP_TINY=1 cargo run --release                # skip tiny-solver
SLAM_VERBOSE=1 cargo run --release                  # arael per-iteration trace
G2O_VERIFY_JAC=1 cpp/build/g2o_slam scene.txt lm out.txt   # check g2o Jacobians
```

Env knobs: `SLAM_POSES`, `ROUNDS`, `SLAM_SKIP_TINY`, `SLAM_NO_MEM`,
`SLAM_DRIVER=nielsen`, `SLAM_LAMBDA0`, `CERES_SOLVERS=<comma list>`,
`CERES_RADIUS0`, `TINY_MAXITER`, `TINY_RADIUS0`, `SYMFORCE_LAMBDA0`,
`SYMFORCE_MAXITER`, `G2O_LAMBDA_INIT`, `G2O_GAIN`, `G2O_VERIFY_JAC`.
