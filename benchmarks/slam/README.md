# Heterogeneous visual-inertial SLAM benchmark

Batch optimization of a synthetic visual-inertial SLAM problem,
comparing arael against three other solvers:
[tiny-solver](https://crates.io/crates/tiny-solver) (Rust, dual-number
autodiff), [factrs](https://crates.io/crates/factrs) (Rust, dual-number
ForwardProp autodiff), [Ceres](http://ceres-solver.org) (C++ 2.2,
template autodiff; four linear-solver configurations), and
[SymForce](https://symforce.org) (Skydio's symbolic code-generation
path; its own `sym::Optimizer`, templated over f64/f32). Unlike the
pose-graph ([benchmarks/pgo](../pgo)) and bundle-adjustment
([benchmarks/bal](../bal)) benchmarks -- each a single factor type --
this problem is heterogeneous: six factor types, several nonlinear.

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
  all five implementations (arael, tiny-solver, factrs, Ceres, SymForce)
  and the reference agree bit-for-bit.
- **One validation.** A row is "at the common optimum" when its cost is
  within 1% of the best AND its per-pose translation RMSE to the best
  solution is under 5 cm (the gauge is fixed by GPS + priors, so
  absolute positions are comparable without alignment).
- **Timing.** Total time is the min over N interleaved rounds; per-step
  (ms/iter) is total / total iterations; first-iteration time is a fresh
  solve capped at one iteration. arael, Ceres, and SymForce report
  `accepted(total)` steps.
- **Peak memory** (peak MB) is the process high-water mark (`VmHWM`),
  each solver measured in a fresh process: arael, tiny-solver, and
  factrs via a single-solver subprocess of this binary; Ceres and
  SymForce as their own subprocesses.
- **Single core**, verified: threading env vars forced to 1, process
  pinned to a fixed core, each subprocess's `Cpus_allowed_list` asserted.

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
starts at 1e-10 -- already near-Gauss-Newton). All stop at a shared
termination class (1e-5 absolute or relative). With this policy every
exact-factorization solver converges in 3-6 steps, so **ms/iter -- the
per-step pipeline cost (linearize + assemble + factorize + solve) -- is
the durable cross-system comparison**; total time and iteration count
reflect the damping strategy, which is tuning, not implementation.

arael runs f64 and f32; its residuals and SymForce's are code-generated
(SymForce from `symforce_gen.py`, flattened C++ under `cpp/symforce_gen/`).
Ceres runs three linear solvers: the two exact factorizations
(`sparse_normal_cholesky`, `sparse_schur`) are the like-for-like per-step
rows; `iterative_schur` is matrix-free preconditioned CG -- cheaper per
step but inexact, and at near-Gauss-Newton damping it does not reach the
validation gate here.

## Results (aarch64 VM, single core, min of rounds)

### 60 poses (240 landmarks, 5,370 observations, 1,080 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     14.5 |   3(3) |    4.84 |         6.8 |    12.6 |  3062.0482 |
| arael LM f32                  |     38.8 |  3(11) |    3.53 |         6.2 |    10.1 |  3062.0483 |
| tiny-solver LM                |    104.0 |      3 |   34.65 |        40.0 |    26.5 |  3062.0482 |
| factrs LM                     |     37.9 |      3 |   12.62 |        16.4 |    23.3 |  3062.0482 |
| ceres sparse_normal_cholesky  |     18.1 |   3(3) |    6.03 |        10.3 |    16.9 |  3062.0482 |
| ceres sparse_schur            |     18.4 |   3(3) |    6.14 |        11.3 |    16.5 |  3062.0482 |
| ceres iterative_schur         |     24.9 |   6(6) |    4.15 |         6.3 |    12.9 |  3067.3849 (RMSE 0.745 m) |
| symforce LM f64               |     26.9 |   3(4) |    6.72 |        17.4 |    27.1 |  3062.0482 |
| symforce LM f32               |     31.7 |   4(5) |    6.34 |        16.6 |    23.3 |  3062.0500 |

8/9 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 125 poses (500 landmarks, 12,830 observations, 2,250 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |     43.3 |   3(3) |   14.43 |        19.2 |    26.1 |  7424.1484 |
| arael LM f32                  |     37.6 |   3(3) |   12.52 |        17.6 |    20.3 |  7424.1506 |
| tiny-solver LM                |    254.0 |      3 |   84.67 |        95.8 |    54.0 |  7424.1484 |
| factrs LM                     |     98.9 |      3 |   32.96 |        42.2 |    50.9 |  7424.1484 |
| ceres sparse_normal_cholesky  |     49.8 |   3(3) |   16.60 |        29.3 |    28.4 |  7424.1485 |
| ceres sparse_schur            |     56.5 |   3(3) |   18.82 |        31.5 |    28.3 |  7424.1485 |
| ceres iterative_schur         |    598.1 | 30(30) |   19.94 |        15.7 |    18.8 |  7424.7966 (RMSE 0.179 m) |
| symforce LM f64               |     86.5 |   3(4) |   21.62 |        51.3 |    55.0 |  7424.1484 |
| symforce LM f32               |    123.7 |   5(6) |   20.62 |        48.2 |    46.3 |  7424.1592 |

8/9 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

### 250 poses (1,000 landmarks, 31,823 observations, 4,500 parameters)

| system                        | total ms | iters  | ms/iter | 1st-iter ms | peak MB | final cost |
|-------------------------------|---------:|-------:|--------:|------------:|--------:|-----------:|
| arael LM f64                  |    191.7 |   3(3) |   63.92 |        79.3 |    65.2 | 18581.8936 |
| arael LM f32                  |    152.2 |   2(3) |   50.73 |        63.4 |    46.7 | 18581.8975 |
| tiny-solver LM                |    733.4 |      3 |  244.47 |       277.8 |   152.3 | 18581.8936 |
| factrs LM                     |    336.6 |      3 |  112.19 |       140.9 |   139.5 | 18581.8936 |
| ceres sparse_normal_cholesky  |    191.1 |   3(3) |   63.69 |       106.2 |    65.1 | 18581.8937 |
| ceres sparse_schur            |    200.4 |   3(3) |   66.80 |       108.4 |    65.9 | 18581.8937 |
| ceres iterative_schur         |    412.2 | 10(10) |   41.22 |        40.0 |    33.8 | 18583.3216 (RMSE 0.185 m) |
| symforce LM f64               |    329.8 |   3(4) |   82.44 |       170.6 |   137.0 | 18581.8936 |
| symforce LM f32               |    565.5 |   6(7) |   80.79 |       160.8 |   108.0 | 18581.9541 |

8/9 at the common optimum; ceres iterative_schur (inexact CG) does not
reach the gate.

Under the unified damping policy every exact solver converges in 3 steps,
so these tables compare the per-step pipeline cost directly. arael f32 has
the lowest ms/iter at every size (3.5 / 12.5 / 50.7 at 60 / 125 / 250);
arael f64 leads the exact field at 60 and 125 and is level with Ceres's
exact solvers at 250 (63.9 vs 63.7). The rest follow: Ceres
(sparse_normal_cholesky / sparse_schur), then SymForce, then factrs, then
tiny-solver. arael also holds the lowest f32 peak memory. ceres
iterative_schur's matrix-free step is cheap but inexact and lands off the
optimum at this damping. (arael's f32 total time at 60 poses reflects a
few damping retries that single precision triggers near convergence -- the
per-step cost, the metric here, is unaffected.)

## Running

```sh
cmake -B cpp/build cpp && cmake --build cpp/build   # Ceres
# add -DSYMFORCE_DIR=/path/to/symforce for the SymForce runner
# (regenerate its factor headers with:
#  symforce-venv/bin/python3 symforce_gen.py cpp/symforce_gen)
ROUNDS=3 cargo run --release                        # 60 poses (default)
SLAM_POSES=250 ROUNDS=3 cargo run --release
SLAM_SKIP_TINY=1 cargo run --release                # skip tiny-solver
SLAM_VERBOSE=1 cargo run --release                  # arael per-iteration trace
```

Env knobs: `SLAM_POSES`, `ROUNDS`, `SLAM_SKIP_TINY`, `SLAM_NO_MEM`,
`SLAM_DRIVER=nielsen`, `SLAM_LAMBDA0`, `CERES_SOLVERS=<comma list>`,
`CERES_RADIUS0`, `TINY_MAXITER`, `TINY_RADIUS0`, `SYMFORCE_LAMBDA0`,
`SYMFORCE_MAXITER`.
