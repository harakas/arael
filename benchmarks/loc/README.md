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
  switches to the general sparse solver for comparison.
- **tiny-solver**, **factrs** -- Rust, dual-number autodiff, their own LM.
- **Ceres** (3 linear solvers), **g2o**, **GTSAM** -- C++, custom factors with
  analytic Jacobians (g2o/GTSAM) or autodiff (Ceres), each cross-checked to
  the reference cost and, for g2o/GTSAM, Jacobian-verified against finite
  differences (`G2O_VERIFY_JAC=1`, `GTSAM_VERIFY_JAC=1`).

The bearing factor is **single-variable** (the fixed landmark carries no
derivatives): a remote block on the pose in arael, a unary edge/factor in
g2o/GTSAM.

## Results

Single core, min of N interleaved rounds. All systems converge in 3 steps to
the same optimum; **ms/iter -- the per-step pipeline cost -- is the headline
metric.** `LOC_POSES=N` sets the pose count (landmarks scale as `4N`).

### 60 poses (240 landmarks, 5,357 observations, 360 parameters)

| system                       | ms/iter | 1st-iter ms | peak MB | final cost |
|------------------------------|--------:|------------:|--------:|-----------:|
| **arael LM f64 (band)**      |  **0.37** |       0.4 |     4.6 |  3274.6025 |
| **arael LM f32 (band)**      |  **0.37** |       0.4 |     4.0 |  3274.6025 |
| g2o LM                       |    1.57 |         2.2 |    11.2 |  3274.6025 |
| ceres sparse_normal_cholesky |    1.51 |         3.1 |    13.0 |  3274.6025 |
| ceres sparse_schur           |    1.58 |         3.1 |    12.9 |  3274.6025 |
| ceres iterative_schur        |    1.68 |         2.9 |    12.1 |  3274.6025 |
| gtsam LM                     |    3.51 |         4.4 |    14.6 |  3274.6025 |
| factrs LM                    |    4.72 |         6.3 |    11.4 |  3274.6025 |
| tiny-solver LM               |   23.14 |        25.4 |    12.4 |  3274.6025 |

### 300 poses (1,200 landmarks, 41,323 observations, 1,800 parameters)

| system                       | ms/iter | 1st-iter ms | peak MB | final cost |
|------------------------------|--------:|------------:|--------:|-----------:|
| **arael LM f64 (band)**      |  **3.05** |       3.0 |    13.3 | 25269.0409 |
| **arael LM f32 (band)**      |  **3.00** |       3.0 |     9.6 | 25269.0410 |
| ceres sparse_normal_cholesky |   12.13 |        23.9 |    42.0 | 25269.0409 |
| g2o LM                       |   13.10 |        19.1 |    37.8 | 25269.0409 |
| ceres iterative_schur        |   13.22 |        22.6 |    36.6 | 25269.0409 |
| gtsam LM                     |   31.83 |        34.6 |    58.8 | 25269.0409 |
| factrs LM                    |   35.34 |        46.3 |    64.4 | 25269.0409 |
| tiny-solver LM               |  187.75 |       211.0 |    67.6 | 25269.0409 |

## Analysis

**The band solver is the whole story.** On a block-tridiagonal system band
Cholesky is O(n) in the pose count, while the general sparse and
Schur-complement solvers pay for a fill-reducing ordering and a sparse
factorization the structure does not need. arael leads by **~4x per iteration**
at both sizes (0.37 vs g2o/Ceres ~1.5 at 60 poses; 3.0 vs ~12-13 at 300), and
the gap holds because arael's band scales near-linearly (0.37 -> 3.0 for 5x the
poses) while the others scale worse. arael's peak memory is also a fraction of
the field at scale -- 13 MB vs 37-68 MB at 300 poses -- because a band factor
stores O(n * kd) rather than a general sparse Cholesky's fill-in.

Running the same problem with `LOC_ARAEL_SOLVER=faer` shows what the structure
buys: arael's general sparse path lands in the same neighborhood as g2o/Ceres,
so the win is the band solver, not the assembly.

f32 matches f64 to the final cost here (the problem is well-conditioned) at
lower memory -- the band factor and its Cholesky halve in width.

## Running

```sh
ROUNDS=10 cargo run --release                    # 60 poses (default)
LOC_POSES=300 ROUNDS=5 cargo run --release       # larger tier
LOC_ARAEL_SOLVER=faer cargo run --release        # arael on general sparse
LOC_NO_MEM=1 cargo run --release                 # skip peak-memory subprocess
```

The C++ runners execute as subprocesses over an exported copy of the scene;
they are skipped with a warning if their binary is absent.

```sh
cmake -B cpp/build cpp && cmake --build cpp/build   # Ceres, g2o, GTSAM
# g2o needs libg2o-dev + cholmod; GTSAM needs libgtsam-dev (+ libtbb-dev)
```

Verification (never in the timed solve): `G2O_VERIFY_JAC=1` and
`GTSAM_VERIFY_JAC=1` check each analytic Jacobian against finite differences.

### SymForce

The SymForce runner is not yet built here. It needs the SymForce Python
codegen toolchain (`pip install -e` of a SymForce source checkout) to
regenerate the loc factor set, then `-DSYMFORCE_DIR=<checkout>` at CMake time,
mirroring [`benchmarks/slam`](../slam/README.md#symforce).
