# 2D pose-graph benchmark: arael vs tiny-solver vs GTSAM

Batch pose-graph optimization on the two canonical 2D SLAM benchmark
datasets, comparing arael against
[tiny-solver](https://crates.io/crates/tiny-solver) (Rust, dual-number
autodiff, faer sparse Cholesky), [GTSAM](https://gtsam.org) (C++,
analytic factors, via the official Python wheel -- timing wraps only the
C++ `optimize()` call), [Ceres](http://ceres-solver.org) (C++ 2.2,
template autodiff, SPARSE_NORMAL_CHOLESKY, modeled on its own
pose_graph_2d example), and [g2o](https://github.com/RainerKuemmerle/g2o)
(C++ 2023-08, analytic Jacobians, CHOLMOD backend).

## Datasets

Vendored in `datasets/` (see `datasets/README.md` for provenance and
citations; `./fetch_datasets.sh` re-downloads from the original sources):

- **M3500** -- Olson's Manhattan world: 3500 poses, 5453 edges. The file
  shipped by tiny-solver-rs (Carlone's `input_M3500_g2o` revision), with
  odometry initialization and diagonal information matrices.
- **city10000** -- the iSAM city dataset: 10000 poses, 20687 edges. From
  the SE-Sync data collection, odometry initialization, diagonal
  information (50, 50, 100).

Three configurations: **M3500 unweighted** (identity information -- the
configuration tiny-solver's shipped benchmark runs, since its g2o reader
drops the info matrices), **M3500** and **city10000** with the files'
information matrices applied by every system. The shipped tiny-solver
benchmark also wraps a Huber(1.0) loss around every factor; the harness
uses no robust loss anywhere (arael has no Huber), which is
value-identical at these optima -- every residual block ends in the Huber
inlier region.

## Fairness rules

- **One cost function.** Every system optimizes the identical weighted
  least-squares problem: standard SE2 between factors with the file's
  sqrt-information row weights, plus a unit-weight prior on pose 0 fixing
  the gauge. tiny-solver's shipped g2o reader drops the information
  matrices, so the problem is assembled through its public `Factor` API
  with a weighted clone of its own `BetweenFactorSE2` -- solver, autodiff
  and linear algebra remain tiny-solver's. GTSAM's `readG2o` honors the
  information matrices natively.
- **One validation.** A single reference cost function (in `src/g2o.rs`)
  evaluates every system's final poses. A row counts as converged only
  when BOTH its cost is within 1% of the best AND its solution is within
  5 cm rigid-aligned RMSE of the best solution -- the cost surface has
  near-flat directions where a plateau under 1% above the optimum can
  sit meters away geometrically (g2o LM lands on exactly such a plateau
  on the weighted M3500). Hard asserts: arael rows must converge and at
  least one external system must agree (independent-implementation
  anchor).
- **One termination class.** All systems stop when a step improves the
  cost by less than 1e-5 absolute or 1e-5 relative -- the shipped
  defaults of both tiny-solver and GTSAM; arael is configured to the same
  thresholds (`patience = 1`).
- **Problem-appropriate initial damping for every LM.** Damping-schedule
  strategy is a per-problem tuning knob, not an intrinsic property of a
  solver -- so each LM implementation gets an initial damping suited to
  these well-initialized pose graphs instead of its shipped default:
  arael `initial_lambda = 1e-8` (its own docs recommend small values
  here), Ceres and tiny-solver `initial_trust_region_radius = 1e12`,
  g2o `setUserLambdaInit(1e-12)`, GTSAM its default `1e-5` (insensitive:
  its inner lambda search adapts within an iteration). All are
  env-overridable (`ARAEL_LAMBDA0`, `CERES_RADIUS0`, `TINY_RADIUS0`,
  `G2O_LAMBDA_INIT`, `GTSAM_LAMBDA0`; also `G2O_GAIN`). With this policy
  every LM converges in 6-7 steps on M3500 and the durable cross-system
  comparison becomes the METAL: time per step (linearize + assemble +
  factorize + solve). Shipped-default behavior is documented under
  "known solver behaviors" below.
- **Threads: verified, not assumed.** Watching `/proc/<pid>/status`
  during a solve showed the official GTSAM wheel spawns 8 TBB threads;
  tiny-solver parallelizes assembly and its faer factorization through
  rayon; arael is single-threaded by construction (`faer::Par::Seq`).
  The harness therefore enforces single-core itself: it sets every
  threading env var to 1 (OMP_NUM_THREADS, OPENBLAS/MKL/TBB/VECLIB/
  NUMEXPR, RAYON_NUM_THREADS) before any pool spawns, pins itself to
  CPU 0 with sched_setaffinity (inherited by the GTSAM subprocess), and
  ASSERTS the GTSAM subprocess's reported Cpus_allowed_list is "0".
  Measured effect on GTSAM was small (~3%) -- its TBB threads contribute
  little on these problems -- and tiny-solver's default multi-threaded
  numbers are listed separately.
- **Timing.** Total time is the minimum over N interleaved rounds
  (contention only ever slows a run down). First-iteration time is a
  fresh optimize capped at one iteration -- setup + first linearization +
  symbolic + numeric factorization + solve, identically defined for all
  systems.
- **Iteration semantics differ.** GTSAM and tiny-solver report outer
  iterations (inner damping retries hidden); arael reports both:
  "accepted(total)" -- accepted cost-decreasing steps, and total damped
  solve attempts including retries. ms/iter is computed from the total
  and is comparable only within a system; total time is the cross-system
  number.

## Precision

arael solves the same models in f64 and f32 (`solve_sparse_faer` /
`solve_sparse_faer_f32`), selected per model, both precisions in one
binary; both rows are validated against the common optimum. None of the
other systems offers single precision: tiny-solver's problem/optimizer
layer is hardwired f64 (its Factor trait alone is generic), GTSAM and
Ceres bake double into their public APIs, and g2o's number_t alias is
hardcoded to double with no build option.

## Known solver behaviors (reported, not hidden)

These were observed with the systems' SHIPPED initial damping (before
the problem-appropriate policy above); all three damping pathologies
disappear under the policy:

- **tiny-solver LM** (default trust region 1e4) checks its
  error-decrease thresholds even on a *rejected* damping step; a
  rejection reads as "no improvement" and terminates the solve -- on the
  weighted M3500 it stopped at cost 208 (optimum 138). With a
  problem-appropriate initial trust region it takes no rejected steps
  and converges everywhere.
- **g2o LM** auto-computes lambda0 = 1e-5 * max Hessian diagonal
  (additive lambda-I damping), which over-damps heavily weighted graphs;
  the resulting slow descent then trips the 1e-5 gain-threshold stop
  meters short of the optimum (cost 139.11, 1.9 m off, on weighted
  M3500) -- not a local minimum: with patience (gain 1e-9) or a small
  lambda0 it reaches the exact optimum.
- **Ceres LM** (default trust region 1e4) converged everywhere but
  ground ~30 near-identical iterations on M3500; with a large initial
  region it takes 6-7.
- **GTSAM on city10000 (batch, odometry init)**: LM stops in a local
  minimum (cost 2.39M vs 512) and GN diverges, at any damping -- this
  one is not a damping artifact. GTSAM's own city10000 examples solve
  it incrementally with ISAM2. Its Pose2 between-factor residual uses
  the SE(2) log map, whose basin of attraction from this initialization
  differs from the rotation-frame residual the other systems use.
- **gtsam ISAM2 row**: the incremental reference -- GTSAM driven the way
  its own city10000 example works (one update per pose, odometry-composed
  initial guesses, loop closures at their later endpoint). It DOES solve
  city10000 (cost 512.51, 0.1% above the batch optimum, within its
  default relinearization slack). It answers the online-estimation
  question, not the batch one: "iters" is incremental updates, timing is
  the sum of update() calls plus the final estimate, run once. Not
  comparable to the batch rows -- listed for context.

## Running

```sh
./fetch_datasets.sh
ROUNDS=5 cargo run --release      # single-core pinning is built in
GTSAM_PYTHON=/path/to/venv/bin/python3   # override if needed (pip install gtsam)
```

## Results (2026-07-05, aarch64 VM, single core enforced by the harness, min of 5 interleaved rounds)

Iters are "accepted(total)" where the system reports both; total
includes damping retries. Final costs are all evaluated by the one
reference cost function, so they are directly comparable. With the
initial-damping policy nearly every system converges in 6-7 steps, so
ms/iter -- the per-step pipeline cost -- is the durable comparison.

### M3500 unweighted (10500 parameters)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     19.8 |   6(6) |    3.31 |         6.3 |     3.0218 |
| arael LM f32    |     30.8 | 10(10) |    3.08 |         5.5 |     3.0219 |
| tiny-solver GN  |    123.9 |      6 |   20.64 |        28.3 |     3.0218 |
| tiny-solver LM  |    132.3 |      6 |   22.05 |        30.1 |     3.0218 |
| gtsam LM        |    111.2 |      7 |   15.88 |        17.1 |     3.0221 |
| gtsam GN        |     77.6 |      6 |   12.93 |        14.2 |     3.0221 |
| ceres LM        |     35.6 |   6(6) |    5.93 |        13.3 |     3.0218 |
| g2o LM          |     26.0 |      6 |    4.33 |         8.1 |     3.0218 |
| g2o GN          |     23.9 |      6 |    3.99 |         7.7 |     3.0218 |
| gtsam ISAM2 (incremental reference) | 650.8 | 3500 upd | 0.19 | 0.2 | 3.0246 |

### M3500 (10500 parameters, information matrices applied)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     19.9 |   6(6) |    3.32 |         6.0 |   137.9130 |
| arael LM f32    |     21.8 |   7(7) |    3.12 |         5.6 |   137.9544 |
| tiny-solver GN  |    122.4 |      6 |   20.40 |        28.4 |   137.9130 |
| tiny-solver LM  |    132.9 |      6 |   22.15 |        29.9 |   137.9130 |
| gtsam LM        |     95.3 |      6 |   15.88 |        17.6 |   137.9273 |
| gtsam GN        |     77.1 |      6 |   12.85 |        14.3 |   137.9273 |
| ceres LM        |     35.3 |   6(6) |    5.88 |        13.3 |   137.9136 |
| g2o LM          |     26.0 |      6 |    4.34 |         8.1 |   137.9136 |
| g2o GN          |     24.3 |      6 |    4.05 |         7.4 |   137.9136 |
| gtsam ISAM2 (incremental reference) | 658.2 | 3500 upd | 0.19 | 0.1 | 138.0320 |

### city10000 (30000 parameters, information matrices applied)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     87.4 |   7(7) |   12.49 |        22.4 |   511.9852 |
| arael LM f32    |     71.4 |   7(7) |   10.20 |        18.8 |   512.0045 |
| tiny-solver GN  |    569.3 |      7 |   81.33 |       111.3 |   511.9852 |
| tiny-solver LM  |    617.2 |      7 |   88.18 |       115.6 |   511.9852 |
| gtsam LM        |   4299.6 |     30 |  143.32 |        71.2 |   2.39e6 (local minimum) |
| gtsam GN        |    214.1 |      4 |   53.52 |        59.9 |   2.48e8 (diverged) |
| ceres LM        |    179.8 |   7(7) |   25.69 |        54.3 |   511.9880 |
| g2o LM          |    145.9 |      7 |   20.85 |        37.2 |   511.9880 |
| g2o GN          |    138.6 |      7 |   19.81 |        35.6 |   511.9880 |
| gtsam ISAM2 (incremental reference) | 10378.0 | 10000 upd | 1.04 | 0.1 | 512.5080 |

arael is the fastest system in every validated cell, in both total time
and per-step cost. 10/10 systems converge on both M3500 configurations
under the initial-damping policy; on city10000 batch GTSAM remains the
only non-converger (residual parameterization, not damping).

tiny-solver GN with its default rayon threading (8 cores, not
core-pinned): M3500 78.0 ms, city10000 386.9 ms.
