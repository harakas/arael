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

- **tiny-solver LM** checks its error-decrease thresholds even on a
  *rejected* damping step; a rejection leaves the error unchanged, which
  reads as "no improvement" and terminates the solve. On M3500 it stops
  at cost 208 (optimum 138). Its own examples benchmark with GN.
- **GTSAM on city10000 (batch, odometry init)**: LM stops in a local
  minimum (cost 2.39M vs 512) after 30 iterations regardless of
  tolerances; GN diverges. GTSAM's own city10000 examples solve it
  incrementally with ISAM2. Its Pose2 between-factor residual uses the
  SE(2) log map, whose basin of attraction from this initialization
  differs from the rotation-frame residual arael and tiny-solver use.
- **g2o LM on the weighted datasets** stops on flat plateaus: cost 139.11
  (0.9% above the 137.91 optimum but 1.9 m away geometrically) on M3500,
  1484.7 on city10000. Its GN converges everywhere and is the fastest
  system on both M3500 configurations. g2o's termination is a relative
  chi2 gain threshold (SparseOptimizerTerminateAction, 1e-5).
- **Ceres LM** converges everywhere; its trust-region loop grinds ~30
  near-identical iterations on M3500 before its function_tolerance
  fires.
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

Iters are "accepted(total)" where the system reports both (arael,
ceres); total includes damping retries. Final costs are all evaluated
by the one reference cost function, so they are directly comparable.

### M3500 unweighted (10500 parameters) -- tiny-solver's shipped benchmark configuration

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     28.5 |   9(9) |    3.17 |         5.9 |     3.0218 |
| arael LM f32    |     43.6 | 15(15) |    2.91 |         5.5 |     3.0219 |
| tiny-solver GN  |    131.3 |      6 |   21.88 |        29.8 |     3.0218 |
| tiny-solver LM  |    615.7 |     28 |   21.99 |        31.5 |     3.0218 |
| gtsam LM        |    110.9 |      7 |   15.84 |        17.7 |     3.0221 |
| gtsam GN        |     76.7 |      6 |   12.78 |        14.2 |     3.0221 |
| ceres LM        |    149.1 | 29(30) |    4.97 |        13.3 |     3.0219 |
| g2o LM          |    122.6 |     31 |    3.95 |         8.1 |     3.0218 |
| g2o GN          |     24.5 |      6 |    4.09 |         7.8 |     3.0218 |
| gtsam ISAM2 (incremental reference) | 676.3 | 3500 upd | 0.19 | 0.1 | 3.0246 |

### M3500 (10500 parameters, information matrices applied)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     88.6 | 21(31) |    2.86 |         5.8 |   137.9310 |
| arael LM f32    |    148.3 | 36(55) |    2.70 |         5.5 |   137.9338 |
| tiny-solver GN  |    129.1 |      6 |   21.52 |        29.1 |   137.9130 |
| tiny-solver LM  |    139.3 |      6 |   23.22 |        30.3 |   208.3 (not converged) |
| gtsam LM        |     97.1 |      6 |   16.18 |        17.9 |   137.9273 |
| gtsam GN        |     78.7 |      6 |   13.12 |        14.7 |   137.9273 |
| ceres LM        |    140.9 | 27(28) |    5.03 |        13.5 |   137.9355 |
| g2o LM          |     94.3 |     22 |    4.29 |         8.2 |   139.1 (plateau, 1.9 m off) |
| g2o GN          |     24.7 |      6 |    4.11 |         7.7 |   137.9136 |
| gtsam ISAM2 (incremental reference) | 649.9 | 3500 upd | 0.19 | 0.1 | 138.0320 |

### city10000 (30000 parameters, information matrices applied)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |    106.4 |   8(9) |   11.82 |        20.9 |   511.9880 |
| arael LM f32    |     90.6 |   9(9) |   10.07 |        18.4 |   511.9883 |
| tiny-solver GN  |    602.3 |      7 |   86.05 |       114.1 |   511.9852 |
| tiny-solver LM  |    891.2 |     10 |   89.12 |       119.2 |   511.9881 |
| gtsam LM        |   4204.1 |     30 |  140.14 |        68.7 |   2.39e6 (local minimum) |
| gtsam GN        |    213.2 |      4 |   53.30 |        58.4 |   2.48e8 (diverged) |
| ceres LM        |    243.4 | 10(10) |   24.34 |        54.1 |   511.9886 |
| g2o LM          |    342.5 |     18 |   19.03 |        37.7 |   1484.7 (plateau) |
| g2o GN          |    136.2 |      7 |   19.46 |        35.6 |   511.9880 |
| gtsam ISAM2 (incremental reference) | 10417.3 | 10000 upd | 1.04 | 0.1 | 512.5080 |

Headline reading: arael is the fastest system on city10000 (both
precisions beat every competitor); g2o GN is the fastest on both M3500
configurations. arael's per-iteration cost is the lowest of all batch
systems throughout; on the weighted M3500 its fixed-factor lambda
schedule spends 10 of 31 attempts on rejected retries chasing the last
0.3% of cost (a known outer-loop work item), which is exactly the gap
to g2o GN there.

tiny-solver with its default rayon threading (8 cores, not core-pinned):
M3500 GN 78.0 ms, city10000 GN 386.9 ms / LM 630.1 ms.
