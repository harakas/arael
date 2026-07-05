# 2D pose-graph benchmark: arael vs tiny-solver vs GTSAM

Batch pose-graph optimization on the two canonical 2D SLAM benchmark
datasets, comparing arael against
[tiny-solver](https://crates.io/crates/tiny-solver) (Rust, dual-number
autodiff, faer sparse Cholesky) and [GTSAM](https://gtsam.org) (C++,
analytic factors, via the official Python wheel -- timing wraps only the
C++ `optimize()` call).

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
  evaluates every system's final poses. Hard asserts: arael rows must
  reach the best cost within 1%, at least one *external* system must
  agree (independent-implementation anchor), and all converged solutions
  must agree pairwise under rigid alignment to < 5 cm RMSE.
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
`solve_sparse_faer_f32`); both rows are validated against the common
optimum. tiny-solver's problem/optimizer layer and GTSAM are f64-only, so
no f32 rows exist for them.

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

arael iters are "accepted(total)": accepted cost-decreasing steps, and
total damped solve attempts including retries. Final costs are all
evaluated by the one reference cost function, so they are directly
comparable.

### M3500 unweighted (10500 parameters) -- tiny-solver's shipped benchmark configuration

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     29.0 |   9(9) |    3.23 |         6.4 |     3.0218 |
| arael LM f32    |     44.3 | 15(15) |    2.95 |         5.9 |     3.0219 |
| tiny-solver GN  |    130.4 |      6 |   21.74 |        30.0 |     3.0218 |
| tiny-solver LM  |    625.8 |     28 |   22.35 |        32.3 |     3.0218 |
| gtsam LM        |    115.2 |      7 |   16.46 |        18.8 |     3.0221 |
| gtsam GN        |     81.6 |      6 |   13.60 |        14.8 |     3.0221 |
| gtsam ISAM2 (incremental reference) | 695.5 | 3500 upd | 0.20 | 0.1 | 3.0246 |

### M3500 (10500 parameters, information matrices applied)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |     89.9 | 21(31) |    2.90 |         6.1 |   137.9310 |
| arael LM f32    |    149.2 | 36(55) |    2.71 |         5.6 |   137.9338 |
| tiny-solver GN  |    128.1 |      6 |   21.35 |        29.5 |   137.9130 |
| tiny-solver LM  |    138.3 |      6 |   23.05 |        31.4 |   208.3 (not converged) |
| gtsam LM        |     98.6 |      6 |   16.44 |        18.3 |   137.9273 |
| gtsam GN        |     79.8 |      6 |   13.31 |        15.2 |   137.9273 |
| gtsam ISAM2 (incremental reference) | 668.6 | 3500 upd | 0.19 | 0.1 | 138.0320 |

### city10000 (30000 parameters, information matrices applied)

| system          | total ms |  iters | ms/iter | 1st-iter ms | final cost |
|-----------------|---------:|-------:|--------:|------------:|-----------:|
| arael LM f64    |    111.0 |   8(9) |   12.34 |        22.1 |   511.9880 |
| arael LM f32    |     89.8 |   9(9) |    9.98 |        19.2 |   511.9883 |
| tiny-solver GN  |    601.1 |      7 |   85.87 |       114.4 |   511.9852 |
| tiny-solver LM  |    904.0 |     10 |   90.40 |       121.3 |   511.9881 |
| gtsam LM        |   4405.2 |     30 |  146.84 |        73.6 |   2.39e6 (local minimum) |
| gtsam GN        |    221.1 |      4 |   55.27 |        60.3 |   2.48e8 (diverged) |
| gtsam ISAM2 (incremental reference) | 11148.4 | 10000 upd | 1.11 | 0.1 | 512.5080 |

On the weighted M3500, arael's fixed-factor lambda schedule oscillates
around the optimal damping (heterogeneous edge weights narrow the good-
lambda band): 10 of 31 attempts are rejected retries and the accepted
tail chases the last 0.3% of cost -- a known outer-loop work item;
per-iteration cost is unaffected.

tiny-solver with its default rayon threading (8 cores, not core-pinned):
M3500 GN 78.0 ms, city10000 GN 386.9 ms / LM 630.1 ms.
