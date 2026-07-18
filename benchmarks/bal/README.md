# Bundle-adjustment benchmark: arael vs Ceres vs g2o (BAL)

Bundle adjustment on the [BAL](https://grail.cs.washington.edu/projects/bal/)
(Bundle Adjustment in the Large) datasets -- the standard BA solver
benchmark -- comparing arael against [Ceres](http://ceres-solver.org)
on Ceres's home turf (BAL is the problem family Ceres's examples,
defaults, and Schur solvers were built around) and
[g2o](https://github.com/RainerKuemmerle/g2o) in its own bal_example
configuration: flat 9-parameter camera vertices, point vertices
marginalized so g2o performs its Schur elimination (its sparse-BA
heritage), CHOLMOD on the reduced camera system, the residual
autodiffed with g2o's bundled AD. With unit information g2o's chi2 IS
the reference cost, asserted at the initial estimate like the others.

It runs on the shared [benchmarks/harness](../harness) -- the same probes,
timing rules, table and core pin as [pgo](../pgo/README.md),
[slam](../slam/README.md) and [loc](../loc/README.md). What is local to this
benchmark is its problem: the BAL loader, the one reference cost function every
system's final parameters are scored by, the initial cost ASSERTED against it on
every run, and a validation gate that has to cope with a 7-DOF gauge.

## Problem

BAL problems are the post-front-end refinement step of
structure-from-motion: given an initial reconstruction (3D points,
camera poses, per-camera intrinsics -- internet photos, so `f, k1, k2`
are per-image unknowns), minimize total squared reprojection error.
The residual is the Snavely convention: `X_cam = R X + t`, perspective
divide with negative z, radial distortion `1 + k1 r^2 + k2 r^4`, scale
by `f`; unit pixel weights, no robust loss, no gauge prior (the 7-DOF
gauge is left to LM damping, as Ceres runs these problems).

- Camera: 9 parameters. arael parameterizes the rotation with
  `EulerAngleParam` (delta composed with a re-centered reference,
  initialized from the file's Rodrigues vector); Ceres runs its own
  example's plain axis-angle block. The parameterization is
  solver-internal; the cost is defined on the rotation itself, and a
  unit test pins arael's symbolic residual to the reference cost on the
  real dataset to 1e-9 relative.
- Point: 3 parameters, `CrossBlock<Camera, Point>` per observation.

## Validation

A row converges when BOTH its reference cost is within 1% of the best
AND its similarity-aligned (rotation + translation + scale: the BA
gauge) camera-center RMSE is under 5e-3 of the scene extent. Landmark
positions are deliberately NOT the geometric gate: BAL scenes contain
weakly-observed points (two rays at tiny parallax) that slide along
their rays almost cost-free -- converged solutions with costs within
0.004% of each other differ by 3-8% in all-points RMSE. Camera centers
are constrained by hundreds of residuals each; the 5e-3 threshold sits
above the measured scatter of genuinely converged solutions (up to
~1.3e-3 between stops agreeing in cost to 0.13% -- the valley is that
flat) and below measured non-converged plateaus (5.8e-3 and beyond,
which fail the cost gate too).

## Damping

arael runs the gain-ratio `NielsenLambdaDriver` here, not the fixed-multiplier
ladder the pose-graph benchmarks use (`DRIVER=fixed` selects that one).

The initial damping is per dataset, and per system: each is tuned so that the
system's first two iterations are single accepted steps. That is what the
per-iteration measurement needs -- full-iter is t(2 iterations) - t(1 iteration),
and a rejected step in either of them makes it unreportable. `BAL_LAMBDAS` sweeps
the values (see Running).

| dataset | arael `ARAEL_LAMBDA0` | Ceres `CERES_RADIUS0` |
|---|---|---|
| Ladybug-49 | 5e-5 | 1e4 (Ceres's own default) |
| Ladybug-138 | 1e-3 | 1e2 |
| Ladybug-372 | 1e-4 | 1e4, `iterative_schur` 1e2 |
| Ladybug-1723 | 1e-1 | 1e1 |

g2o runs its auto lambda heuristic, which scales the initial damping to each
problem and already lands both steps (`G2O_LAMBDA_INIT` overrides it).

## What is measured

**One iteration.** Linearize, assemble, factorize, solve -- the pipeline every
system runs on every step, over the identical validated cost function. It is
measured as t(2 iterations) - t(1 iteration), so the one-time setup (first
assembly, ordering, symbolic factorization) cancels out, and it is reported only
when the first iteration was one accepted step, so a rejected step's wasted
factorization cannot leak into it.

This is the number that compares across systems.\* **Total time and iteration count
do not**, and reading them as a ranking will mislead you: they are set by each
system's damping schedule and its termination rule. The tables report them
because they are facts about the run, not because they are comparable.

> DEPRECATED: numbers below pending regeneration before release.

\* Ceres's `iterative_schur` solves the step equation with a conjugate-gradient
method rather than by factorization, so its iterations cannot be compared one on
one with the others'. It never forms the reduced camera system, only multiplies by
it, which is why its memory is the lowest in the field, and its per-iteration cost
moves with the damping.

The two arael routes are the full-system sparse solve (`sparse`) and the
Schur-complement solve (`schur`: points marginalized on every damped solve, only
the camera system factorized). They reach the same optimum by construction, so
the rows compare linear solvers directly.

## Results (2026-07-14, Apple M4 Pro, single core enforced by the harness, min of 32 interleaved rounds)

What each column means:

| column | meaning |
|--------|---------|
| **total ms** | the whole solve: setup plus every iteration, retries included. |
| **iters** | `accepted(attempts)`. An attempt is one linear solve; a rejected step raises the damping and costs a factorization. |
| **ms/iter** | total ms divided by attempts -- an average, and it carries the one-time setup amortized over however many iterations the solver took. |
| **full-iter** | one complete iteration. Measured as t(2 iterations) - t(1 iteration), so the setup cancels. The durable cross-system number. |
| **1st-iter ms** | one iteration plus the setup the others do not pay again. |
| **peak MB** | process high-water mark (`VmHWM`), each solver measured in a process of its own. |
| **final cost** | evaluated by the one reference cost function for every system. |

full-iter and 1st-iter are dropped ("-") for any system whose first iteration was
not a single accepted step: such an iteration is mostly wasted factorizations,
and every number derived from it inherits that.

### Ladybug-49 (23769 parameters)

| system                | total ms |  iters | ms/iter | full-iter | 1st-iter ms | peak MB | final cost |
|-----------------------|---------:|-------:|--------:|----------:|------------:|--------:|-----------:|
| arael LM f64 sparse   |    497.4 | 21(21) |   23.69 |     22.76 |        44.2 |    70.7 | 26689.2948 |
| arael LM f32 sparse   |    479.7 | 19(25) |   19.19 |     19.32 |        32.8 |    59.4 | 26691.0378 |
| arael LM f64 schur    |    236.2 | 21(21) |   11.25 |     10.76 |        15.5 |    38.4 | 26689.2948 |
| arael LM f32 schur    |    216.7 | 19(27) |    8.03 |      8.84 |        13.1 |    28.4 | 26692.2399 |
| ceres dense_schur     |    439.8 | 21(21) |   20.94 |     18.52 |        37.6 |    38.3 | 26689.7174 |
| ceres sparse_schur    |    436.8 | 21(21) |   20.80 |     18.73 |        47.8 |    41.7 | 26689.7174 |
| ceres iterative_schur\* |    644.1 | 23(23) |   28.00 |     29.63 |        34.2 |    36.9 | 26689.5841 |
| g2o LM (schur)        |    946.1 | 42(58) |   16.31 |     19.78 |        33.0 |    45.2 | 26714.5561 |

### Ladybug-138 (60876 parameters)

| system                | total ms |  iters | ms/iter | full-iter | 1st-iter ms | peak MB |  final cost |
|-----------------------|---------:|-------:|--------:|----------:|------------:|--------:|------------:|
| arael LM f64 sparse   |   1982.4 | 21(24) |   82.60 |     78.91 |       199.9 |   183.2 | 119055.9244 |
| arael LM f32 sparse   |   1682.6 | 21(24) |   70.11 |     63.85 |       179.5 |   151.4 | 119054.8981 |
| arael LM f64 schur    |   1016.6 | 21(24) |   42.36 |     41.02 |        66.6 |   115.2 | 119055.9244 |
| arael LM f32 schur    |    729.0 | 21(24) |   30.38 |     29.90 |        48.2 |    81.7 | 119055.3497 |
| ceres dense_schur     |   1790.0 | 22(23) |   77.83 |     72.60 |       134.1 |    97.9 | 119056.6359 |
| ceres sparse_schur    |   1631.4 | 22(23) |   70.93 |     65.98 |       162.0 |   105.6 | 119056.6359 |
| ceres iterative_schur\* |   2416.7 | 22(26) |   92.95 |     51.48 |       111.9 |    85.3 | 118753.5001 |
| g2o LM (schur)        |   3390.7 | 40(59) |   57.47 |     65.69 |       121.9 |   120.3 | 118904.3429 |

### Ladybug-372 (145617 parameters)

| system                | total ms |  iters | ms/iter | full-iter | 1st-iter ms | peak MB |  final cost |
|-----------------------|---------:|-------:|--------:|----------:|------------:|--------:|------------:|
| arael LM f64 sparse   |   5657.6 | 10(19) |  297.77 |    286.37 |       625.9 |   471.7 | 225431.3936 |
| arael LM f32 sparse   |   4176.5 | 10(18) |  232.03 |    220.60 |       531.5 |   367.6 | 225488.7411 |
| arael LM f64 schur    |   3824.5 | 10(19) |  201.29 |    206.66 |       303.8 |   350.1 | 225431.3936 |
| arael LM f32 schur    |   2105.8 |  9(16) |  131.61 |    134.60 |       209.7 |   239.7 | 225509.7227 |
| ceres dense_schur     |   7434.9 |  9(16) |  464.68 |    453.13 |       657.9 |   285.4 | 225447.1709 |
| ceres sparse_schur    |   4291.7 |  9(16) |  268.23 |    256.99 |       567.9 |   287.2 | 225447.1709 |
| ceres iterative_schur\* |   2288.6 | 12(16) |  143.04 |    148.36 |       319.2 |   191.9 | 225696.1695 |
| g2o LM (schur)        |   9291.8 | 28(35) |  265.48 |    277.01 |       434.1 |   357.7 | 226586.4232 |

All eight rows validate on all three datasets.

### Ladybug-1723 (485k parameters, exploratory: `BAL_ONLY=1723`)

**Not a validated result, and the table below is not a ranking.** With the damping
in the Damping section the iterations are measurable -- the four rows that run
report a full-iter -- but no system reaches the shared 1e-5 tolerances here. They
stop on different plateaus of a long descent, 765376 to 772215, a 0.9% spread, so
the mutual validation gate cannot pass and the run has nothing to anchor against.
The dataset is excluded from the default suite until it has a termination
criterion the systems can share. `full-iter` is the only column here that means
what it means elsewhere; `total ms`, `iters` and `final cost` are each reporting a
different, arbitrary stopping point.

| system                | total ms |   iters | ms/iter | full-iter | 1st-iter ms | peak MB |  final cost |
|-----------------------|---------:|--------:|--------:|----------:|------------:|--------:|------------:|
| arael LM f64 sparse   |  88469.0 |  20(30) | 2948.97 |   2910.07 |      3925.5 |  1818.2 |  769222.381 |
| arael LM f64 schur    |  54161.6 |  20(30) | 1805.39 |   1805.54 |      2198.2 |  1345.0 |  769222.381 |
| ceres sparse_schur    |  89710.9 |  24(31) | 2893.90 |   2813.32 |      4057.8 |  1250.1 |  766134.386 |
| ceres iterative_schur\* |   8706.3 |  16(23) |  378.53 |    397.21 |      1191.1 |   621.3 |  772215.452 |
| g2o LM (schur)        | 364393.4 | 84(112) | 3253.51 |   2318.33 |      2942.3 |  1540.0 |  765376.644 |

The Schur route still has the cheaper iteration here (1806 vs 2910 ms), on the
largest reduced system in the suite -- 15,507 parameters. That holds only under
nested dissection; see the section below.

`iterative_schur`'s result suggests that on larger systems a conjugate-gradient
method is likely the more viable route than a factorization. Its iteration against
arael's Schur route, across the suite: 2.8x more expensive at Ladybug-49, 1.3x at
138, then 1.4x CHEAPER at 372 and 4.5x cheaper here. It crosses over, and the lead
grows with the problem. It stops on a plateau like everything else on this
dataset, so read it as a direction, not a measurement.

**arael's f32 rows cannot run here at all** (0 accepted steps). The input data is
bad: 199 observations lie behind the camera and fourteen sit on the optical centre
(`pc.z` down to 3.65e-9). In f32 those cancel to exactly zero and the perspective
divide becomes 0/0, which arrives as a NaN on the Hessian diagonal. f64 has the
digits to survive it; f32 does not, and no damping value changes that. Running f32
here would need the data fixed and/or a loss function that tolerates a degenerate
projection. The smaller datasets carry no such observation, which is why f32
solves them.

## The Schur reduction, and the ordering it needs

**Marginalizing the points is automatic.** arael's sparse backend defaults to
`SchurPolicy::Auto`: it finds the eliminable blocks in the model's coupling graph
and decides for itself, declining the reduction when the reduced factor would hold
relatively more fill than the full one. It takes the reduction on all four BAL
datasets, and on the two largest the fill check runs and passes with room to spare
(0.47 on Ladybug-372, 0.55 on Ladybug-1723, against a decline threshold of 0.8).
The two arael rows in the tables exist because the benchmark OVERRIDES that policy
in both directions -- `Never` for the `sparse` rows, `Force` for the `schur` ones
-- so that it measures the two routes rather than the policy's verdict about them.

**The ordering is the opt-in, and it is what makes the reduction pay.** The
reduced system S is camera-camera, and a 3D point makes a CLIQUE of every camera
that sees it, so S is a union of cliques -- the structure AMD is worst at. The
benchmark therefore orders S with nested dissection (`BAL_ORDERING=amd` goes back
to AMD, which is arael's own default). It is not a small effect: `cargo run -r
--bin schur_stats` measures S under AMD and reports 4716 ms to factorize it on
Ladybug-1723, where under nested dissection the WHOLE Schur iteration -- assembly,
reduction, factorization, solve -- takes 1806 ms. The same swap moves the fill
ratio the policy decides on from 1.21 to 0.55, which is the difference between
declining the reduction and taking it.

With that ordering the Schur route has the cheaper iteration on every dataset in
the suite, including the 1723 exploratory one (1806 ms against the full system's
2910). `schur_stats` reports S's size, density, fill and the split between forming
it and factorizing it, per dataset.

## Covariance recovery (2026-07-16, Apple M4 Pro, single core)

Parameter covariance at the solution, `Sigma = 2 H^-1`, recovered without
inverting `H`. Intrinsics are held (known calibration), so a camera marginal is
its 6-DOF pose; the gauge is fixed by holding cameras 0 and 1. Three arael methods,
and the cost of scaling from one marginal to all of them:

- **`PerQuery`** factors `H` once, then solves for each queried block -- cheap for
  a few, linear in the count.
- **`AllMarginals`** runs one bulk selected inverse over the factor: every camera
  AND point marginal at once, at a cost that does not grow with how many you read.
- **Ceres** (`SPARSE_QR`) and **g2o** (`computeMarginals`) recover the same
  marginals for comparison. arael and Ceres build cold (assemble + factor +
  query); g2o reuses the factor from the solve it just ran (warm). All agree on the
  std devs to four figures.

BAL is rank-deficient at the solution (weakly triangulated point depths), which
Ceres's QR does not invert without a small anchoring prior on the points; arael
and g2o (Cholesky) factor through it and need none. g2o marginalizes the points,
so it recovers camera poses only.

Time to recover N marginals from the solved state, median ms (reps). `all` is
every free camera / every point; `-` a count a method does not cover; `*` a cell
past the 120 s cap (`COV_CELL_CAP_S`).

Camera pose (6-DOF):

| method | 1 | 2 | 8 | 32 | all |
|--------|--:|--:|--:|--:|--:|
| **Ladybug-49** (all=47) | | | | | |
| arael PerQuery | 44.5 (113) | 46.9 (107) | 60.3 (83) | 112.5 (45) | 145.0 (35) |
| arael AllMarginals | - | - | - | - | 76.8 (65) |
| Ceres SPARSE_QR | 276.4 (19) | 279.4 (18) | 296.2 (17) | 356.7 (15) | 395.1 (13) |
| g2o computeMarginals | 12.9 (200) | 12.9 (200) | 19.1 (200) | 20.9 (200) | 21.5 (200) |
| **Ladybug-138** (all=136) | | | | | |
| arael PerQuery | 184.7 (27) | 190.6 (27) | 227.1 (22) | 377.8 (14) | 1020.3 (5) |
| arael AllMarginals | - | - | - | - | 330.1 (16) |
| Ceres SPARSE_QR | 1740.2 (3) | 1795.0 (3) | 1834.4 (3) | 3134.3 (2) | 3068.9 (2) |
| g2o computeMarginals | 17.4 (200) | 54.6 (92) | 281.7 (18) | 294.4 (17) | 293.7 (18) |
| **Ladybug-372** (all=370) | | | | | |
| arael PerQuery | 541.4 (10) | 552.1 (10) | 649.3 (8) | 1061.4 (5) | 6745.3 (1) |
| arael AllMarginals | - | - | - | - | 1488.8 (4) |
| Ceres SPARSE_QR | 15594 (1) | 15057 (1) | 15281 (1) | 16006 (1) | * |
| g2o computeMarginals | 345.0 (16) | 3540.6 (2) | 3519.3 (2) | 6817.4 (1) | 7004.9 (1) |

Point (3-DOF). `AllMarginals` is the same bulk pass as above (it returns cameras
and points together):

| method | 1 | 2 | 8 | 32 | all |
|--------|--:|--:|--:|--:|--:|
| **Ladybug-49** (all=7776) | | | | | |
| arael PerQuery | 44.1 (114) | 45.6 (110) | 54.2 (93) | 87.9 (57) | 10903 (1) |
| arael AllMarginals | - | - | - | - | 76.8 (65) |
| Ceres SPARSE_QR | 276.0 (19) | 278.5 (18) | 290.5 (18) | 342.0 (15) | 16049 (1) |
| **Ladybug-138** (all=19878) | | | | | |
| arael PerQuery | 183.1 (28) | 186.1 (27) | 210.3 (24) | 302.3 (17) | * |
| arael AllMarginals | - | - | - | - | 330.1 (16) |
| Ceres SPARSE_QR | 1779.2 (3) | 1790.5 (3) | 1842.6 (3) | 1992.3 (3) | * |
| **Ladybug-372** (all=47423) | | | | | |
| arael PerQuery | 528.5 (10) | 536.3 (10) | 601.4 (9) | 861.4 (6) | * |
| arael AllMarginals | - | - | - | - | 1488.8 (4) |
| Ceres SPARSE_QR | 15096 (1) | 15066 (1) | 15136 (1) | 15690 (1) | * |

`BAL_COV=1 cargo run --release` reproduces this (`COV_BUDGET_S` the per-cell
budget, `COV_CELL_CAP_S` the cap).

## Running

```sh
./fetch_datasets.sh                                  # Ladybug-49 is vendored
cmake -B cpp/build cpp && cmake --build cpp/build    # Ceres, g2o (+ cholmod)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1      # before load; see below
ROUNDS=5 cargo run --release
BAL_ONLY=1723 cargo run --release                    # the exploratory dataset
BAL_COV=1 cargo run --release                        # covariance recovery (above)
cargo run -r --bin schur_stats                       # S's size, density, cost split
```

The run prints all of its settings in a header, so a pasted result carries what
produced it.

| env | effect |
|-----|--------|
| `ROUNDS` | interleaved rounds; the reported time is the minimum over them |
| `BAL_ONLY` | substring; runs only the matching datasets (`1723` reaches the exploratory one) |
| `BAL_COV` | covariance-recovery benchmark instead of the solve (`COV_BUDGET_S` sets the per-cell budget) |
| `BAL_SYSTEMS` | comma-separated substrings; runs only the matching rows (a filtered run cannot validate across systems) |
| `ARAEL_LAMBDA0`, `G2O_LAMBDA_INIT`, `CERES_RADIUS0` | initial damping, per system |
| `BAL_LAMBDAS` | comma-separated damping values: sweep them instead of running the benchmark (below) |
| `DRIVER=fixed` | arael's fixed damping ladder instead of the gain-ratio driver it defaults to here |
| `BAL_ORDERING=amd` | AMD on the reduced camera system instead of nested dissection |
| `BAL_SCHUR_POLICY=auto` | let the fill analysis decide the reduction instead of forcing it |
| `VERBOSE` | arael's per-iteration solver trace |
| `TIMING` | arael's per-solve phase breakdown |
| `BAL_NO_MEM` | skip the peak-memory pass |

The thread caps must be exported **before** the process starts: OpenBLAS sizes
its pool when its shared library loads, ahead of any code that could set them,
and those early threads escape the single-core pin.

### Tuning the damping

`BAL_LAMBDAS` sweeps damping values instead of running the benchmark. For each
selected row and each value it runs ONE solve capped at 1 iteration and one
capped at 2 -- no warmup, no sub-rounds, no full solve, no memory pass -- and
says whether those two iterations were clean:

```sh
BAL_ONLY=1723 BAL_SYSTEMS="f64 schur" BAL_LAMBDAS=1e-2,1e-1,1 cargo run --release
```

```
system                     damping    t(1) acc/att     t(2)   acc  full-iter    cost after 2
arael LM f64 schur            1e-2    2199   1/1     4019   1/-          -      21104621.5   <- first two iterations not clean
arael LM f64 schur            1e-1    2210   1/1     4031   2/-       1821       7300667.1
arael LM f64 schur             1e0    2224   1/1     4042   2/-       1818       8545466.9
```

That is all the per-iteration number needs -- full-iter is t(2) - t(1), reported
only when the first iteration was a single accepted step -- and it is the
difference between seconds and hours on Ladybug-1723. The values are arael's
initial lambda for the arael rows, Ceres's trust-region radius for its rows, and
g2o's initial lambda for g2o (which normally runs its own auto heuristic, so
forcing a value there is usually the wrong move). The external runners are held
to the same two iterations by `BENCH_QUICK`, which never belongs in a
measurement.

`--features cholmod-gpl` adds an `arael LM f64 cholmod-gpl` row solving the full
system with CHOLMOD's supernodal backend (not in the tables above, which come
from a default build). That module is GPL-licensed, so the resulting binary is
subject to the GPL -- enable knowingly. Measured earlier on this benchmark, it is
SLOWER than faer at every size, so it is not the row to reach for here. Such a
build also links a BLAS/LAPACK stack whose shared-library baseline inflates every
arael row's peak memory; `BENCH_MEM_EXE=<default-build binary>` sources those
rows from a clean build instead.
