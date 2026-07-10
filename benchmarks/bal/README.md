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

Same methodology as [benchmarks/pgo](../pgo/README.md): one loader,
one reference cost function evaluating every system's final
parameters, Ceres's self-reported initial cost ASSERTED against the
reference on every run, min-of-N interleaved timing, verified
single-core pinning.

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

arael runs the gain-ratio `NielsenLambdaDriver` here (the benchmark
default; `ARAEL_DRIVER=fixed` selects the classic fixed-multiplier
schedule). This benchmark is what drove the `LambdaDriver` abstraction
into arael (see docs/SOLVERS.md): under the fixed schedule no
(lambda0, floor) pair fits all sizes -- on Ladybug-138 the default
floor let lambda decay until the gauge-singular damped system lost
positive definiteness (Cholesky failures, a catastrophic +23000-cost
rejected step, a give-up 2.8% above the optimum), while the 1e-4 floor
that fixed it forced 54 crawling steps and pushed Ladybug-49's f32
into the iteration cap. The Nielsen driver dissolves all of it: zero
rejections on Ladybug-49 (22 accepted steps -- exactly Ceres's count),
Ladybug-138 converging in 20 steps to a cost marginally BELOW
Ceres's, and the damping floor rendered moot (measured identical at
1e-12 and 1e-6; the library default stays).

Initial lambda is per-dataset (harness table, env `ARAEL_LAMBDA0`):
1e-3 on Ladybug-138 (1e-4 plateaus 1.4% high there), 1e-4 elsewhere.
Ceres runs its shipped trust-region defaults -- BAL is the problem
family they were tuned on. g2o runs its auto lambda heuristic, which
scales the initial damping to each problem (`G2O_LAMBDA_INIT`
overrides). `ARAEL_VERBOSE=1` prints the solver's per-iteration trace.

## Results (Ladybug problems, aarch64 VM, single core, min of 5 rounds; arael rows use the Nielsen driver)

`iters` is accepted(attempts) for every system -- g2o's attempts come
from its batch statistics (it retries lambda inside an iteration, so
its bare iteration count used to hide them). `full-it ms` and `peak MB`
are defined under Running; peaks here are from the cholmod-gpl build,
whose BLAS/LAPACK stack adds a few MB of shared-library baseline to the
non-gpl arael rows (`BAL_MEM_EXE` removes it).

### Ladybug-49 (23769 parameters)

| system                   | total ms |  iters  | ms/iter | full-it ms | 1st-iter ms | peak MB | final cost |
|--------------------------|---------:|--------:|--------:|-----------:|------------:|--------:|-----------:|
| arael LM f64             |    542.0 |  22(22) |   24.63 |      23.36 |        52.9 |    74.8 | 26689.3493 |
| arael LM f64 cholmod-gpl |    620.5 |  22(22) |   28.20 |      26.85 |        55.0 |    76.5 | 26689.3493 |
| arael LM f32             |    498.4 |  20(24) |   20.77 |      20.07 |        46.9 |    64.2 | 26690.8537 |
| ceres dense_schur        |    445.0 |  22(22) |   20.23 |      18.10 |        43.0 |    38.4 | 26689.7174 |
| ceres sparse_schur       |    467.2 |  22(22) |   21.24 |      18.73 |        54.4 |    41.7 | 26689.7174 |
| ceres iterative_schur    |    675.6 |  24(24) |   28.15 |      26.03 |        38.9 |    37.0 | 26689.5841 |
| g2o LM (schur)           |    983.8 |  42(58) |   16.96 |      19.86 |        35.1 |    45.3 | 26714.5561 |

### Ladybug-138 (60876 parameters)

| system                   | total ms |  iters  | ms/iter | full-it ms | 1st-iter ms | peak MB |  final cost |
|--------------------------|---------:|--------:|--------:|-----------:|------------:|--------:|------------:|
| arael LM f64             |   1835.6 |  20(21) |   87.41 |      81.27 |       218.6 |   189.1 | 119055.9961 |
| arael LM f64 cholmod-gpl |   1974.9 |  20(21) |   94.04 |      88.89 |       201.1 |   198.4 | 119055.9961 |
| arael LM f32             |   1516.1 |  19(20) |   75.80 |      69.72 |       200.0 |   161.6 | 119054.9134 |
| ceres dense_schur        |   1893.5 |  22(25) |   75.74 |      72.20 |       148.7 |    98.0 | 119056.2145 |
| ceres sparse_schur       |   1744.8 |  22(25) |   69.79 |      65.29 |       178.7 |   105.7 | 119056.2145 |
| ceres iterative_schur    |   3284.3 |  43(62) |   52.97 |      54.97 |       115.3 |    84.4 | 190424.2644 |
| g2o LM (schur)           |   3481.4 |  40(59) |   59.01 |      68.49 |       127.5 |   121.0 | 118904.3429 |

`iterative_schur` does not reach the common optimum on Ladybug-138: its
inexact CG solve stalls at 190424 (60% above the 118904 optimum, camera
RMSE 1.38e-2). The other six validate; g2o's 118904 is the deepest.

### Ladybug-372 (145617 parameters)

| system                   | total ms |  iters  | ms/iter | full-it ms | 1st-iter ms | peak MB |  final cost |
|--------------------------|---------:|--------:|--------:|-----------:|------------:|--------:|------------:|
| arael LM f64             |   3475.2 |   5(11) |  315.92 |     293.50 |       652.5 |   463.5 | 225577.5976 |
| arael LM f64 cholmod-gpl |   3808.6 |   5(11) |  346.24 |     328.31 |       641.5 |   474.6 | 225577.5976 |
| arael LM f32             |   4054.4 |   8(17) |  238.49 |     226.77 |       575.4 |   381.4 | 225464.4215 |
| ceres dense_schur        |   7516.1 |  10(17) |  442.12 |     448.31 |       687.8 |   285.4 | 225447.1709 |
| ceres sparse_schur       |   4324.7 |  10(17) |  254.39 |     256.94 |       602.3 |   287.2 | 225447.1709 |
| ceres iterative_schur    |   3334.0 |  15(26) |  128.23 |     137.75 |       332.6 |   190.1 | 225798.8928 |
| g2o LM (schur)           |   9161.3 |  28(35) |  261.75 |     273.79 |       442.1 |   358.8 | 226586.4232 |

Every row validates on Ladybug-49 and 372. On Ladybug-138 six of seven
validate -- Ceres's `iterative_schur` is the exception (above). One
Ceres reference point measured outside the tables on Ladybug-49 (same
run protocol): `sparse_normal_cholesky`, the full-system strategy arael
uses, at ~39 ms/step (~1.6x arael's per-step). Fixed-schedule arael
numbers for comparison (ARAEL_DRIVER=fixed, per-dataset floors): 821 /
4968 / 3242 ms -- the Nielsen driver is worth ~1.5x on Ladybug-49,
~2.7x on Ladybug-138, and a lower-cost stop on Ladybug-372.

### Ladybug-1723 (485k parameters, exploratory: `BAL_ONLY=1723`)

With the shared 1e-5 tolerances NEITHER system converges -- they stop
at different plateaus of a long descent, so the mutual validation gate
cannot pass and the problem is excluded from the default suite pending
a tighter shared termination criterion. The exploratory numbers
(Nielsen): arael f64 84.4 s / 16(26) steps / 3246 ms/step reaching
cost 766181; Ceres sparse_schur 71.6 s / 14(24) / 2984 ms/step
stopping at 776854 -- 1.4% HIGHER than arael's plateau. arael f32
fails at this scale before the first step: the f32 ASSEMBLY produces a
NaN Hessian diagonal (single-precision range overflow accumulating
J^T J at 485k parameters), and the solve terminates loudly.

## Reading the results

With the Nielsen driver the size story flattens into near parity:

1. **Ladybug-49: Ceres wins by ~22%** (445 vs 542 ms), both taking 22
   rejection-free steps -- the remaining gap is per-step cost (18.1 vs
   23.4 ms/full-it), i.e. dense-Schur elimination at the problem size
   it is best at.
2. **Ladybug-138: arael f32 wins the dataset** -- 1516 ms, the fastest
   of any solver, with the lowest cost of the arael/Ceres cluster
   (119054.91). arael f64 (1836 ms) sits between the two Ceres variants
   (1745 / 1894). g2o alone descends deeper (118904, 0.13% below the
   cluster) but takes ~2x the time (3481 ms).
3. **Ladybug-372: Ceres's inexact-CG `iterative_schur` takes the
   dataset** (3.33 s, stopping at a slightly higher but still-converged
   225799), with arael faer f64 next (3.48 s); Ceres sparse-Schur takes
   4.32 s (10 accepted steps to arael's 5). Run-to-run step counts vary
   (5-12) here; min-of-rounds applies to every system equally.
4. **Per-full-iteration, arael's full-system faer solve tracks Ceres's
   sparse-Schur within 14-25% across the tables** (23.4/18.7 ->
   81.3/65.3 -> 293.5/256.9 ms full-it; exploratory 485k: 3246/2984
   ms/step), while Ceres running the same non-Schur strategy costs
   ~1.6x more than arael at the small end. On Ladybug's banded camera
   connectivity, explicit Schur elimination is NOT the decisive
   advantage it is reputed to be.
5. **arael's f32 solves bundle adjustment and validates on 49, 138 and
   372** (Ceres offers no f32 mode at all) and wins Ladybug-138. It
   fails at 485k parameters (NaN in f32 assembly, above). Its peak
   memory is only ~14-18% below f64's (64/74 -> 161/189 -> 381/463
   MB), not the halving the value size alone would suggest.
6. **g2o's full iteration is cheaper than arael's (19.9 / 68.5 / 273.8
   vs 23.4 / 81.3 / 293.5 ms, on its Schur-reduced camera system) yet
   it lands 1.8-2.7x behind on totals** because it needs roughly twice
   the accepted steps (42 / 40 / 28), plus internal lambda retries the
   bare iteration count used to hide (58 / 59 / 35 damped attempts).
   We have not established why. One confirmed code-level difference is
   the damping form: g2o adds a single lambda to every Hessian diagonal
   (plain Levenberg, H + lambda*I, block_solver.hpp), whereas arael
   scales by curvature (Marquardt, H + lambda*diag(H)) and Ceres does
   the equivalent through its default Jacobi column scaling. On BAL's
   badly scaled blocks -- camera focal/rotation/translation and point
   coordinates span orders of magnitude -- a single global lambda may
   possibly not damp all scales at once, so this could contribute to
   the extra iterations. That is a hypothesis only: it is not isolated
   by experiment, and g2o's initial-lambda heuristic, termination
   criteria, or step-acceptance trials could matter as much. The lambda
   schedule is probably not the cause -- g2o's LM carries gain-ratio
   fields (`_goodStepLowerScale` / `_ni`), so it appears to run a
   Nielsen-style schedule like arael's rather than a fixed-multiplier
   one -- but we have not read its update code to confirm. The flip
   side: on Ladybug-138 those extra iterations grind to the deepest
   stop of the run (118904, 0.13% below everyone else's).
7. **CHOLMOD supernodal does not pay off on BAL.** The cholmod-gpl row
   is SLOWER than faer at every size (26.9 vs 23.4, 88.9 vs 81.3, and
   328.3 vs 293.5 ms/full-it) -- in contrast to the SLAM benchmark,
   where the same backend is ~15% faster than faer at 300 poses. Both
   f64 rows take identical steps to the identical cost; only the linear
   solver differs. Consistent with CHOLMOD's own design: its AUTO mode
   picks supernodal only when the factor's flops/nnz(L) ratio is high
   (dense supernodes that reward BLAS panels), and
   `CholmodSupernodalLLT` forces it regardless. We have not measured
   the BAL factors' flop density to confirm that is the mechanism here.
8. **arael's full-system peak memory is the highest in the field at
   the larger sizes** (189 MB on 138, 463 MB on 372 -- 1.6x Ceres
   sparse-Schur's 287, 2.4x iterative_schur's 190). Factorizing the
   full camera+point system buys per-iteration speed but not memory;
   the Schur solvers work on far smaller reduced systems.

## Running

```sh
./fetch_datasets.sh
cmake -B cpp/build cpp && cmake --build cpp/build
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # thread caps before load
ROUNDS=5 cargo run --release
```

`--features cholmod-gpl` adds an `arael LM f64 cholmod-gpl` row solving
the same full system with CHOLMOD's supernodal backend. That module is
GPL-licensed (unlike the LGPL simplicial one), so the resulting binary
is subject to the GPL -- enable knowingly.

The `peak MB` column is each solver's peak RSS (VmHWM): arael rows from
a fresh subprocess running only that solver (`BAL_NO_MEM=1` skips the
re-solves), external rows self-reported. A cholmod-gpl build inflates
the plain arael rows with its BLAS/LAPACK shared-library baseline;
point `BAL_MEM_EXE` at a default-build binary to source them from a
clean build instead (the cholmod-gpl row always self-measures).

The `full-it ms` column is the cost of one FULL accepted iteration
(linearize + factorize/solve + trial cost). `ms/iter` divides total
time by all damped attempts, and rejected attempts skip the
re-linearization, so heavy damping deflates it; `full-it ms` is immune
to that. Per system: arael sums the solver's steady-state per-phase
means (assembly, damped solve, trial cost, advance; first calls
excluded -- they carry one-time structure costs like the symbolic
factorization), Ceres sums its Summary's per-phase totals over call
counts (jacobian, linear solve, residual; firsts amortized in), g2o
averages whole-iteration times over iterations that ran exactly one
lambda trial, first iteration excluded. `-` when a system had too few
clean iterations to measure.
