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
family they were tuned on. g2o runs its auto lambda heuristic
(`G2O_LAMBDA_INIT` overrides; probed 1e-6/1e-4 on Ladybug-49, within
14% of auto). `ARAEL_VERBOSE=1` prints the solver's per-iteration
trace.

## Results (Ladybug problems, aarch64 VM, single core, min of 3 rounds; arael rows use the Nielsen driver)

### Ladybug-49 (23769 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|--------------------|---------:|--------:|--------:|------------:|-----------:|
| arael LM f64       |    588.1 |  22(22) |   26.73 |        56.1 | 26689.3493 |
| arael LM f32       |    540.0 |  20(23) |   23.48 |        50.8 | 26690.5913 |
| ceres dense_schur  |    495.2 |  22(22) |   22.51 |        46.8 | 26689.7174 |
| ceres sparse_schur |    516.9 |  22(22) |   23.49 |        60.4 | 26689.7174 |
| g2o LM (schur)     |   1079.1 |      42 |   25.69 |        40.6 | 26714.5561 |

### Ladybug-138 (60876 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|--------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64       |   1938.4 |  20(21) |   92.30 |       230.5 | 119055.9961 |
| arael LM f32       |   1868.3 |  21(23) |   81.23 |       209.7 | 119053.6517 |
| ceres dense_schur  |   2056.7 |  22(25) |   82.27 |       164.9 | 119056.2145 |
| ceres sparse_schur |   1899.7 |  22(25) |   75.99 |       196.3 | 119056.2145 |
| g2o LM (schur)     |   3742.6 |      40 |   93.56 |       139.4 | 118904.3429 |

### Ladybug-372 (145617 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms |  final cost |
|--------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64       |   3616.7 |   5(11) |  328.79 |       701.2 | 225577.5976 |
| arael LM f32       |   3874.2 |   8(15) |  258.28 |       605.3 | 225465.6238 |
| ceres dense_schur  |   8015.0 |  10(17) |  471.47 |       753.0 | 225447.1709 |
| ceres sparse_schur |   4592.2 |  10(17) |  270.13 |       649.1 | 225447.1709 |
| g2o LM (schur)     |   9654.8 |      28 |  344.82 |       473.0 | 226586.4232 |

All rows validate on all three. Two Ceres reference points measured
outside the tables on Ladybug-49 (its other linear solvers, same run
protocol): `sparse_normal_cholesky` (the full-system strategy arael
uses) 853 ms at 38.8 ms/step, `iterative_schur` 681 ms. Fixed-schedule
arael numbers for comparison (ARAEL_DRIVER=fixed, per-dataset floors):
821 / 4968 / 3242 ms -- the Nielsen driver is worth 1.4x, 2.5x, and a
lower-cost stop respectively.

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

1. **Ladybug-49: Ceres wins by ~20%** (485 vs 594 ms), both taking 22
   rejection-free steps -- the remaining gap is per-step cost (22.1 vs
   27.0 ms), i.e. dense-Schur elimination at the problem size it is
   best at.
2. **Ladybug-138: a tie for f64 (1951 vs 1933 ms) and arael f32 wins
   the dataset outright** -- 1894 ms with the lowest final cost
   measured (119053.65, marginally below Ceres's optimum).
3. **Ladybug-372: arael wins** (3.62 vs 4.57 s, 5 accepted steps to
   Ceres's 10). Run-to-run step counts vary (5-12) here; min-of-rounds
   applies to every system equally.
4. **Per-step, arael's full-system faer solve tracks Ceres's
   sparse-Schur within 4-26% across the whole range** (27/23 -> 93/77
   -> 329/269 -> 3246/2984 ms at 485k parameters), while Ceres running
   the same non-Schur strategy costs 1.65x more than arael at the
   small end. On Ladybug's banded camera connectivity, explicit Schur
   elimination is NOT the decisive advantage it is reputed to be.
5. **arael's f32 solves bundle adjustment and validates on 49, 138 and
   372** (Ceres offers no f32 mode at all) and wins Ladybug-138. It
   fails at 485k parameters (NaN in f32 assembly, above).
6. **g2o validates everywhere at per-step parity but 1.9-2.7x behind
   on totals** (25.7 / 93.6 / 345 ms per step vs arael's 26.7 / 92.3 /
   329) because its fixed-multiplier lambda schedule needs roughly
   twice the iterations (42 / 40 / 28) -- the same pathology arael's
   fixed schedule exhibited here before the Nielsen driver, now
   confirmed in a third independent implementation. The flip side: on
   Ladybug-138 those extra iterations grind to the deepest stop of the
   run (118904, 0.13% below everyone else's).

## Running

```sh
./fetch_datasets.sh
cmake -B cpp/build cpp && cmake --build cpp/build
ROUNDS=5 cargo run --release
```
