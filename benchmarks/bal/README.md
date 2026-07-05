# Bundle-adjustment benchmark: arael vs Ceres (BAL)

Bundle adjustment on the [BAL](https://grail.cs.washington.edu/projects/bal/)
(Bundle Adjustment in the Large) datasets -- the standard BA solver
benchmark -- comparing arael against [Ceres](http://ceres-solver.org)
on Ceres's home turf: BAL is the problem family Ceres's examples,
defaults, and Schur solvers were built around.

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
gauge) camera-center RMSE is under 1e-3 of the scene extent. Landmark
positions are deliberately NOT the geometric gate: BAL scenes contain
weakly-observed points (two rays at tiny parallax) that slide along
their rays almost cost-free -- converged solutions with costs within
0.004% of each other differ by 3-8% in all-points RMSE. Camera centers
are constrained by hundreds of residuals each.

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
family they were tuned on. `ARAEL_VERBOSE=1` prints the solver's
per-iteration trace.

## Results (Ladybug problems, aarch64 VM, single core, min of 3 rounds; arael rows use the Nielsen driver)

### Ladybug-49 (23769 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|--------------------|---------:|--------:|--------:|------------:|-----------:|
| arael LM f64       |    593.8 |  22(22) |   26.99 |        58.1 | 26689.3493 |
| arael LM f32       |    531.0 |  20(23) |   23.09 |        52.4 | 26690.5913 |
| ceres dense_schur  |    485.3 |  22(22) |   22.06 |        47.2 | 26689.7174 |
| ceres sparse_schur |    506.8 |  22(22) |   23.04 |        59.3 | 26689.7174 |

### Ladybug-138 (60876 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|--------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64       |   1950.8 |  20(21) |   92.89 |       249.3 | 119055.9961 |
| arael LM f32       |   1893.5 |  21(23) |   82.33 |       221.7 | 119053.6517 |
| ceres dense_schur  |   2070.5 |  22(25) |   82.82 |       169.6 | 119056.2145 |
| ceres sparse_schur |   1932.7 |  22(25) |   77.31 |       206.1 | 119056.2145 |

### Ladybug-372 (145617 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms |  final cost |
|--------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64       |   3618.2 |   5(11) |  328.93 |       698.3 | 225577.5976 |
| arael LM f32       |   3919.3 |   8(15) |  261.28 |       629.6 | 225465.6238 |
| ceres dense_schur  |   8113.6 |  10(17) |  477.27 |       787.6 | 225447.1709 |
| ceres sparse_schur |   4567.6 |  10(17) |  268.68 |       656.1 | 225447.1709 |

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

## Running

```sh
./fetch_datasets.sh
cmake -B cpp/build cpp && cmake --build cpp/build
ROUNDS=5 cargo run --release
```
