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

## Results (Ladybug problems, aarch64 VM, single core, min of 10 rounds; arael rows use the Nielsen driver)

### Ladybug-49 (23769 parameters)

| system                | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|-----------------------|---------:|--------:|--------:|------------:|-----------:|
| arael LM f64          |    533.8 |  22(22) |   24.26 |        53.2 | 26689.3493 |
| arael LM f32          |    478.3 |  20(23) |   20.80 |        46.2 | 26690.5913 |
| ceres dense_schur     |    436.3 |  22(22) |   19.83 |        41.8 | 26689.7174 |
| ceres sparse_schur    |    456.0 |  22(22) |   20.73 |        53.0 | 26689.7174 |
| ceres iterative_schur |    662.6 |  24(24) |   27.61 |        38.2 | 26689.5841 |
| g2o LM (schur)        |    972.8 |      42 |   23.16 |        34.3 | 26714.5561 |

### Ladybug-138 (60876 parameters)

| system                | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|-----------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64          |   1834.8 |  20(21) |   87.37 |       214.8 | 119055.9961 |
| arael LM f32          |   1720.4 |  21(23) |   74.80 |       194.8 | 119053.6517 |
| ceres dense_schur     |   1874.8 |  22(25) |   74.99 |       145.6 | 119056.2145 |
| ceres sparse_schur    |   1734.1 |  22(25) |   69.36 |       174.8 | 119056.2145 |
| ceres iterative_schur |   3275.9 |  43(62) |   52.84 |       112.5 | 190424.2644 |
| g2o LM (schur)        |   3477.0 |      40 |   86.93 |       127.3 | 118904.3429 |

`iterative_schur` does not reach the common optimum on Ladybug-138: its
inexact CG solve stalls at 190424 (60% above the 118904 optimum, camera
RMSE 1.38e-2). The other five validate; g2o's 118904 is the deepest.

### Ladybug-372 (145617 parameters)

| system                | total ms |  iters  | ms/iter | 1st-iter ms |  final cost |
|-----------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64          |   3474.2 |   5(11) |  315.83 |       658.3 | 225577.5976 |
| arael LM f32          |   3622.9 |   8(15) |  241.53 |       572.0 | 225465.6238 |
| ceres dense_schur     |   7481.2 |  10(17) |  440.07 |       680.5 | 225447.1709 |
| ceres sparse_schur    |   4283.0 |  10(17) |  251.94 |       597.7 | 225447.1709 |
| ceres iterative_schur |   3326.6 |  15(26) |  127.94 |       327.5 | 225798.8928 |
| g2o LM (schur)        |   9071.4 |      28 |  323.98 |       438.1 | 226586.4232 |

Every row validates on Ladybug-49 and 372. On Ladybug-138 five of six
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

1. **Ladybug-49: Ceres wins by ~22%** (436 vs 534 ms), both taking 22
   rejection-free steps -- the remaining gap is per-step cost (19.8 vs
   24.3 ms), i.e. dense-Schur elimination at the problem size it is
   best at.
2. **Ladybug-138: arael f32 wins the dataset** -- 1720 ms, the fastest
   of any solver, with the lowest cost of the arael/Ceres cluster
   (119053.65). arael f64 (1835 ms) sits between the two Ceres variants
   (1734 / 1875). g2o alone descends deeper (118904, 0.13% below the
   cluster) but takes ~2x the time (3477 ms).
3. **Ladybug-372: arael is the fastest direct solver** (3.47 vs Ceres
   sparse-Schur's 4.28 s, 5 accepted steps to Ceres's 10). Ceres's
   `iterative_schur` edges it on wall-clock (3.33 s) with inexact CG,
   stopping at a slightly higher but still-converged cost (225799 vs
   225578). Run-to-run step counts vary (5-12) here; min-of-rounds
   applies to every system equally.
4. **Per-step, arael's full-system faer solve tracks Ceres's
   sparse-Schur within 9-26% across the whole range** (24/21 -> 87/69
   -> 316/252 -> 3246/2984 ms at 485k parameters), while Ceres running
   the same non-Schur strategy costs ~1.6x more than arael at the
   small end. On Ladybug's banded camera connectivity, explicit Schur
   elimination is NOT the decisive advantage it is reputed to be.
5. **arael's f32 solves bundle adjustment and validates on 49, 138 and
   372** (Ceres offers no f32 mode at all) and wins Ladybug-138. It
   fails at 485k parameters (NaN in f32 assembly, above).
6. **g2o validates everywhere at per-step parity but 1.8-2.6x behind
   on totals** (23.2 / 86.9 / 324 ms per step vs arael's 24.3 / 87.4 /
   316) because it needs roughly twice the iterations (42 / 40 / 28).
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

## Running

```sh
./fetch_datasets.sh
cmake -B cpp/build cpp && cmake --build cpp/build
ROUNDS=5 cargo run --release
```
