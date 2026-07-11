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
are defined under Running. The two arael routes are the full-system
sparse solve (`sparse`) and the Schur-complement solve (`schur`, points
marginalized every damped solve, only the camera system factorized) --
they reach the same optimum by construction, so the rows compare linear
solvers directly.

### Ladybug-49 (23769 parameters)

| system                   | total ms |  iters  | ms/iter | full-it ms | 1st-iter ms | peak MB | final cost |
|--------------------------|---------:|--------:|--------:|-----------:|------------:|--------:|-----------:|
| arael LM f64 sparse      |    536.3 |  21(22) |   24.38 |      23.48 |        45.9 |    70.9 | 26689.4804 |
| arael LM f32 sparse      |    470.5 |  18(23) |   20.46 |      20.06 |        41.9 |    63.0 | 26692.4865 |
| arael LM f64 schur       |    249.4 |  21(22) |   11.33 |      11.19 |        18.0 |    37.6 | 26689.4804 |
| arael LM f32 schur       |    172.4 |  16(20) |    8.62 |       8.90 |        13.6 |    28.1 | 26693.2360 |
| ceres dense_schur        |    443.7 |  22(22) |   20.17 |      18.02 |        42.6 |    38.4 | 26689.7174 |
| ceres sparse_schur       |    462.8 |  22(22) |   21.04 |      18.52 |        54.2 |    41.6 | 26689.7174 |
| ceres iterative_schur    |    670.2 |  24(24) |   27.93 |      25.81 |        38.6 |    37.0 | 26689.5841 |
| g2o LM (schur)           |    983.5 |  42(58) |   16.96 |      19.86 |        35.8 |    45.3 | 26714.5561 |

### Ladybug-138 (60876 parameters)

| system                   | total ms |  iters  | ms/iter | full-it ms | 1st-iter ms | peak MB |  final cost |
|--------------------------|---------:|--------:|--------:|-----------:|------------:|--------:|------------:|
| arael LM f64 sparse      |   1846.6 |  20(21) |   87.93 |      82.00 |       206.2 |   183.3 | 119055.9961 |
| arael LM f32 sparse      |   1512.1 |  19(20) |   75.61 |      69.73 |       187.8 |   161.4 | 119054.9134 |
| arael LM f64 schur       |   1038.4 |  20(21) |   49.45 |      48.63 |        71.1 |   112.2 | 119055.9961 |
| arael LM f32 schur       |    718.2 |  20(21) |   34.20 |      33.56 |        49.5 |    80.1 | 119055.5616 |
| ceres dense_schur        |   1901.0 |  22(25) |   76.04 |      72.43 |       150.5 |    98.0 | 119056.2145 |
| ceres sparse_schur       |   1741.8 |  22(25) |   69.67 |      65.07 |       180.5 |   105.6 | 119056.2145 |
| ceres iterative_schur    |   3274.1 |  43(62) |   52.81 |      54.71 |       116.8 |    84.4 | 190424.2644 |
| g2o LM (schur)           |   3495.5 |  40(59) |   59.25 |      68.97 |       130.1 |   120.9 | 118904.3429 |

`iterative_schur` does not reach the common optimum on Ladybug-138: its
inexact CG solve stalls at 190424 (60% above the 118904 optimum, camera
RMSE 1.38e-2). The other six validate; g2o's 118904 is the deepest.

### Ladybug-372 (145617 parameters)

| system                   | total ms |  iters  | ms/iter | full-it ms | 1st-iter ms | peak MB |  final cost |
|--------------------------|---------:|--------:|--------:|-----------:|------------:|--------:|------------:|
| arael LM f64 sparse      |   3462.8 |   5(11) |  314.80 |     294.78 |       627.9 |   473.7 | 225577.5976 |
| arael LM f32 sparse      |   4022.5 |   8(17) |  236.62 |     227.02 |       535.6 |   392.2 | 225464.4215 |
| arael LM f64 schur       |   3520.8 |   5(11) |  320.07 |     322.48 |       411.3 |   399.1 | 225577.5976 |
| arael LM f32 schur       |   5216.9 |  16(27) |  193.22 |     197.54 |       267.0 |   270.2 | 225629.5986 |
| ceres dense_schur        |   7514.3 |  10(17) |  442.02 |     447.77 |       692.0 |   285.4 | 225447.1709 |
| ceres sparse_schur       |   4307.5 |  10(17) |  253.38 |     255.63 |       600.4 |   287.2 | 225447.1709 |
| ceres iterative_schur    |   3329.9 |  15(26) |  128.07 |     137.15 |       331.7 |   190.0 | 225798.8928 |
| g2o LM (schur)           |   9179.6 |  28(35) |  262.27 |     274.54 |       445.9 |   358.8 | 226586.4232 |

Every row validates on Ladybug-49 and 372 (8/8). On Ladybug-138 seven
of eight validate -- Ceres's `iterative_schur` is the exception
(above). One Ceres reference point measured outside the tables on
Ladybug-49 (same run protocol): `sparse_normal_cholesky`, the
full-system strategy arael's `sparse` rows use, at ~39 ms/step (~1.6x
their per-step). Fixed-schedule arael
numbers for comparison (ARAEL_DRIVER=fixed, per-dataset floors): 821 /
4968 / 3242 ms -- the Nielsen driver is worth ~1.5x on Ladybug-49,
~2.7x on Ladybug-138, and a lower-cost stop on Ladybug-372.

### Ladybug-1723 (485k parameters, exploratory: `BAL_ONLY=1723`)

With the shared 1e-5 tolerances NEITHER system converges -- they stop
at different plateaus of a long descent, so the mutual validation gate
cannot pass and the problem is excluded from the default suite pending
a tighter shared termination criterion. The exploratory numbers
(Nielsen): arael f64 sparse 84.4 s / 16(26) steps / 3246 ms/step
reaching cost 766181; Ceres sparse_schur 71.6 s / 14(24) / 2984 ms/step
stopping at 776854 -- 1.4% HIGHER than arael's plateau. arael f32
fails at this scale before the first step: the f32 ASSEMBLY produces a
NaN Hessian diagonal (single-precision range overflow accumulating
J^T J at 485k parameters), and the solve terminates loudly.

The Schur route is the WRONG choice at this size and shows it: 5306
ms/step (1.6x the full-system route) and a 1.9 GB peak, reaching the
identical cost in the identical 16(26) steps. The reduced camera system
is 15,507 parameters and only 8.1% dense, but faer's AMD ordering
leaves a factor of ~83M values -- 69% of a dense triangle -- costing
4755 ms to factorize on its own. Ceres factorizes the same S in ~2.5 s
(its whole linear solve is 2586 ms, from its own stats), so the gap is
fill, not kernel speed: the two factorization kernels measure at parity
elsewhere (see benchmarks/slam). For reference, Ceres's DENSE Schur
does degrade here exactly as the arithmetic demands -- 28.3 s per
linear solve -- so nobody escapes a dense S; the question is only how
much fill the ordering leaves.

## Reading the results

Which arael route to use is the whole story, and it turns on the size
of the camera system:

1. **Ladybug-49 goes to arael's Schur route, decisively** -- f32 schur
   at 172 ms and f64 schur at 249 ms, against 444 ms for the closest
   C++ system (Ceres dense-Schur). Every exact solver takes the same
   21-22 rejection-free steps to the same optimum, so the 1.8x is
   entirely per-step cost: 11.2 ms/full-it against Ceres's 18.0 and
   g2o's 19.9.
2. **Ladybug-138 likewise: arael f32 schur wins outright** at 718 ms --
   2.4x faster than the next system (Ceres sparse-Schur, 1742 ms) -- and
   f64 schur (1038 ms) beats every C++ solver too. g2o alone descends
   deeper (118904, 0.13% below the cluster) but takes 3.5 s to do it.
3. **Ladybug-372 goes to Ceres's inexact-CG `iterative_schur`** (3.33 s,
   128 ms/iter, stopping at a slightly higher but still-converged
   225799). The exact solvers cluster behind it: arael f64 sparse
   3.46 s, arael f64 schur 3.52 s, Ceres sparse-Schur 4.31 s. arael f32
   schur has the cheapest exact step of that group (193 ms/iter) but
   spends it on 16 accepted steps instead of 5, so its total lands last.
   Run-to-run step counts vary (5-16) here; min-of-rounds applies to
   every system equally.
4. **Per full iteration, arael's Schur step is the cheapest exact step
   in the field at 49 and 138** (11.2 and 48.6 ms/full-it, against Ceres
   sparse-Schur's 18.5 and 65.1 and g2o's 19.9 and 69.0), while its
   full-system step tracks Ceres's sparse-Schur within 15-27% (23.5/18.5
   -> 82.0/65.1 -> 294.8/255.6 ms). This CORRECTS an earlier reading of
   this benchmark, taken before arael had an explicit Schur backend:
   marginalizing the points is decisive after all -- but only while the
   camera system stays small (finding 7).
5. **arael's f32 solves bundle adjustment and validates on 49, 138 and
   372** (Ceres offers no f32 mode at all) and wins both smaller
   datasets. It fails at 485k parameters (NaN in f32 assembly, above).
   Its peak memory saving over f64 is real but modest -- 11-17% on the
   full-system rows (63/71 -> 161/183 -> 392/474 MB), 25-32% on the
   Schur rows (28/38 -> 80/112 -> 270/399) -- not the halving the value
   size alone would suggest.
6. **g2o's full iteration (19.9 / 69.0 / 274.5 ms, on its Schur-reduced
   camera system) is cheaper than arael's full-system step but no longer
   cheaper than arael's Schur step** (11.2 / 48.6 / 322.5), **and it
   lands 1.8-2.7x behind arael's full-system totals and 2.6-3.9x behind
   its Schur totals** because it needs roughly twice
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
7. **The Schur route wins big while the camera system stays small, and
   stops winning as it grows.** Marginalizing the points costs 2.1x
   less per step than factorizing the full system on Ladybug-49 (11.3
   vs 24.4 ms/iter, f32 8.6 vs 20.5) and 1.8x less on Ladybug-138
   (49.5 vs 87.9), with the same steps to the same cost -- but on
   Ladybug-372 it is a wash for f64 (320.1 vs 314.8). The reduced
   system S is camera-camera: 441 parameters at 49 cameras but 3,348 at
   372, and factorizing it costs roughly cubically in that size
   (measured: 0.9 / 14.3 / 228.4 ms across the three datasets --
   `cargo run -r --bin schur_stats` reports S's size, density and cost
   split per dataset). Beyond the suite the trend continues: on the
   exploratory Ladybug-1723 the Schur route is 1.6x SLOWER (5306 vs
   3246 ms/iter) because S reaches 15,507 parameters. This is why the
   Schur backend is an explicit opt-in per model, never an automatic
   choice.
8. **Memory follows the same shape.** The Schur route roughly halves
   arael's peak on the sizes where it wins (37.6 vs 70.9 MB on
   Ladybug-49, 112 vs 183 on Ladybug-138), because it never forms the
   full camera+point factor. The full-system rows remain the heaviest
   in the field at the larger sizes (183 MB on 138, 474 on 372 -- 1.6x
   Ceres sparse-Schur's 287, 2.5x iterative_schur's 190).

## Running

```sh
./fetch_datasets.sh
cmake -B cpp/build cpp && cmake --build cpp/build
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # thread caps before load
ROUNDS=5 cargo run --release
```

`--features cholmod-gpl` adds an `arael LM f64 cholmod-gpl` row solving
the full system with CHOLMOD's supernodal backend (it is not in the
tables above, which come from a default build). That module is
GPL-licensed, so the resulting binary is subject to the GPL -- enable
knowingly. Measured earlier on this benchmark, it is SLOWER than faer
at every size, so it is not the row to reach for here.

The `peak MB` column is each solver's peak RSS (VmHWM): arael rows from
a fresh subprocess running only that solver (`BAL_NO_MEM=1` skips the
re-solves), external rows self-reported. A cholmod-gpl build links a
BLAS/LAPACK stack that inflates the arael rows with its shared-library
baseline; point `BAL_MEM_EXE` at a default-build binary to source them
from a clean build instead.

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
