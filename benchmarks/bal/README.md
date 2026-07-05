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

Bundle adjustment is far less linear than the pose graphs, so the
pose-graph lambda values do not transfer: arael runs
`initial_lambda = 1e-4` (probed best together with 1e-5; larger values
add iterations), env `ARAEL_LAMBDA0`. Ceres runs its shipped
trust-region defaults -- BAL is the problem family they were tuned on.

The damping FLOOR is per-dataset (harness table; `ARAEL_LAMBDA_FLOOR`
overrides), and the probes behind it are the sharpest schedule evidence
in the repo: on Ladybug-138 the library-default floor (1e-12) lets
lambda decay until the gauge-singular damped system loses positive
definiteness -- Cholesky failures, a catastrophic +23000-cost rejected
step, and a give-up 2.8% above the optimum -- while a 1e-4 floor
converges everywhere but makes Ladybug-49's f32 crawl into the
iteration cap. No fixed (lambda0, floor) pair fits all sizes; an
adaptive schedule (gain-ratio style, or a learned controller) is the
single biggest lever on this benchmark. `ARAEL_VERBOSE=1` prints the
solver's per-iteration trace to watch it happen.

## Results (Ladybug problems, aarch64 VM, single core, min of rounds)

### Ladybug-49 (23769 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|--------------------|---------:|--------:|--------:|------------:|-----------:|
| arael LM f64       |    821.0 |  23(35) |   23.46 |        53.1 | 26688.7584 |
| arael LM f32       |    695.3 |  22(35) |   19.86 |        47.1 | 26691.7649 |
| ceres dense_schur  |    452.7 |  22(22) |   20.58 |        43.8 | 26689.7174 |
| ceres sparse_schur |    469.5 |  22(22) |   21.34 |        54.7 | 26689.7174 |

### Ladybug-138 (60876 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms | final cost |
|--------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64       |   4968.3 |  54(59) |   84.21 |       216.8 | 119271.7865 |
| arael LM f32       |   4587.0 |  53(63) |   72.81 |       206.8 | 119130.2961 |
| ceres dense_schur  |   1944.3 |  22(25) |   77.77 |       152.2 | 119056.2145 |
| ceres sparse_schur |   1787.6 |  22(25) |   71.50 |       186.2 | 119056.2145 |

### Ladybug-372 (145617 parameters)

| system             | total ms |  iters  | ms/iter | 1st-iter ms |  final cost |
|--------------------|---------:|--------:|--------:|------------:|------------:|
| arael LM f64       |   3242.0 |   6(10) |  324.20 |       666.6 | 225480.2326 |
| arael LM f32       |   4575.5 |   9(19) |  240.82 |       577.8 | 225579.5838 |
| ceres dense_schur  |   7652.9 |  10(17) |  450.17 |       720.8 | 225447.1709 |
| ceres sparse_schur |   4479.1 |  10(17) |  263.47 |       614.5 | 225447.1709 |

All rows validate on all three. Two Ceres reference points measured
outside the tables on Ladybug-49 (its other linear solvers, same run
protocol): `sparse_normal_cholesky` (the full-system strategy arael
uses) 853 ms at 38.8 ms/step, `iterative_schur` 681 ms.

### Ladybug-1723 (485k parameters, exploratory: `BAL_ONLY=1723`)

With the shared 1e-5 tolerances NEITHER system converges -- they stop
at plateaus 1.9% apart, so the mutual validation gate cannot pass and
the problem is excluded from the default suite pending a tighter
shared termination criterion. The exploratory numbers: arael f64
88.4 s / 16(29) steps / 3047 ms/step reaching cost 762281; Ceres
sparse_schur 70.2 s / 14(24) / 2925 ms/step stopping at 776854 (1.9%
HIGHER). arael f32 fails outright at this scale (the damped f32 system
loses positive definiteness immediately -- the same failure mode as
factrs's f32 on city10000).

## Reading the results

The story inverts with size:

1. **Ladybug-49/138: Ceres wins on total time (1.8x / 2.8x), and the
   entire gap is the damping schedule.** Ceres's gain-ratio trust
   region takes 22 rejection-free steps on both; arael burns 12
   rejected attempts on 49 and needs 54 crawling floor-damped steps on
   138 (see Damping above). Per-step costs are near parity throughout
   (23.5 vs 20.6 on 49; 84 vs 72 on 138).
2. **Ladybug-372: arael wins on total time (3.24 vs 4.40 s)** -- it
   reaches the optimum in 6 accepted steps to Ceres's 10(17).
3. **Per-step, arael's full-system faer solve tracks Ceres's
   sparse-Schur within 4-26% across the whole range** (23.5/21.3 ->
   84/72 -> 324/259 -> 3047/2925 ms at 485k parameters), while Ceres
   running the same non-Schur strategy costs 1.65x more than arael at
   the small end. On Ladybug's banded camera connectivity, explicit
   Schur elimination is NOT the decisive advantage it is reputed to
   be; the schedule is.
4. **arael's f32 solves bundle adjustment and validates on 49, 138 and
   372** (Ceres offers no f32 mode at all), and on 372 its per-step
   cost is the cheapest measured (241 ms). It dies at 485k parameters
   (positive-definiteness loss).

## Running

```sh
./fetch_datasets.sh
cmake -B cpp/build cpp && cmake --build cpp/build
ROUNDS=5 cargo run --release
```
