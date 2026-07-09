# SO(3) parameterization conditioning benchmark

How arael's three rotation parameterizations
(`SimpleEulerAngleParam`, `EulerAngleParam`, `QuaternionParam`) condition a
hard nonlinear solve. The companion [benchmarks/slam](../slam/README.md#rotation-parameterization-simple-vs-euler-vs-quaternion)
measures their **per-iteration cost** on a well-initialized problem (where all
three converge in 3 steps); this one measures the opposite regime -- **how many
iterations** each needs when the rotations are large and the initialization is
bad.

## Problem

A pose chain flies nine 360-degree barrel rolls, an Immelmann half-loop (pitch
through the gimbal at 90 and on to 180), a half-roll back to upright, and three
climbing barrel rolls -- a corkscrew that sweeps orientation space and crosses
the euler gimbal repeatedly. Consecutive poses are tied by a relative-rotation
constraint (the 9 elements of `R_prev^T R_cur - delta`, whitened). The first
pose is fixed at identity and **every free pose starts at identity**, maximally
far from the flown trajectory, so the solver has to rotate them across many
gimbal crossings and accumulate thousands of degrees of rotation. Only the
pose rotation parameter differs between the three variants; the constraint,
scene, and solver config (`max_iters = 500`, default fixed-lambda schedule) are
identical.

This is the same maneuver as the `aerobatics_slam_barrel_roll_and_immelmann`
test in `tests/euler_param.rs`, and the three-way convergence is pinned by
`tests/rot_param_compare.rs`.

## Results (Apple M4 Pro, median of 20 rounds)

313 poses, 312 relative-rotation constraints.

| parameterization        | iters | accepted | total ms | ms/iter | final cost |
|-------------------------|------:|---------:|---------:|--------:|-----------:|
| `SimpleEulerAngleParam` |   285 |      171 |     23.8 |   0.083 |    2.2e-26 |
| `EulerAngleParam`       |    81 |       51 |      9.4 |   0.116 |    6.4e-26 |
| `QuaternionParam`       |    52 |       32 |      6.1 |   0.117 |    1.5e-26 |

All three reach the same trajectory (final cost at the f64 floor, per-pose
orientation recovered to better than 1e-4). What differs is the iteration
count, and with it the total time:

- **`QuaternionParam` wins on both** -- 52 iterations, 6.1 ms. Its
  rotation-vector (exp-map) delta is isotropic (equal conditioning in every
  direction) and the exponential map is exact, so LM's quadratic model stays
  accurate step to step.
- **`EulerAngleParam`** takes 81 iterations / 9.4 ms. Re-centering keeps its
  delta near zero, dodging the gimbal, but the euler-angle delta is still
  anisotropic, so the model is a little worse and it needs more steps.
- **`SimpleEulerAngleParam`** takes 285 iterations / 23.8 ms -- 3.9x slower than
  quaternion despite being the **cheapest per iteration** (0.083 ms/iter, no
  re-centering, division-free rotation). It optimizes the angles directly, so
  near every pitch = 90 passage the Jacobian degenerates (gimbal lock); 114 of
  its 285 steps are rejected as LM inflates lambda to fight the ill-conditioning.

So the per-iteration ranking (simple cheapest) inverts on total time (simple
slowest): cheap steps are false economy when you need 5.5x as many of them.

## When this matters (and when it does not)

**This 5.5x gap is a worst case, not a typical solve.** It is manufactured by
an adversarial setup: every pose initialized at identity, maximally far from a
trajectory that accumulates thousands of degrees, solved as one batch, with the
path deliberately crossing the gimbal over and over. A properly engineered
estimator never hands the solver a problem shaped like this:

- **Good initialization.** A real front-end -- wheel / visual / inertial
  odometry, a motion model, the previous frame -- seeds each pose close to its
  answer, so the increment the solver refines is a small, near-identity
  rotation, nowhere near a gimbal.
- **Incremental / windowed estimation.** Fixed-lag smoothing, sliding windows,
  or incremental factor graphs keep each solve's update small; you do not
  re-solve a 300-pose aerobatic trajectory from scratch from identity.
- **Small rotations linearize identically.** For increments away from the
  gimbal, all three parameterizations agree to first order (the retraction and
  the exp map differ only at O(delta^3)), so they converge in the same handful
  of steps.

Under those normal conditions the parameterizations are interchangeable on
conditioning, so the choice comes down to cost. Each pose precomputes its
rotation Jacobian once per update, so how much the exact parameterizations cost
depends on how many constraints amortize it: in a bearing-dense problem the
[slam benchmark](../slam/README.md#rotation-parameterization-simple-vs-euler-vs-quaternion)
spreads it over ~90 bearings per pose and all three land within ~3% on assembly
and ~0.3% on total solve time; in a sparse pose graph like this one (~1
constraint per pose) it barely amortizes, so the table above keeps a per-iteration
edge for the naive variant. Either way this benchmark measures the other axis --
robustness when the initialization is bad and the rotations are large -- and
there the exp-map delta's isotropy earns its keep.

The honest summary: pick the parameterization for the regime you actually
operate in. Well-initialized and incremental (almost always) -> any of them;
per-iteration cost is close (a wash once bearings amortize the precompute), so
favor simplicity with the naive `SimpleEulerAngleParam`. Large uncontrolled
rotations from a cold start (rare, and usually a sign the front-end should be
fixed instead) -> the quaternion, whose far lower iteration count dominates
total time.

## Running

```sh
cargo run --release            # median of 20 rounds
ROUNDS=50 cargo run --release
```
