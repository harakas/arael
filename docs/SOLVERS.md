# Solvers

arael ships a Levenberg-Marquardt solver with multiple linear-algebra
backends sharing one trait (`LmSolver`) and one config struct
(`LmConfig`). This document covers how to pick a backend, every
`LmConfig` field, the verbose trace format, and the LM algorithm in
brief.

## TL;DR -- which backend?

**Default to `solve_sparse_faer_f32` (or `solve_sparse_faer` for
f64).** For most real problems the Hessian is sparse enough that
sparse Cholesky is the right choice, and `faer` is the best-supported
backend -- pure Rust, no external dependency, benchmarks cleanly, and
handles the full sparsity pattern of a SLAM-like problem.

Pick anything else only when you have a specific reason:

| Backend | When |
|---|---|
| **`solve_sparse_faer[_f32]`** | **default**. Any non-trivial problem -- SLAM, bundle adjustment, sketch solver, anything with > ~10 parameters or a sparse Hessian structure |
| `solve[_f32]` (dense) | toy problems (≤ 4 parameters), or when the Hessian is actually dense and small. Simple, no overhead from sparse bookkeeping |
| `solve_band[_f32]` | **only** when the Hessian is genuinely block-tridiagonal with a known half-bandwidth `kd` (pose-only localisation, smoother-like problems). ~10x faster than dense at 500 poses but hard-errors on any off-band element |

Feature-gated backends (`BandLapack`, `SparseDirect`,
`SparseEigen`, `SparseCholmod`) target specific environments: LAPACK
interop, alternative factorisation strategies, external-solver
interop. Enable them via cargo features only when you need the
interop -- they don't outperform faer on any common workload.

## Basic usage

Define the model with `#[arael(root)]`, build it, and call the generated
solve method. It reads the parameters out of the model, runs LM, and writes
the optimized values back in:

```rust,ignore
use arael::simple_lm::LmConfig;

let cfg = LmConfig::<f32> { verbose: true, ..Default::default() };
let result = model.solve_sparse(&cfg);   // indexed sparse faer -- the default backend
println!("{} iterations: {:.4} -> {:.4}",
    result.iterations, result.start_cost, result.end_cost);
// `model` now holds the optimized parameters.
```

`#[arael(root)]` generates `solve_sparse` and `solve_dense` on the root
(along with the `LmProblem` impl the solver actually consumes) -- you never
write either by hand. Reach for `solve_dense` on small or dense problems;
otherwise `solve_sparse` is the default (see the backend table above).

## Advanced usage

**Choose a damping schedule.** The schedule driver lives on the config; the
default is the fixed-multiplier one. For strongly nonlinear problems (bundle
adjustment) attach the gain-ratio Nielsen schedule, and set a
problem-appropriate `initial_lambda` -- matching the schedule and tolerances
to the problem is where the performance is (see
[Damping-schedule drivers](#damping-schedule-drivers)):

```rust,ignore
let cfg = LmConfig::<f64> { initial_lambda: 1e-6, ..Default::default() }
    .with_driver(NielsenLambdaDriver::default());
let result = model.solve_sparse(&cfg);
```

**Use an explicit backend.** `solve_with` takes any `LmSolver` instance --
for backends that need construction, e.g. band Cholesky with a known
half-bandwidth `kd`:

```rust,ignore
let result = model.solve_with(&mut Band::new(kd), &cfg);
```

**Manage the parameter vector yourself.** The generated methods own the
serialize -> solve -> deserialize round trip. Drop to the free `solve_*`
functions when you need the flat parameter vector directly -- warm-starting
from a previous estimate, reusing one buffer across many solves, or timing
the first iteration on its own:

```rust,ignore
let mut params = Vec::<f32>::new();
model.serialize32(&mut params);
let result = solve_sparse_faer_f32(&params, &mut model, &cfg);   // free function
model.deserialize32(&result.x);
```

## `LmConfig` -- every field, with defaults

```rust,ignore
LmConfig {
    abs_precision:   1e-6,   // small-step cost drop threshold
    rel_precision:   1e-4,   // small-step relative-drop threshold
    max_iters:       100,    // hard cap on iterations
    min_iters:       5,      // don't terminate before this many accepted steps
    patience:        3,      // this many consecutive small steps → stop
    initial_lambda:  1e-4,   // starting LM damping
    cost_threshold:  0.0,    // stop when cost ≤ this (0.0 disables)
    verbose:         false,  // print per-iteration trace to stderr
}
```

| Field | Default | Meaning |
|---|---|---|
| `abs_precision` | `1e-6` | a step is "small" when cost drop falls below this **and** below `rel_precision` |
| `rel_precision` | `1e-4` | fractional cost improvement below this counts as "small". `(old - new) / old` |
| `max_iters` | `100` | hard cap on total iterations (counts inner damping retries) |
| `min_iters` | `5` | solver will not terminate before this many accepted steps, regardless of precision |
| `patience` | `3` | consecutive small steps before termination. Prevents premature termination from one lucky step |
| `initial_lambda` | `1e-4` | starting damping. Small ≈ Gauss-Newton (fast, may overshoot), large ≈ gradient descent (slow, stable) |
| `cost_threshold` | `0.0` | terminate immediately when cost drops to or below this. Useful for feasibility-style problems with a known target |
| `verbose` | `false` | per-iteration line on stderr. **Turn on first whenever debugging** |

## Tuning for performance vs quality

The defaults are a reasonable middle ground; for a production solve
you'll usually want to tune. The central trade-off is **iterations
vs convergence quality**: every iteration costs time (one Hessian
assembly + one Cholesky + step evaluation), and the last few
iterations often deliver very little cost improvement. Too many
iterations and you pay for marginal refinement; too few and the
final solution is biased toward the initial guess. Spend time
measuring the actual vs the needed on representative inputs.

Knobs, in rough order of impact:

- **`verbose: true` first**. Read the per-iteration cost and lambda
  trace for a few real solves. You'll see immediately whether the
  solver converges in 3, 10, or 80 iterations, whether it keeps
  making progress near the end, and whether Cholesky is rejecting
  steps. You cannot tune what you haven't measured.
- **`max_iters`**. The hard cap. Default 100 is safe for
  development; in production it's often worth reducing to a value
  informed by the verbose trace (e.g. 20 if convergence typically
  happens by iteration 15). Too low and solves terminate
  mid-descent; too high and pathological inputs waste budget.
- **`rel_precision` / `abs_precision`**. Loosening these is the
  fastest way to shave iterations off a converged solve. For a
  real-time pass that just needs "good enough", bumping
  `rel_precision` from `1e-4` to `1e-3` often halves the iteration
  count. For batch refinement where precision matters, tighten to
  `1e-5` or below; watch the trace to confirm you're buying real
  cost improvement and not just grinding numerical noise.
- **`patience`**. Higher = more conservative about terminating (one
  lucky step below threshold won't end the solve). 3 is a good
  default; drop to 1 only if you've already measured that single
  below-threshold steps are reliable signals of convergence on your
  problem. Raising patience above 3 rarely helps except on very
  noisy cost landscapes.
- **`min_iters`**. Floor on accepted steps before termination can
  fire. Useful when the initial state happens to be near a local
  minimum and the precision check would stop after one step -- keep
  at default 5 unless you have a specific reason.
- **`initial_lambda`**. Should match how far the initial state is
  from the minimum:
  - **Warm start** (the system was already solved and you're adding
    a small amount of new data, e.g. incremental SLAM with one new
    pose appended to a converged trajectory) -- use a **small**
    `initial_lambda` like `1e-6`. Near the minimum the linearisation
    is accurate and Gauss-Newton-style steps converge in a handful
    of iterations.
  - **Cold start** (fresh batch solve, noisy initial estimates far
    from the true state) -- use a **larger** `initial_lambda` like
    `1e-2` or `1e-1`. Gradient-descent-like steps are more stable
    when the Jacobian is a poor local approximation; you'll save
    several iterations otherwise spent rejecting overshoots and
    ratcheting λ up.
  Default `1e-4` is a middle-ground guess when you don't know which
  regime you're in. If the verbose trace shows the first few
  iterations repeatedly rejecting, λ was too small; if cost barely
  moves across the first few iterations, λ was too large.
- **`cost_threshold`**. Set to a non-zero value only when you
  actually have a known target cost (feasibility-style problems,
  e.g. "get residuals below the measurement-noise floor"). Terminates
  as soon as the cost drops below the threshold without waiting for
  the patience counter. Leave at `0.0` for ordinary least-squares.

Process:

1. Run with `verbose: true` on representative input; look at the
   per-iteration cost trace and the final cost.
2. Find the iteration at which cost improvement effectively stops.
   Call that `K`.
3. Set `max_iters` to a comfortable margin above `K` (e.g. `2*K`).
4. Loosen `rel_precision` until the solver terminates around
   iteration `K`, not past it. If the final cost moves meaningfully
   on real data, back off.
5. Re-run on the full input distribution to confirm you haven't
   regressed any corner case.

For graduated-optimisation loops (see "Graduated optimisation"
below), tune each pass independently -- the early, loose passes
converge fast and can use aggressive thresholds; the final tight
pass typically needs more iterations and tighter precision.

### At the very top end

Iteration counts are discrete, so the savings stack multiplicatively
when the counts are small. Going from 4 iterations to 3 is a **25%
reduction** in compute cost; 3 to 2 is 33%; 2 to 1 is 50%. In
high-frequency loops (real-time localisation at 60 Hz, per-frame
bundle adjustment) these single-iteration wins dominate the overall
runtime. The tuning process above leaves enough performance on the
table that more advanced techniques are worth considering at this
scale:

- **Learned initial parameters.** Train a small model (regression,
  small NN, gradient-boosted tree) on pairs of *(problem
  features, optimised params)* from past solves and use its
  prediction as the LM starting point. A good initial guess can
  turn a 4-iteration solve into a 1- or 2-iteration solve.
- **Learned termination decision.** Train a classifier on the
  *verbose trace* of past solves (cost, Δcost, λ, step size) to
  predict "further iterations won't materially change the
  solution". Replacing the fixed `patience + rel_precision`
  criterion with a learned predictor can cut the tail iterations
  that deliver negligible cost improvement.
- **Problem-specific λ schedules** beyond the single
  `initial_lambda` knob -- e.g. warm-start λ from the previous
  frame's converged damping in an incremental pipeline.

These are research-grade refinements, not drop-in features. Reach
for them only after the basic tuning process has plateaued and the
per-solve cost is genuinely the bottleneck.

## Termination logic

An accepted step counts as "small" when EITHER its absolute cost
improvement is below `abs_precision` OR its relative improvement is
below `rel_precision` (so `abs_precision` alone can stop a solve whose
cost is tiny, and `rel_precision` alone can stop one that plateaus at
a large cost -- mind the flip side: a badly scaled problem making
large absolute progress at small relative rates will stop; rescale or
lower `rel_precision`).

The solver stops when all of:

- `iter >= min_iters`
- the last `patience` consecutive accepted steps were "small"

or on any of:

- `iter >= max_iters`
- `cost <= cost_threshold`
- the new cost is within `8 * epsilon * cost` of the old one
  (machine-precision noise floor -- further digits are not
  resolvable)
- 20 consecutive damped attempts without an accepted step (the inner
  retry budget; hard-coded)
- the damping schedule gives up (the lambda driver returns `None` on a
  rejection -- for the default schedule, lambda passing 1e10)

## LM in five bullets

On each iteration the solver:

1. **Linearises** the residuals at the current `x`: `r, J`.
2. **Damps**: assemble `J^T J + λ * diag(J^T J)` and right-hand side `J^T r`.
3. **Solves** the damped system with the chosen backend. Cholesky
   either accepts (matrix is positive-definite) or rejects.
4. **Accepts or rejects** the step by comparing new cost to old:
   better → accept and shrink λ (0.2×); worse → reject and grow
   λ (10×), retry with the new damping.
5. **Repeats** until the termination rules fire.

`λ` behaves as a trust-region radius: large `λ` produces small, safe
steps; small `λ` produces large Gauss-Newton steps that move faster
when the linearisation is accurate.

## Verbose trace format

Each accepted or rejected iteration prints one line to stderr:

```
3/0: 44.5679->44.5403 / 0.0276, lambda=2e-5 (step=91)
```

| Field | Meaning |
|---|---|
| `3/0` | iteration / inner-retry counter. `0` means the Cholesky succeeded on the first try |
| `44.5679->44.5403` | cost before → cost after the step (if accepted). On a reject the values are the rolled-back state |
| `0.0276` | absolute cost improvement (old - new) |
| `lambda=2e-5` | damping in effect for this step |
| `(step=91)` | wall-clock microseconds for this iteration |

A Cholesky rejection gets a longer diagnostic line (commit
`6c72586`):

```
5/0: Cholesky failed (damped matrix not positive-definite), lambda=1.6e-7 -> 1.6e-6 (step=177) [non-finite: grad=0 diag=0 x=0 matrix=0] [diag<=0: 0]
```

| Field | What it says |
|---|---|
| `lambda=1.6e-7 -> 1.6e-6` | the bump λ gets on the retry (×10) |
| `non-finite: grad=N diag=N x=N matrix=N` | NaN/Inf counts in each scratch buffer. All zero means the matrix is fully finite, the problem is structural |
| `diag<=0: N` | count of Hessian-diagonal entries ≤ 0. Any non-zero count means some parameter is untouched by every constraint (indices left at `u32::MAX`) or a negative contribution leaked in -- both are bugs, not rounding noise |

When all four non-finite counts are 0 and `diag<=0` is 0, the
rejection is f32 accumulation noise at tiny λ (see commit context in
`loc_global_demo` fix). When any are non-zero, stop and fix the
model -- the solver can't tell you more.

Reference "healthy" trace: run
`cargo run --release --example slam_demo` with `verbose: true`
(the example sets it). Look for:

- cost dropping on most iterations
- occasional 10× λ bumps when a step gets rejected
- no non-finite counts
- clean convergence in 5-15 iterations per isigma pass

## `LmProblem` -- the solver's interface

```rust,ignore
pub trait LmProblem<T> {
    fn calc_cost(&mut self, params: &[T]) -> T;
    fn calc_grad_hessian_dense(...) -> T;   // all assembly methods
    fn calc_grad_hessian_band(...) -> T;    // return the cost as a
    fn calc_grad_hessian_sparse(...) -> T;  // free byproduct
    fn calc_grad_hessian_sparse_direct(...) -> T;
    fn calc_grad_hessian_sparse_indexed(...) -> T;
    fn advance(&mut self, params: &mut [T]);
}
```

The `#[arael(root)]` macro generates all of these from your constraint
attributes. You only call them via `solve*`; you never implement
them by hand.

`advance` is called after every ACCEPTED step. It exists for
`EulerAngleParam`: the accepted delta angles are folded into the
reference rotation and their parameter slots reset to zero, which is
what keeps the parameterization in its small-angle sweet spot on
arbitrarily oriented problems. Hand-written `LmProblem` impls without
re-centering state can leave it empty.

## `LmResult`

```rust,ignore
pub struct LmResult<T> {
    pub x: Vec<T>,          // optimised parameter vector
    pub start_cost: T,
    pub end_cost: T,
    pub iterations: usize,           // including inner damping retries
    pub accepted_iterations: usize,  // cost-decreasing steps only
    pub status: LmStatus,            // why the solve stopped
    pub final_lambda: T,             // damping at exit (seeds a warm restart)
}
```

`status` says *why* the solve stopped, so callers can branch on the outcome
directly instead of inferring convergence from the cost or iteration count:

```rust,ignore
pub enum LmStatus {
    Converged,             // patience small steps / noise floor / zero start cost
    CostThreshold,         // reached LmConfig::cost_threshold
    MaxIterations,         // hit LmConfig::max_iters
    LambdaCeiling,         // driver gave up: lambda past its ceiling
    RetryBudgetExhausted,  // 20 inner retries with no accepted step
    DegenerateDiagonal { param: usize },  // non-positive Hessian diagonal
}
```

`LambdaCeiling` and `RetryBudgetExhausted` both mean "no step could be
accepted," but distinguish the driver hitting its damping ceiling from the
hard inner-retry cap. `DegenerateDiagonal` returns the best parameters found
so far rather than panicking. `LmResult` derives `Clone`/`Debug`.

After a successful solve, hand `result.x` back to the model via
`model.deserialize32(&result.x)` (or `deserialize64`).

## Graduated optimisation

When a constraint is *stiff* -- large sigma spread across residual
groups -- start with the tight constraints *loose* and ramp up. The
LM surface is smoother at low isigma; convergence is more reliable
and faster than throwing the tight problem at the solver from a
noisy initial estimate.

The idiom is a scale field on the root:

```rust,ignore
struct Path {
    // ... other fields ...
    frine_isigma_scale: f32,
}

// constraint body multiplies it in:
let plain1 = atan2(r_f.y, r_f.x) * feature.isigma.x * path.frine_isigma_scale;

// main loop:
for scale in [0.01, 0.1, 1.0] {
    path.frine_isigma_scale = scale;
    let result = solve_sparse_faer_f32(&params, &mut path, &cfg);
    // ...
}
```

See `loc_demo.rs` / `slam_demo.rs` for the full pattern. Keep the
scale on a single root field and set it to the desired absolute
value each pass -- the constraint body reads the current value, so
the sigma you see is always the sigma in effect.

## Pre-solve centering

For models with root-level rigid-transform parameters (translation +
rotation applied to every entity), a coarse pass that **freezes all
per-entity params and optimises only the globals** can absorb
systematic offset in the initial estimates before the main sweep
begins. See `Path::optimise_center` in
[loc_global_demo.rs](../examples/loc_global_demo.rs):

1. Set `pose.pos.optimize = false` / `pose.ea.optimize = false` for every pose.
2. Solve (only root-level params vary).
3. Bake the result into poses via `recenter()`, resetting globals to identity.
4. The Param constructors in `recenter` reset `.optimize = true` automatically.

## Damping-schedule drivers

The lambda schedule is pluggable via the `LambdaDriver` trait: the LM
loop asks the driver for every damping decision and reports each
attempted step's outcome as a `LambdaStep { lambda, cost, new_cost,
grad, diagonal, delta }` -- the gradient, Gauss-Newton Hessian
diagonal, and attempted step vector give a driver everything needed to
compute model-quality measures itself (the Nielsen driver derives its
gain ratio from them). `start` fires after the first assembly with a
`LambdaState { cost, grad, diagonal }`, so even the initial lambda can
be chosen from the problem's actual scale.

- `DefaultLambdaDriver` -- the classic fixed-multiplier schedule and
  what every plain entry point uses: divide lambda by 5 on acceptance
  (clamped to `LmConfig::lambda_floor`), multiply by 10 on rejection or
  factorization failure, give up when a rejection would pass 1e10.
- `NielsenLambdaDriver` -- the gain-ratio adaptive schedule (Nielsen,
  IMM-REP-1999-05): on acceptance lambda scales by
  `max(1/3, 1 - (2 rho - 1)^3)`, on rejection it multiplies by an
  escalating `nu` (2, 4, 8, ... reset to 2 by the next acceptance).
  Use it on strongly nonlinear problems where the fixed schedule
  oscillates -- dividing lambda by a constant after every acceptance
  marches straight into the next rejection (bundle adjustment is the
  canonical case; see benchmarks/bal).

Custom drivers implement the four-method trait (`start`, `accepted`,
`rejected`, `factorization_failed` -- the latter also receives the
current `LambdaState`); returning `None` from `rejected`
abandons the solve like an exhausted retry budget. A driver is a
`#[derive(Clone)]` type; attach it to the config with
`LmConfig::with_driver(...)` and every solve entry point (`lm_solve`,
`solve_sparse_faer`, ...) picks it up from `config.driver`.
