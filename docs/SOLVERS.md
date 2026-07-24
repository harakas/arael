# Solvers

arael ships a Levenberg-Marquardt solver with multiple linear-algebra
backends sharing one trait (`LmSolver`) and one config struct
(`LmConfig`). This document covers how to pick a backend, every
`LmConfig` field, the verbose trace format, and the LM algorithm in
brief.

## TL;DR -- which backend?

**Default to `solve_sparse_f32` (or `solve_sparse` for f64).** For most real problems the Hessian is sparse enough that
sparse Cholesky is the right choice, and `faer` is the best-supported
backend -- pure Rust, no external dependency, benchmarks cleanly, and
handles the full sparsity pattern of a SLAM-like problem.

Pick anything else only when you have a specific reason. Every backend
comes in two forms: an `LmSolver` instance for the root's `solve_with`
(the main API -- see below) and a `simple_lm::` free function over a raw
parameter vector.

| Backend (`solve_with(&mut ..., &cfg)`) | Free function | When |
|---|---|---|
| **`SparseFaer::<T>::new()`** (`T` = `f64`/`f32`) | **`solve_sparse[_f32]`** | **default** (= the root's `solve_sparse`). Any non-trivial problem -- SLAM, bundle adjustment, sketch solver, anything with > ~10 parameters or a sparse Hessian structure. Sparsity pattern discovered once, indexed assembly after |
| `Dense` | `solve_dense[_f32]` | dense nalgebra Cholesky (= the root's `solve_dense`): low parameter counts, or when the Hessian is actually dense and small. The free `solve[_f32]` picks by itself: dense for <= 6 params, SparseFaer otherwise |
| `Band::new(kd)` | `solve_band[_f32]` | **only** when the Hessian is genuinely block-tridiagonal with a known half-bandwidth `kd` (pose-only localisation, smoother-like problems). ~10x faster than dense at 500 poses but hard-errors on any off-band element |
| `BandLapack::new(kd)` | `solve_band_lapack[_f32]` | the same band solve through LAPACK `dpbsv`/`spbsv` (feature `lapack`) -- for LAPACK-standardised environments |
| `SparseEigen::<T>::new()` | `solve_sparse_eigen[_f32]` | Eigen `SimplicialLLT` through a C++ shim (feature `eigen`) -- for Eigen interop/comparison; measured well behind faer |
| `SparseCholmod::new()` | `solve_sparse_cholmod` | CHOLMOD simplicial Cholesky, LGPL (feature `cholmod`; f64 only) -- comparable to Eigen simplicial, behind faer |
| `SparseCholmodSupernodal::new()` | `solve_sparse_cholmod_supernodal` | CHOLMOD supernodal Cholesky (feature `cholmod-gpl`; f64 only). **License warning: the Supernodal module is GPL**, unlike the LGPL simplicial one -- enabling it makes the binary subject to the GPL |
| `SparseCoo::new()` / `SparseDirectCsc::new()` | `solve_sparse_coo` / `solve_sparse_direct_csc` | COO / direct-CSC assembly over a DENSE solve -- validation baselines for the assembly paths, not for production. (the root's `.solve_sparse()` method is faer) |

## Basic usage

Define the model with `#[arael(root)]`, build it, and call the generated
solve method. It reads the parameters out of the model, runs LM, and writes
the optimized values back in:

```rust,ignore
use arael::simple_lm::{LmConfig, LmProblem};   // or `use arael::prelude::*;`

let cfg = LmConfig::conservative().with_verbose(true);
let result = model.solve_sparse(&cfg)?;  // indexed sparse faer -- the default backend
println!("{} iterations: {:.4} -> {:.4}",
    result.iterations, result.start_cost, result.end_cost);
// `model` now holds the optimized parameters.
```

Every solve entry point returns [`SolveResult`](#solve-failures) --
`Ok(LmResult)` when the solve terminated by a stopping rule (including
`MaxIterations` and `TimeLimit`), `Err(SolveFailure)` when it broke: the
system could not be built or factored, or a Hessian diagonal went
NaN / negative / zero. The failure carries the best accepted state when
one exists (see [Solve failures](#solve-failures)).

The solve methods live on `LmProblem` as default methods (hence the
`use`), gated on `RootProblem` -- the two-method parameter round trip
(`serialize` / `deserialize`) that `#[arael(root)]` implements for
every root, along with the `LmProblem` impl the solver actually
consumes. You never write any of it by hand. `solve_with` is the
general entry point taking any backend instance; `solve_sparse` =
`solve_with(SparseFaer)` is the default, `solve_dense` =
`solve_with(Dense)` for small or dense problems (see the backend table
above). The methods match the root's precision: on an
`#[arael(root, f32)]` model they take `f32` configs and `solve_sparse`
uses `SparseFaer<f32>`.

## Advanced usage

**Choose a damping schedule.** The schedule driver lives on the config; the
default is the fixed-multiplier one. For strongly nonlinear problems (bundle
adjustment) attach the gain-ratio Nielsen schedule, and set a
problem-appropriate `initial_lambda` -- matching the schedule and tolerances
to the problem is where the performance is (see
[Damping-schedule drivers](#damping-schedule-drivers)):

```rust,ignore
let cfg = LmConfig::ill_conditioned().with_initial_lambda(1e-6);
let result = model.solve_sparse(&cfg)?;
```

**Use an explicit backend.** `solve_with` takes any `LmSolver` instance --
for backends that need construction, e.g. band Cholesky with a known
half-bandwidth `kd`:

```rust,ignore
let result = model.solve_with(&mut Band::new(kd), &cfg)?;
```

**Marginalize landmark-style blocks.** On landmark SLAM structures --
many small parameter blocks coupled to poses but never to each other --
those blocks can be eliminated before the factorization (a Schur
complement), leaving only the poses to factorize and recovering the
landmarks by back-substitution afterwards.

`solve_sparse` does this by itself. Nothing has to be marked: the macro
hands the backend the model's type coupling graph, `SparseFaer` reads the
marginalizable families off it, and decides from the block structure
whether marginalizing them is actually faster than factorizing the whole
system -- it is not always, and on Ladybug-1723 (1723 cameras) it is 1.6x
slower. Ask what it did with `LmResult::solver`.

Name the blocks yourself when you know better than the graph does:

```rust,ignore
#[arael(root, marginalize(landmarks))]
```

or on the backend, as parameter ranges:

```rust,ignore
let mut solver = SparseFaer::new().with_marginalize(lm_params..n);
let result = model.solve_with(&mut solver, &cfg)?;
```

A named set is used as given, and it must be legal -- the blocks in it may
not couple to each other, or marginalizing them is not defined and the
solve is rejected. Whether it PAYS is still weighed; `with_policy` settles
that by hand:

```rust,ignore
SchurPolicy::Auto { .. }  // default: decide from the structure
SchurPolicy::Force        // marginalize, no questions asked
SchurPolicy::Never        // factorize the whole system
```

Under `Never` a named set is not wasted: it becomes the factorization's
ordering instead ("marginalized parameters first"), which is the same
elimination performed inside the factorization rather than before it.
`with_ordering` controls that directly -- see `FaerOrdering`.

**Manage the parameter vector yourself.** The generated methods own the
serialize -> solve -> deserialize round trip. Drop to the free `solve_*`
functions when you need the flat parameter vector directly -- warm-starting
from a previous estimate, reusing one buffer across many solves, or timing
the first iteration on its own:

```rust,ignore
let mut params = Vec::<f32>::new();
model.serialize32(&mut params);
let result = solve_sparse_f32(&params, &mut model, &cfg);   // free function
model.deserialize32(&result.x);
```

## `LmConfig` -- every field, with defaults

Named presets cover the common regimes; start from one and override fields as
needed. `LmConfig::conservative()` is the `Default`; `well_conditioned()`
fits a good start (near-Gauss-Newton first step, no iteration floor);
`ill_conditioned()` brings the gain-ratio damping driver;
`continue_from(&result)` resumes a capped solve at its ending lambda. See
their rustdoc for the exact settings.

```rust,ignore
LmConfig {
    abs_precision:   1e-6,   // small-step cost drop threshold
    rel_precision:   1e-4,   // small-step relative-drop threshold
    max_iters:       100,    // hard cap on iterations
    min_iters:       5,      // don't terminate before this many accepted steps
    patience:        3,      // this many consecutive small steps → stop
    initial_lambda:  1e-4,   // starting LM damping
    cost_threshold:  0.0,    // stop when cost ≤ this (0.0 disables)
    gradient_tolerance: None,   // stop when max|g_i| <= tol
    parameter_tolerance: None,  // stop when |step| <= tol * (|x| + tol)
    min_diagonal:    None,   // floor under the damping scale
    time_limit:      None,   // wall-clock budget; None = no limit
    num_threads:     1,      // threads for the linear solve (needs `rayon`)
    verbose:         false,  // print per-iteration trace to stderr
    gather_timing:   false,  // collect per-phase timing into LmResult::timing
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
| `gradient_tolerance` | `None` | `Option<T>`. Stop when `max|g_i| <= tol`. **The only criterion that tests for a stationary point** -- the cost tests only say the cost stopped improving, which also happens while drifting along a gauge freedom. arael minimizes `sum r^2`, so its gradient is `2 J^T r`; a solver minimizing `1/2 sum r^2` reads the same number twice as tight, so do not carry a tolerance across without halving it. Respects `min_iters` |
| `parameter_tolerance` | `None` | `Option<T>`. Stop when `\|step\|_2 <= tol * (\|x\|_2 + tol)` -- the parameters have stopped moving. A different question from the cost test: the cost can plateau while the step still does real work, and the step can vanish while the cost still creeps. Checked on an accepted step, before `advance()` re-centers. Respects `min_iters` |
| `min_diagonal` | `None` | `Option<T>`. Floor under the DAMPING scale: `H[i,i] + lambda * max(H[i,i], min_diagonal)`. `None` leaves the scale at `H[i,i]` -- the classic multiplicative damping `(1 + lambda) * H[i,i]`. **Without it a parameter of zero curvature FAILS the solve** (`Err` with `SolveFailureKind::DegenerateDiagonal`) -- `(1 + lambda) * 0` is still 0, so the system is singular and no step can ever be accepted. With it, that parameter gets `lambda * min_diagonal` of damping, the factorization succeeds, and it simply does not move (its gradient is zero too). **A zero diagonal means the system is badly formulated -- a parameter nothing constrains -- so this is a bandaid and should be avoided**; the parameter it damps through stays unconstrained and its value is meaningless. Fix the model first: constrain it, hold it fixed (`Param::fixed`), or leave the entity out. Reach for the floor only when a residual can legitimately switch itself off (a `branch` guarding an undefined observation, a saturated robustifier) and an entity can end one iteration with nothing reaching it. 1e-6 is reasonable. Rescues a ZERO diagonal only: NEGATIVE and NaN stay fatal, since `J^T J`'s diagonal is a sum of squares and either value means the assembly is poisoned |
| `time_limit` | `None` | `Option<Duration>` wall-clock budget for the whole solve. **Overrides `min_iters`** -- a spent budget stops the solve wherever it is, returning the last accepted step (`LmStatus::TimeLimit`). Checked before each assembly and each damped attempt, so the overrun is bounded by one linear solve, not one iteration. It cannot preempt a single factorization. `None` = no limit, and the clock is never read |
| `num_threads` | `1` | threads for the sparse factorization and triangular solve. `1` sequential, `n` uses n, `0` uses every core. **Requires the `rayon` cargo feature**; without it anything but 1 warns and stays sequential. Threading has overhead: whether it helps depends on the model and its parameter count. See [Threads](#threads) |
| `verbose` | `false` | per-iteration line on stderr. **Turn on first whenever debugging** |
| `gather_timing` | `false` | gather per-phase wall-clock timing into `LmResult::timing` (`Some` when on, `None` when off). Off = the clock is never read |

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
5/0: Cholesky failed (damped matrix not positive-definite), lambda=1.6e-7 -> 1.6e-6 (step=177) [non-finite: grad=0 diag=0 x=0 matrix=0]
```

| Field | What it says |
|---|---|
| `lambda=1.6e-7 -> 1.6e-6` | the bump λ gets on the retry (×10) |
| `non-finite: grad=N diag=N x=N matrix=N` | NaN/Inf counts in each scratch buffer. All zero means the matrix is fully finite, the problem is structural |

A non-positive diagonal is not printed here: it is caught before the inner loop and fails the solve with `SolveFailureKind::DegenerateDiagonal { param, fault }`, naming the parameter that no constraint reaches.

When all four non-finite counts are 0, the
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
    pub timing: Option<LmTiming>,    // per-phase wall clock; Some iff gather_timing
}
```

`status` says *why* the solve stopped, so callers can branch on the outcome
directly instead of inferring convergence from the cost or iteration count:

```rust,ignore
pub enum LmStatus {
    Converged,             // patience small steps / noise floor / zero start cost
    CostThreshold,         // reached LmConfig::cost_threshold
    MaxIterations,         // hit LmConfig::max_iters
    GradientTolerance,     // max|g_i| <= LmConfig::gradient_tolerance
    ParameterTolerance,    // |step| <= tol * (|x| + tol)
    PredictedReduction,    // model predicts no meaningful improvement left
    TimeLimit,             // spent LmConfig::time_limit
    DriverTerminated,      // LambdaDriver::accepted returned None -- step KEPT
    LambdaCeiling,         // driver gave up: lambda past its ceiling
    RetryBudgetExhausted,  // 20 inner retries with no accepted step
    Aborted,               // partial state inside a SolveFailure -- never in Ok
}
```

`LambdaCeiling` and `RetryBudgetExhausted` both mean "no step could be
accepted," but distinguish the driver hitting its damping ceiling from the
hard inner-retry cap. `DriverTerminated` is the opposite case: the driver
stopped the solve on a step it *liked*, and that step is kept. All of these
are `Ok` terminations: the returned parameters are the best accepted ones.
`Aborted` never appears in an `Ok` result -- it marks the partial state
carried by a `SolveFailure` (below). `LmResult` derives `Clone`/`Debug`.

## Solve failures

Every solve entry point returns `SolveResult<T> = Result<LmResult<T>,
SolveFailure<T>>`:

```rust,ignore
pub struct SolveFailure<T> {
    pub kind: SolveFailureKind,           // what broke
    pub partial: Option<Box<LmResult<T>>>, // best accepted state, if any
}

pub enum SolveFailureKind {
    Setup(SolveError),  // the linear system could not be built or factored
    DegenerateDiagonal { param: usize, fault: DiagonalFault },
}

pub enum DiagonalFault { Nan, Negative, Zero }
```

`Setup(SolveError)` is a structural failure: a band element outside the
declared bandwidth, a parameter no constraint touches, a failed symbolic
factorization, an illegal marginalization, or a backend compiled out
(`SolverUnavailable`). When it strikes on the first assembly nothing ran:
`partial` is `None` and the model's parameters are untouched.

`DegenerateDiagonal` means a parameter's Gauss-Newton Hessian diagonal
went bad at the current iterate: `Nan` and `Negative` mean the assembly
is poisoned (the diagonal is a sum of squares, so neither can happen in
healthy arithmetic); `Zero` means no constraint curvature reaches the
parameter -- see `LmConfig::min_diagonal` for damping through the
transient case.

When the solve broke after accepted steps, `partial` carries a full
`LmResult` with `status = LmStatus::Aborted` -- the best accepted
parameters, usable for diagnosis or a `LmConfig::continue_from` restart.
`into_partial()` consumes the failure and hands it over. `SolveFailure`
implements `Display` and `std::error::Error`, so `?` works in
`main() -> Result<(), Box<dyn Error>>`.

## Threads

Off by default: arael is a single-threaded solver. The one thing that can be
threaded today is the sparse factorization and triangular solve, which faer runs
on rayon's global thread pool.

```toml
[dependencies]
arael = { version = "0.7", features = ["rayon"] }
```

```rust,ignore
// 1 = sequential (the default), n = n threads, 0 = every core
let cfg = LmConfig::conservative().with_num_threads(4);
let result = model.solve_sparse(&cfg)?;
```

Without the feature, a `num_threads` other than 1 warns and runs sequentially --
it does not silently pretend. `num_threads: 0` resolves to
`rayon::current_num_threads()`, so it honours `RAYON_NUM_THREADS` or whatever
`ThreadPoolBuilder` the application installed; the pool is shared with the rest of
the process.

Threading has overhead. Whether it helps, and by how much, depends on the model
and its number of parameters -- measure.

Only the sparse factorization and triangular solve are threaded; assembly, the
Schur reduction and every other backend are sequential.

## Where the log goes

`arael::log` sets the level and the destination of every `info!` / `warn!` /
`error!` arael emits -- the `verbose` trace, the backend's report of what it
chose, and the warnings from saturated guards.

```rust,ignore
use arael::log::{self, Level};

log::silence();                  // emit nothing
log::set_level(Level::Error);    // errors only
log::set_sink(|level, msg| {     // route them anywhere
    my_logger::write(level.tag(), msg);
});
log::reset_sink();               // back to stderr
```

`Level` is ordered `Off < Error < Warn < Info`. The check happens at the call
site before the message is formatted, so a silenced arael allocates nothing.

## Reporting a solve -- `LmResult::print`

```rust,ignore
let r = model.solve_sparse(&cfg)?;

r.print();          // plain ASCII, to stdout
r.pretty_print();   // colour and glyphs, to stdout

let s: String = r.report();          // the same text print() writes
let s: String = r.pretty_report();   // the same text pretty_print() writes
let s: String = r.render(Style { colour: true, unicode: false });  // pick both
println!("{r}");    // Display is report()
```

`report()` is pure ASCII with no escape sequences, so it is safe in a log or a
file. `pretty_report()` carries ANSI colour and box glyphs and is for a terminal.
Both draw the same facts:

```text
LM converged in 12 iterations (9 accepted, 3 retried)
  ----------------------------------------------------------
  cost      8129.39 -> 0.000925674  (100.00% down)
  lambda    2.56e-10 at exit
  time      21.40 ms
    assembly          8.20 ms   38.3%  ########............    9 calls, first 3.10 ms
    linear solve      9.91 ms   46.3%  #########...........   12 calls, first 4.00 ms
    cost eval         2.10 ms    9.8%  ##..................   12 calls, first 0.20 ms
    advance           0.01 ms    0.1%  ....................    9 calls, first 0.00 ms
    steps         +++-+x++-+++
                  + accepted   - rejected   x not positive definite
  backend   Schur: 480 blocks / 1440 params eliminated, 360 kept
            fill ratio 0.45, ordered by Amd
```

The timing block appears only when `gather_timing` was set, and the backend line
only when the backend reported a plan.

## Profiling a solve -- `LmTiming`

Set `gather_timing: true` to find out where a solve spends its time. The
result's `timing` is then `Some(LmTiming)` (otherwise `None`, and the clock
is never read):

```rust,ignore
pub struct LmTiming {
    pub total: Duration,          // whole solve
    pub analysis: Duration,       // the backend's ONE-TIME structural work
    pub assembly: Duration,       // residual + Jacobian + Hessian, all iters
    pub first_assembly: Duration, //   ...of which the first
    pub linear_solve: Duration,   // damped NUMERIC factorization + solve, all attempts
    pub first_linear_solve: Duration,  // ...of which the first
    pub cost_eval: Duration,      // trial-point cost (residual only)
    pub first_cost_eval: Duration,
    pub advance: Duration,        // post-step re-centering
    pub first_advance: Duration,
    // plus a *_count for each phase

    pub steps: Vec<LmStep>,       // the per-iteration timeline -- see below
}
```

### What the setup costs -- `LmTiming::analysis`

Before the backend can factorize anything it discovers the sparsity pattern and
the value-position map, detects the marginalizable blocks, weighs the Schur
reduction (two trial symbolic factorizations), chooses the fill-reducing ordering,
and factorizes symbolically. All of it runs inside the first `compute`, and
`analysis` is what it cost.

`assembly` is then the model's residual and Jacobian work, on every iteration
including the first. The split is exact: a backend reports how much of its
`compute` was assembly, and the solver takes the remainder.

```text
    analysis          0.08 ms   14.4%  ###.................    1 call
    assembly          0.05 ms   10.0%  ##..................   10 calls, first 0.01 ms
```

A seventh of a small solve. On a large one it can dominate: Ladybug-1723's
ordering comparison alone runs into seconds.

The symbolic factorization is here, not in `linear_solve` -- `solve_damped` only
ever factorizes numerically, against the symbolic factorization `compute` made
once.

### The timeline -- `LmTiming::steps`

One record per **attempt**, damping retries included, so
`steps.len() == LmResult::iterations`. The totals bucket a rejected attempt and
an accepted one together; the timeline keeps them apart.

```rust,ignore
pub struct LmStep {
    pub iter: usize,          // 1-based, counts retries -- matches the verbose trace
    pub inner: usize,         // retry index at THIS linearization; 0 = first attempt
    pub accepted: bool,
    pub factorization_failed: bool,   // damped Cholesky said not positive definite

    pub lambda: f64,          // the damping this attempt used
    pub cost: f64,            // before
    pub new_cost: f64,        // at the trial point; NaN if the factorization failed
    pub step_norm: f64,       // |delta|_2 -- how far it proposed to move
    pub grad_max: f64,        // max|g_i| here; constant across the retries that share it

    pub time: Duration,           // the whole attempt (NOT including `assembly`/`analysis`)
    pub assembly: Duration,       // charged to inner == 0 only; a retry re-factorizes,
                                  //   it does not re-assemble
    pub analysis: Duration,       // the one-time structural work; non-zero on iter 1 only
    pub linear_solve: Duration,   // damped factorization + solve, this attempt
    pub cost_eval: Duration,      // trial-point cost; zero if the factorization failed
    pub advance: Duration,        // re-centering; zero unless the step was kept
}
```

The scalars are `f64` at any solve precision (an `f32` converts exactly), which
keeps `LmTiming` free of a type parameter. The per-step phases sum to the
aggregate totals above.

A run of increasing `inner` at a rising `lambda` is a solve retrying its own
step; `assembly` is charged only to `inner == 0`, so the timeline shows what each
retry actually cost.


Every phase records `{total, first, count}`. The first call is broken out
because the first iteration carries a one-time structure cost the steady
state does not: the first assembly discovers the sparsity pattern and builds
the value-position map, and the first solve runs the symbolic factorization
(fill-reducing ordering + elimination tree). On a fresh sparse problem those
two often dominate the whole solve; a warm re-solve pays neither. Each
`first_*` is part of its phase total (`first_assembly` ⊂ `assembly`, etc.).
Phase times are single-threaded wall clock and cover only the work inside
each phase, so they sum to slightly less than `total`.

For the **steady-state cost per iteration**, use the `mean_*` methods --
`mean_assembly()`, `mean_linear_solve()`, `mean_cost_eval()`,
`mean_advance()`. Each returns `(phase - first) / (count - 1)` as an
`Option<Duration>` (the first iteration is excluded so the one-time structure
cost does not skew the average), or `None` when only the first call ran.

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
    let result = solve_sparse_f32(&params, &mut path, &cfg);
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
`rejected`, `factorization_failed` -- the latter two also receive the
current `LambdaState`). All three step hooks return `Option<T>`, and
**`None` always means "stop the solve"**. Which status comes back says
whether a step survived:

| Hook returns `None` | Status | The step |
|---|---|---|
| `accepted` | `DriverTerminated` | **kept** -- it is already in the parameters and the cost |
| `rejected` | `LambdaCeiling` | none was produced; the last accepted one comes back |
| `factorization_failed` | `LambdaCeiling` | none was produced; the last accepted one comes back |

`accepted -> None` is the hook for a stopping rule the config cannot
express: a step-norm test (`step.delta`), a gradient-norm test
(`step.grad`), an external deadline, a good-enough cost. It is the only
way to stop on a *good* step, and it beats `min_iters` -- the driver's
rule wins over the config's. The built-in schedules never use it.

A driver is a `#[derive(Clone)]` type; attach it to the config with
`LmConfig::with_driver(...)` and every solve entry point (`lm_solve`,
`solve_sparse`, ...) picks it up from `config.driver`.
