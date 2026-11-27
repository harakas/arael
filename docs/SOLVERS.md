# Solvers

arael's Levenberg-Marquardt (LM) solver has several linear-algebra
backends behind one trait (`LmSolver`) and one config struct
(`LmConfig`).

## Which backend

Use `solve_sparse`, the default: sparse Cholesky through faer, pure Rust
with no external dependency. Pick another backend only for a specific
reason. Every backend is an `LmSolver` instance: a root hands it to
`solve_with`, a hand-written problem to `lm_solve` with its parameter
vector, and `SolverKind` names one at run time.

| Backend (`solve_with(&mut ..., &cfg)`) | When |
|---|---|
| `SparseFaer::<T>::new()` (`T` = `f64`/`f32`) | the default (= the root's `solve_sparse`): SLAM, bundle adjustment, a sketch solver, any problem with a sparse Hessian. The sparsity pattern is found once, and the factorization runs in block form (see [The block supernodal Cholesky](#the-block-supernodal-cholesky)). The only backend that takes [mixed precision](#mixed-precision) and a [GPU](#solving-on-a-gpu) |
| `Dense` | dense nalgebra Cholesky (= the root's `solve_dense`): few parameters, or a Hessian that is dense and small |
| `Band::new(kd)` | only when the Hessian is banded with a known half-bandwidth `kd` (pose-only localisation, smoothers). Far cheaper than dense on such a system; an element outside the band is an error |
| `BandLapack::new(kd)` | the same band solve through LAPACK `dpbsv`/`spbsv` (feature `lapack`) |
| `SparseEigen::<T>::new()` | Eigen `SimplicialLLT` through a C++ shim (feature `eigen`), for comparison with Eigen; slower than faer |
| `SparseCholmod::new()` | CHOLMOD simplicial Cholesky, LGPL (feature `cholmod`; f64 only); slower than faer |
| `SparseCholmodSupernodal::new()` | CHOLMOD supernodal Cholesky (feature `cholmod-gpl`; f64 only). License: the Supernodal module is GPL, unlike the LGPL simplicial one, so enabling it makes the binary subject to the GPL |

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

The solve methods are default methods of `LmProblem` (hence the `use`),
which `#[arael(root)]` implements for every root. `solve_with` takes any
backend instance; `solve_sparse` is `solve_with(SparseFaer)` and
`solve_dense` is `solve_with(Dense)`. The methods match the root's
precision: on an `#[arael(root, f32)]` model they take `f32` configs and
`solve_sparse` uses `SparseFaer<f32>`.

## Advanced usage

### Choose a damping schedule

The schedule driver is part of the config; the default multiplies lambda
by fixed factors. For strongly nonlinear problems (bundle adjustment)
attach the gain-ratio Nielsen schedule and set an `initial_lambda` that
suits the problem (see
[Damping-schedule drivers](#damping-schedule-drivers)):

```rust,ignore
let cfg = LmConfig::ill_conditioned().with_initial_lambda(1e-6);
let result = model.solve_sparse(&cfg)?;
```

### Use an explicit backend

`solve_with` takes any `LmSolver` instance, for backends that need
construction, such as band Cholesky with a known half-bandwidth `kd`:

```rust,ignore
let result = model.solve_with(&mut Band::new(kd), &cfg)?;
```

### Marginalize landmark-style blocks

On landmark SLAM structures (many small parameter blocks coupled to
poses but never to each other) those blocks can be eliminated before the
factorization, by a Schur complement. Only the poses are then
factorized, as the reduced system, and the landmarks are recovered by
back-substitution.

`solve_sparse` does this by itself. Nothing has to be marked: the macro
hands the backend the model's type coupling graph, `SparseFaer` reads
the families that can be marginalized off it, and decides from the block
structure whether marginalizing them is faster than factorizing the
whole system. It is not always: on a bundle problem with many cameras
the reduced system can factor slower than the whole one. `LmResult::solver`
says what it did.

Name the blocks yourself when you know better than the graph does:

```rust,ignore
#[arael(root, marginalize(landmarks))]
```

or on the backend, as parameter ranges:

```rust,ignore
let opts = SparseFaerOptions::default().with_marginalize(lm_params..n);
let result = model.solve_with(&mut SparseFaer::from_options(&opts), &cfg)?;
```

`SparseFaerOptions` is the one configuration of the sparse backend: plain
fields with a builder each, `Default` for the analysis deciding everything.
`SparseFaer::new()` is that default as a solver, `SparseFaer::from_options`
any other; `SolverKind::Sparse` carries the same struct.

A named set is used as given, and it must be legal: the blocks in it may
not couple to each other, or marginalizing them is not defined and the
solve is rejected. Whether it pays is still weighed; `with_policy` settles
that by hand:

```rust,ignore
SchurPolicy::Auto { .. }  // default: decide from the structure
SchurPolicy::Force        // marginalize, whatever the structure
SchurPolicy::Never        // factorize the whole system
```

Under `Never` a named set becomes the factorization's ordering instead
("marginalized parameters first"), which is the same elimination done
inside the factorization. `with_ordering` controls that directly; see
`SolveOrdering`.

### The block supernodal Cholesky

`SparseFaer` factorizes in block form by default: the model's blocks are
kept whole and each supernode of the factor is one dense panel. Nothing
has to be set. The system it factorizes is the whole Hessian, or the
reduced system when blocks are marginalized. These are the other routes
`SparseFaer` can take for that system:

| Route | Taken when |
|---|---|
| faer's scalar Cholesky | the model has no block structure (a hand-written problem, or Hessian entries that all come through `coo`), or `BlockSupernodalMode::Never` asks |
| the envelope factorization | a reduced system left in its natural order, when `EnvelopeMode` chooses it. Less memory than the block route, and slower (see `with_envelope_schur`) |
| the narrow-band Cholesky | a banded whole system, when `with_narrow_band(true)` asks |
| conjugate gradients | when asked (see [Conjugate gradients on the reduced system](#conjugate-gradients-on-the-reduced-system)) |

```rust,ignore
BlockSupernodalMode::Auto    // default: the block route wherever the model has block structure
BlockSupernodalMode::Always  // the same choice as Auto
BlockSupernodalMode::Never   // faer's scalar Cholesky
```

The block route runs at any thread count (see
[The linear solve](#the-linear-solve)). Two options trade speed and
memory:

```rust,ignore
SparseFaer::from_options(&SparseFaerOptions::default()
    .with_block_supernodal_batching(None)          // off; the default is Some(1.5)
    .with_block_supernodal_memory_lean(true))      // smaller factor, a little slower
```

Batching packs consecutive small updates into one matrix product, which
speeds the factorization a little at no memory cost. The memory-lean
option merges fewer supernodes: a smaller factor, a little slower.
`SchurPlan::block_supernodal`, `envelope` and `cg_iterations` report
which route ran.

### Mixed precision

`with_mixed_precision` computes and stores the factor in `f32` and
refines each damped solve to `f64`: half the factor's memory and the
speed of single-precision arithmetic, for a few more solves through the
factor.

```rust,ignore
SparseFaer::from_options(&SparseFaerOptions::default()
    .with_mixed_precision(Some(MixedPrecision::default())))
```

It pays where the factor is large: bundle adjustment from a few hundred
cameras up, SLAM maps with landmarks. It is a loss on pose graphs, whose
factors are small and whose refinement converges slowly.

The matrix and the right-hand side stay `f64`. Each damped solve is
refined against the `f64` matrix (`r = b - A x`, `A dx = r` through the
`f32` factor, `x += dx`) until the relative residual is under `tol` or
`max_refinements` steps are spent (defaults 1e-10 and 4). A system too
ill-conditioned for single precision stops refining early and hands the
LM loop an inexact step, which it meets with more damping.

- Only the block supernodal route factors this way. The other routes
  ignore the setting.
- On an `f32` solver the factor is single precision already. The setting
  then adds only the refinement, which runs in `f64`, with `tol` floored
  at about 1e-6.
- `SchurPlan::f32_factor`, `refinements` and `refinement_shortfalls`
  report what the solve took.
- On a GPU the setting also decides the precision of the system the GPU
  solves (see [Precision](#precision)).

### Conjugate gradients on the reduced system

`with_iterative_schur` solves the reduced system by preconditioned
conjugate gradients in place of factorizing it. Nothing is factorized,
so there is no fill and no factor to store, and the step is inexact. It
pays where factorizing the reduced system is most of a damped solve, as
on a bundle adjustment with many cameras.

```rust,ignore
let opts = SparseFaerOptions::default()
    .with_policy(SchurPolicy::Force)              // there must be a reduction
    .with_iterative_schur(CgParams::default());
```

With nothing marginalized there is no reduced system and the solve
fails, hence `SchurPolicy::Force`. The preconditioner is block Jacobi:
the reduced system's diagonal blocks. `CgParams` holds the settings:

| Field | Default | Meaning |
|---|---|---|
| `tol` | `1e-6` | stop when the preconditioned residual `r^T M^-1 r` is down to this fraction of its starting value. The quantity is squared: 1e-6 is 1e-3 on the residual's norm |
| `max_iters` | `0` | the most iterations a solve takes; 0 is the system's dimension |
| `restart_every` | `0` | recompute the residual `r = b - A x` every this many iterations in place of updating it, at one matrix-vector product each time; 0 never does |

`with_implicit_schur` is the same solve on a reduced system that is not
formed: each product walks the Hessian instead. It trades the reduction
for one such product per iteration, so it pays where a solve takes few
iterations. `SchurPlan::cg_iterations` is the count of iterations of
either form, summed over the solve.

`with_iterative_schur` also runs on a GPU (see
[Conjugate gradients on a GPU](#conjugate-gradients-on-a-gpu)).

### Manage the parameter vector yourself

The generated methods serialize the parameters, solve and deserialize
them. Use `lm_solve`, the form every method is built on, when you need
the flat parameter vector: to start from a previous estimate, to reuse
one buffer across many solves, or to time the first iteration on its
own. It takes the vector, a backend instance and the problem:

```rust,ignore
let mut params = Vec::<f32>::new();
model.serialize(&mut params);                                            // RootProblem
let result = lm_solve(&params, &mut SparseFaer::<f32>::new(), &mut model, &cfg)?;
model.deserialize(&result.x);
```

`lm_solve_with_context` is the same solve under a caller-owned `Context`
(see [Threads](#threads)).

### Pick the backend at run time

`SolverKind` names a backend as a value (`Dense`, `Band { kd }`,
`BandLapack { kd }`, `Sparse(SparseFaerOptions)`, `Eigen`, `Cholmod`,
`CholmodSupernodal`), for a choice made from a config file or a command
line:

```rust,ignore
let result = model.solve(SolverKind::Sparse(SparseFaerOptions::default()), &cfg)?;
```

A kind that is not compiled in, or not available at the model's scalar,
returns `SetupError::SolverUnavailable` with the parameters untouched,
instead of failing to build.

## `LmConfig` -- every field, with defaults

Named presets cover the common regimes; start from one and override fields as
needed. `LmConfig::conservative()` is the `Default`; `well_conditioned()`
fits a good start (near-Gauss-Newton first step, no iteration floor);
`ill_conditioned()` brings the gain-ratio damping driver;
`continue_from(&result)` resumes a capped solve at its ending lambda. See
their rustdoc for the exact settings.

| Field | Default | Meaning |
|---|---|---|
| `abs_precision` | `1e-6` | a step is small when the cost drops by less than this, or by less than `rel_precision` of the cost |
| `rel_precision` | `1e-4` | a relative cost drop `(old - new) / old` below this is small |
| `max_iters` | `100` | the most iterations, damping retries included |
| `max_accepted_iters` | `None` | `Option<usize>`. Stop after this many accepted steps (`LmStatus::MaxIterations`) |
| `min_iters` | `5` | the stopping rules do not apply before this many accepted steps |
| `patience` | `3` | small steps in a row that stop the solve |
| `initial_lambda` | `1e-4` | the damping the solve starts with. Small is close to Gauss-Newton (fast, may overshoot), large close to gradient descent (slow, stable) |
| `lambda_floor` | `1e-12` (f32: its epsilon) | lambda never drops below this after an accepted step. Raise it (1e-6) for a near-singular problem, such as a bundle adjustment with its gauge freedom |
| `driver` | `DefaultLambdaDriver` | how lambda moves from one attempt to the next (a [`LambdaDriver`](#damping-schedule-drivers)); `NielsenLambdaDriver` through `with_nielsen()`. Reads `initial_lambda` and `lambda_floor` when the solve starts |
| `cost_threshold` | `0.0` | stop as soon as the cost is at or below this; `0.0` turns it off. For problems with a known target cost |
| `gradient_tolerance` | `None` | `Option<T>`. Stop when `max_i \|g_i\| / sqrt(H_ii)` is at or below this. The division by the Hessian diagonal makes the value independent of the parameters' units. The only rule that tests for a stationary point: the cost rules only see the cost stop improving, which also happens while drifting along a gauge freedom. Respects `min_iters` |
| `parameter_tolerance` | `None` | `Option<T>`. Stop when `\|step\|_2 <= tol * (\|x\|_2 + tol)`: the parameters have stopped moving. Checked on an accepted step, before `advance()` re-centers. Respects `min_iters` |
| `predicted_reduction_tolerance` | `None` | `Option<T>`. Stop when, after an accepted step, the model predicts less than this fraction of the cost as the next gain |
| `min_diagonal` | `None` | `Option<T>`. Floor under the damping scale: `H[i,i] + lambda * max(H[i,i], min_diagonal)`. Without it a parameter of zero curvature fails the solve with `SolveFailureKind::DegenerateDiagonal`; with it that parameter is damped by `lambda * min_diagonal` and does not move. Such a parameter is unconstrained and its value means nothing, so fix the model first: constrain it, hold it fixed (`Param::fixed`), or leave the entity out. Use the floor only where a residual can switch itself off for an iteration (a `branch` guarding an undefined observation, a saturated robust loss). 1e-6 is a reasonable value. A negative or NaN diagonal still fails the solve |
| `time_limit` | `None` | `Option<Duration>`, a wall-clock budget for the solve. When it is spent the solve returns the last accepted step (`LmStatus::TimeLimit`), whatever `min_iters` says. Checked before each assembly and each damped attempt, so one factorization can overrun it. `None`: no limit, and the clock is never read |
| `num_threads` | `1`, or `ARAEL_NUM_THREADS` | threads for the linear solve and for the cost and assembly sweeps: `1` sequential, `n` uses n, `0` every core. The environment variable sets the default; a count set in code wins. Needs the `threads` cargo feature; without it any count but 1 warns and stays sequential. See [Threads](#threads) |
| `assembly_threads` | `None` | `Option<usize>`. A thread count for the cost and assembly sweeps alone, on the same scale; the linear solve keeps `num_threads`. `None` uses `num_threads` for both |
| `covariance` | `None` | `Option<CovMode>`. `Some(mode)` makes the solve assemble the parameter covariance at its solution into `LmResult::covariance` ([docs/COVARIANCE.md](COVARIANCE.md)). Set with `with_covariance` |
| `covariance_options` | `CovOptions::auto()` | how that covariance is assembled: the elimination ordering and whether the block supernodal Cholesky factorizes. Read only when `covariance` asks. Set with `with_covariance_options` |
| `verbose` | `false` | log one line per iteration (see [Verbose trace format](#verbose-trace-format)). Turn it on first when debugging |
| `observer` | `None` | an [`LmObserver`](#iteration-observer) called once per damped attempt; can stop the solve. Set with `with_observer` |
| `gather_timing` | `false` | record where the time went, in `LmResult::timing`. Off, the clock is never read |
| `device_observer_params` | `false` | with the LM loop on a GPU, read the parameters back after each accepted step for the observer; off, `LmIter::params` is empty there. See [Observers and drivers on a GPU](#observers-and-drivers-on-a-gpu) |
| `device_driver_vectors` | `false` | with the LM loop on a GPU, read the gradient, the Hessian diagonal and the step back for the lambda driver; off, those slices are empty there |

## Tuning for performance vs quality

Every iteration costs an assembly, a factorization and a cost
evaluation, and the last iterations of a solve improve the cost little.
Tuning is choosing where to stop: measure on representative inputs.

The settings, in rough order of effect:

- `verbose: true` first. The trace of a few real solves shows how many
  iterations the solve takes, whether it still improves near the end,
  and whether steps are rejected.
- `max_iters`. The default of 100 is safe for development. Lower it to
  a margin above where the trace shows the cost stops improving.
- `rel_precision` and `abs_precision`. Loosening them removes
  iterations from the end of a solve; tightening them (`1e-5` or below)
  buys precision. Check on the trace that the extra iterations still
  improve the cost.
- `patience`. A higher value is more careful about stopping: one small
  step does not end the solve. Lower it to 1 only when single small
  steps are a reliable sign of convergence on the problem.
- `min_iters`. Keeps a solve that starts near a minimum from stopping
  after one step.
- `initial_lambda`. Match it to how far the start is from the minimum:
  - a warm start (a solved system with a little new data, such as one
    pose appended to a converged trajectory): small, like `1e-6`. Near
    the minimum the linearisation is accurate and Gauss-Newton steps
    converge in a few iterations;
  - a cold start (noisy estimates far from the solution): larger, like
    `1e-2` or `1e-1`, which saves the iterations spent rejecting
    overshoots.
  If the first iterations of the trace are rejected, lambda was too
  small; if the cost barely moves in them, it was too large.
- `cost_threshold`. Set it only with a known target cost, such as the
  measurement-noise floor. The solve stops as soon as the cost is at or
  below it, without waiting for `patience`.

A procedure:

1. Solve representative input with `verbose: true`.
2. Find the iteration `K` at which the cost stops improving.
3. Set `max_iters` to a margin above `K`, such as `2*K`.
4. Loosen `rel_precision` until the solve stops around iteration `K`.
   If the final cost moves on real data, tighten it again.
5. Check the whole range of inputs again.

In a [graduated optimisation](#graduated-optimisation) tune each pass
on its own.

## Termination logic

An accepted step is small when its cost improvement is below
`abs_precision`, or below `rel_precision` of the cost. So
`abs_precision` alone can stop a solve whose cost is tiny, and
`rel_precision` alone one that improves slowly at a large cost: lower
it, or rescale the problem.

The solver stops when both hold:

- `iter >= min_iters`
- the last `patience` accepted steps were small

or on any of:

- `iter >= max_iters`
- `accepted >= max_accepted_iters`, when set
- `cost <= cost_threshold`
- `gradient_tolerance`, `parameter_tolerance` or
  `predicted_reduction_tolerance` is met, when set and
  `iter >= min_iters`
- `time_limit` is spent, whatever `min_iters` says
- a rejected step's new cost is within `16 * epsilon * cost` of the
  old one, and so is the reduction the linear model predicted for it,
  or the step is within `16 * epsilon * |x|`: the limit of the
  scalar's precision
- 20 damped attempts in a row without an accepted step
- the lambda driver returns `None` (the default driver does when
  lambda passes 1e10 on a rejection)
- an observer returns `Break`

[`LmStatus`](#lmresult) names which rule stopped the solve.

## The algorithm in brief

On each iteration the solver:

1. linearises the residuals at the current `x`: `r`, `J`;
2. damps: assembles `J^T J + lambda * diag(J^T J)` and the right-hand
   side `J^T r`;
3. solves the damped system with the chosen backend. The Cholesky
   factorization fails when the matrix is not positive definite;
4. accepts or rejects the step by comparing the new cost to the old:
   on a lower cost it accepts and divides lambda by 5, otherwise it
   rejects, multiplies lambda by 10 and tries again;
5. repeats until a termination rule applies.

Lambda acts as a trust-region radius: a large lambda gives small, safe
steps, a small one large Gauss-Newton steps, which move faster where
the linearisation is accurate. Each try at one damping is called an
attempt below.

## Verbose trace format

With `verbose` each attempt logs one line (see
[Where the log goes](#where-the-log-goes); the default destination,
stderr, prefixes each line with `[arael INFO] `):

```text
3/0: 44.5679->44.5403 / 0.0276, lambda=2e-5 (step=91)
```

| Field | Meaning |
|---|---|
| `3/0` | the iteration and the retry counter; `0` is the first attempt at this linearization |
| `44.5679->44.5403` | the cost before and after the step; on a rejection, the state the solve went back to |
| `0.0276` | the cost improvement (old - new) |
| `lambda=2e-5` | the damping of this attempt |
| `(step=91)` | wall-clock microseconds of this attempt |

`SparseFaer` adds where the microseconds went, inside the same
parentheses (another backend can through `LmSolver::attempt_note`):

```text
(step=2194, factor=229, solve=16)
(step=95403, factor=40633, solves=1687, residuals=668, 1 refinement step, residual 7.2e-11)
(step=899, gpu=889)
(step=3692, gpu=3624, cg 38 iterations (64 queued in 3 rounds), residual 9.6e-4)
```

| Addition | Meaning |
|---|---|
| `factor=229, solve=16` | the block supernodal Cholesky: its factorization and its triangular solves |
| `solves=1687, residuals=668, 1 refinement step, residual 7.2e-11` | under `MixedPrecision`: the solves and the residuals of the refinement, its steps and the relative residual reached (`no refinement` with none asked, `short of tolerance` when it stopped short) |
| `gpu=889` | on a GPU: the part of the attempt spent on the GPU's damped solve, timed on the host, followed by the refinement as above |
| `cg 38 iterations (64 queued in 3 rounds), residual 9.6e-4` | conjugate gradients on a GPU: the iterations the solve took, the iterations sent to the GPU and in how many batches (a batch can outlast the solve), and the relative residual reached |

A failed factorization logs a longer line:

```text
5/0: Cholesky failed (damped matrix not positive-definite), lambda=1.6e-7 -> 1.6e-6 (step=177) [non-finite: grad=0 diag=0 x=0 matrix=0]
```

| Field | Meaning |
|---|---|
| `lambda=1.6e-7 -> 1.6e-6` | the damping of the failed attempt and of the next one |
| `non-finite: grad=N diag=N x=N matrix=N` | counts of NaN and infinite values in the gradient, the diagonal, the parameters and the matrix |

With all four counts 0 the matrix is finite and not positive definite
at this damping; the next attempt damps more. A non-zero count means
the model produces non-finite values: fix the model.

A zero, negative or NaN diagonal is not logged here: it fails the solve
with `SolveFailureKind::DegenerateDiagonal { param, fault }`, which
names the parameter.

`cargo run --release --example slam_demo` prints a trace of a solve
that converges: the cost drops on most iterations, lambda rises on an
occasional rejection, and no count is non-zero.

## `LmProblem` -- the solver's interface

```rust,ignore
pub trait LmProblem<T> {
    fn calc_cost(&mut self, params: &[T]) -> T;
    fn calc_grad_hessian_dense(...) -> T;   // both assembly methods
    fn calc_grad_hessian_sparse(...) -> T;  // return the cost as a
    fn advance(&mut self, params: &mut [T]);// free byproduct
}
```

`#[arael(root)]` generates all of these from the constraint attributes;
a model calls them through the `solve*` methods. The two assembly
methods are also how to get a Hessian to look at: dense for a small
model, COO sparse for a large one. On a generated root each plain call
builds the model's block store and drops it again; the `_with_context`
forms keep it (see [What a solve
keeps](#what-a-solve-keeps----context-and-lmsession)).

`LmProblemInternals` is the other half: the routes a backend picks
(band, CSC, indexed), the context forms of the evaluations, the walks of
the structure the sparse patterns are built from, and the elimination
hints. It is inner API, like `arael::store`: public so a caller can
reach the context forms and the structure walks, and it may change
without a major version bump. Every method has a default, so a
hand-written problem opts in with one empty impl:

```rust,ignore
impl LmProblem<f64> for MyProblem { /* cost, dense, sparse, advance */ }
impl LmProblemInternals<f64> for MyProblem {}
```

Solving through a route the problem does not assemble then panics,
naming the route.

`advance` is called after every accepted step. It exists for
`EulerAngleParam`: the accepted angle steps are folded into the
reference rotation and their parameters reset to zero, which keeps the
angles small on problems of any orientation. A hand-written problem
without such state can leave it empty.

## What a solve keeps -- `Context` and `LmSession`

A solve of a generated root builds a block store: one array per
declared block field, holding every Hessian block and gradient entry
the sweeps write, and one store per thread when the sweeps thread. A
solve that assembles on its GPU builds none, and drops the stores a
context holds. The plain entry points, `lm_solve` and the `solve_*`
methods, build it per solve and drop it after. To keep it across solves of one model, hold an
`arael::Context` and solve through `lm_solve_with_context`, or solve
through an `LmSession`, which keeps one of its own. A solve through a
context fills the same allocations again; the `_with_context`
evaluations of `LmProblemInternals` panic on a model of another block
count than the one the context was built for.

`LmSession` keeps a backend and what it learns about one problem's
structure -- the sparsity pattern, the position map, the ordering, the
symbolic factorization, the Schur plan and the context -- so every
solve after the first skips the analysis. The config is read on every
solve, so tolerances, caps and the driver may change per call. The
parameter values may change freely; that is what the session is for.
The structure may not: after any change to the constraints, their
couplings or the block layout, call `invalidate` first, or the next
warm solve is undefined. A changed parameter count or block count
panics, as it cannot be the same structure; a change that keeps every
count passes unnoticed, so `invalidate` stays the rule. The sweep
thread count is part of the structure too: the count the first solve
used holds for the session's life, a later solve asking for another is
run at the pinned count with a warning, and `invalidate` releases it.
The linear solve's thread count is free to change.

```rust,ignore
let mut session = LmSession::new(SparseFaer::new());
let r1 = session.solve(&mut model, &cfg); // cold: full analysis
let r2 = session.solve(&mut model, &cfg); // warm: assembly + numerics
```

A covariance the config asks for (`with_covariance`) comes back in the
result here as on every other entry point ([docs/COVARIANCE.md](COVARIANCE.md)).

With a GPU (see [Solving on a GPU](#solving-on-a-gpu)) the session's
backend also keeps what it set up on the GPU for the structure, so a
warm solve sets nothing up there again.

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
    pub solver: Option<SolverReport>, // the sparse backend's plan; None for dense and band
    pub gpu: Option<GpuReport>,      // what the solve did on its GPU; None without one
    pub timing: Option<LmTiming>,    // per-phase wall clock; Some iff gather_timing
    pub threads: ThreadReport,       // what the solve's threads did
    pub covariance: Result<CovAssembly, CovError>, // Ok when the config asked for one (LmConfig::covariance)
}
```

`status` says *why* the solve stopped, so callers can branch on the outcome
directly instead of inferring convergence from the cost or iteration count:

```rust,ignore
pub enum LmStatus {
    Converged,             // patience small steps / noise floor / zero start cost
    CostThreshold,         // reached LmConfig::cost_threshold
    MaxIterations,         // hit LmConfig::max_iters
    GradientTolerance,     // max_i |g_i| / sqrt(H_ii) <= LmConfig::gradient_tolerance
    ParameterTolerance,    // |step| <= tol * (|x| + tol)
    PredictedReduction,    // model predicts no meaningful improvement left
    TimeLimit,             // spent LmConfig::time_limit
    DriverTerminated,      // LambdaDriver::accepted returned None -- step KEPT
    ObserverTerminated,    // LmObserver returned Break -- best state kept
    LambdaCeiling,         // driver gave up: lambda past its ceiling
    RetryBudgetExhausted,  // 20 inner retries with no accepted step
    Aborted,               // partial state inside a SolveFailure -- never in Ok
}
```

`LambdaCeiling` and `RetryBudgetExhausted` both mean that no step could
be accepted: the first that the driver reached its damping ceiling, the
second that the retry limit was reached. `DriverTerminated` is the
opposite case: the driver stopped the solve on a step it accepted, and
that step is kept. All of these are `Ok` terminations, and the returned
parameters are the best accepted ones. `Aborted` marks the partial state
carried by a `SolveFailure` (below) and never appears in an `Ok` result.

## Solve failures

Every solve entry point returns `SolveResult<T> = Result<LmResult<T>,
SolveFailure<T>>`:

```rust,ignore
pub struct SolveFailure<T> {
    pub kind: SolveFailureKind,           // what broke
    pub partial: Option<Box<LmResult<T>>>, // best accepted state, if any
}

pub enum SolveFailureKind {
    Setup(SetupError),  // the linear system could not be built or factored
    DegenerateDiagonal { param: usize, fault: DiagonalFault },
    Device(DeviceFailure),  // a GPU operation failed
}

pub enum DiagonalFault { Nan, Negative, Zero }

pub struct DeviceFailure {
    pub operation: &'static str,  // what the solve asked of the GPU: "assembly", "attempt", "setup of the CUDA solver", ...
    pub message: String,          // the error as the GPU's runtime reported it
}
```

`Setup(SetupError)` is a structural failure: a band element outside the
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

`Device` means a GPU operation failed: setting the GPU up (a kernel
compile, an allocation) or a step of the solve (an upload, a launch, a
read). The solve stops there. With the LM loop on the GPU its state is
not read back after a failure (an accepted step may be half applied), so
`partial` holds the parameters and the cost from where the GPU took
over, which agree with the model's own references. The next solve on the
same GPU starts afresh.

When the solve broke after accepted steps, `partial` carries a full
`LmResult` with `status = LmStatus::Aborted` -- the best accepted
parameters, usable for diagnosis or a `LmConfig::continue_from` restart.
`into_partial()` consumes the failure and hands it over. `SolveFailure`
implements `Display` and `std::error::Error`, so `?` works in
`main() -> Result<(), Box<dyn Error>>`.

## Validation -- `model.validate()`

The model linter: one pass reports every formulation problem it can
find, instead of a solve failing on the first one.

```rust,ignore
let d = model.validate();
if !d.is_clean() {
    println!("{}", d);        // one line per issue
}
assert!(model.validate().is_clean());   // the test / CI form
```

`validate()` returns an `arael::validate::Diagnostic` whose `issues`
are:

| Issue | Meaning |
|---|---|
| `NonFiniteParam { param, value }` | a parameter serialized to NaN / inf |
| `StaleRef { path }` | a `Ref` whose slot was removed or replaced -- would panic during a solve; `path` names it (`ties[3].a`) |
| `UnconstrainedParam { param }` | no constraint curvature reaches the parameter -- a solve would fail with `DegenerateDiagonal { fault: Zero }` |
| `BadDiagonal { param, value }` | negative / NaN Hessian diagonal -- poisoned assembly |
| `GradientMismatch { param, analytic, numeric }` | the assembled gradient disagrees with central finite differences of `calc_cost` -- a wrong hand-declared derivative (`#[arael::function]` `derivs`, a `deriv =` cache) |

The checks run in stages: stale refs are reported alone (nothing else
can even serialize past them), then non-finite parameters, then the
assembly-based checks and the gradient comparison. Fix what a stage
reports and validate again. The pass costs one assembly plus `2n` cost
evaluations -- a debugging tool, not something to run per solve. The
stale-ref walk covers the root's refs, direct struct fields, and
root-level collections; refs inside nested sub-model collections still
panic with the stale-Ref message instead of appearing as an issue.

The gradient pieces stand alone too:

```rust,ignore
let fd = model.numeric_gradient(&params);        // central differences, 2n cost evals
let d  = model.check_gradients(&params);         // fd vs assembled gradient, default tol
let d  = model.check_gradients_tol(&params, 1e-3); // explicit tolerance (noisy f32 costs)
```

`check_gradients` is the check that catches a wrong `derivs = [...]` on
an `#[arael::function]` or a wrong hand `deriv =` cache -- the cost is
computed from the function values, the gradient from the declared
derivatives, and only a finite-difference comparison sees them disagree.

## Threads

Off by default: arael is a single-threaded solver. The `threads` feature runs
the linear solve on rayon's global thread pool, and the cost and assembly
sweeps on arael's own worker threads.

```toml
[dependencies]
arael = { version = "0.9", features = ["threads"] }
```

```rust,ignore
// 1 = sequential (the default), n = n threads, 0 = every core
let cfg = LmConfig::conservative().with_num_threads(4);
let result = model.solve_sparse(&cfg)?;
```

Without the feature, a `num_threads` other than 1 warns and runs
sequentially. `num_threads: 0` is every
core: the rayon pool's size, which honours `RAYON_NUM_THREADS` or whatever
`ThreadPoolBuilder` the application installed. The cost and assembly
sweeps, the Schur reduction and the block factorization's subtrees and
windows run on arael's own workers (`arael::pool`); faer's dense kernels
run on rayon's pool, shared with the rest of the process.

`ARAEL_NUM_THREADS` in the environment is the default of `num_threads` for
every config built in the process, `LmConfig::default()` and the presets
alike, on the same scale. It is read once, and a count set in code wins over
it, as with `RAYON_NUM_THREADS` and `OMP_NUM_THREADS`. A value that is not a
count is an error, not 1.

### The sweeps

The cost evaluation and the assembly of the gradient and Hessian run on
arael's own workers: one thread per store beyond the calling thread,
spawned on first use, parked between sweeps and kept for the life of
the process, or until `arael::pool::shutdown()` joins them; the next
threaded sweep spawns them again. One sweep dispatches at a time, so two
solves running at once in one process take turns with their sweeps. A
root that should stay sequential says so, and keeps its linear solve
threaded:

```rust,ignore
#[arael(root, seq)]
struct Scene { .. }
```

A solve divides every top-level collection into one contiguous range per
thread, and each thread sweeps its own ranges into its own store: a
copy of the Hessian blocks and the gradient entries those constraints
write, holding only the entities that range reaches. No two threads
write the same value and nothing is locked. A serial pass then adds the
stores into the gradient. On the sparse routes the Hessian is written
from the stores by the same threads: the pattern's tiles are sorted by
where they lie in the value buffer and cut into one range of it per
store, and each thread writes the first block of a tile over the tile
and adds the rest of its blocks, so the buffer is never zeroed and no
two threads write one value here either. The dense and band routes add the
stores into the Hessian serially. The ranges are cut by the work under
each slot, not by slot count, so a collection whose entries carry very
different amounts of work still divides evenly.

The cost evaluation of a trial step is split the same way, over the same
ranges. It only reads, so it has nothing to gather afterwards but the
sums.

The model is read from every thread at once, so a root must be `Sync`. A
root holding a `Cell`, `RefCell` or `Rc` fails to compile; the error
names the field and says to opt out with `seq`.

Summing each entity's contributions per range and then across the ranges
reorders the additions: a threaded assembly matches the sequential one to
rounding. The cost and the parameters a solve lands on differ in the last
bits between thread counts.

A threaded solve assembles into its stores and never touches the one a
sequential solve uses. Neither costs anything in the model: a block field
holds no values, only its place in the storage the solve owns (see
[docs/MODEL.md](MODEL.md#a-block-field-is-a-declaration)).

### What the threads did

Every solve's [report](#reporting-a-solve----lmresultprint) carries a
threads block: the counts the two halves were given and the form each
sweep ran in. With `gather_timing` it also carries where the sweeps' own
time went.

```
  threads   sweeps 4, linear 4
    assembly    threaded      7 calls
    cost        threaded      8 calls
    stores      4 hold 1260 self blocks against the model's 1000 (1.26x), 4000 cross blocks against 4000, 2.1 MB against 1.9
    per assembly region 0.39 ms, tasks max 0.22 mean 0.19, gather 0.02, scatter 0.42
    per cost    region 0.16 ms, tasks max 0.06 mean 0.06
```

`stores` is what the split stores hold against one whole store of the
model. A cross block sits in one store; a self block sits in every store
whose range touches its entity, so the self count above the model's is
what the split duplicates. `SweepReport::held` and `whole` carry the
same counts.

`region` is dispatch to join; `tasks max` is the longest single thread of
that region, so the gap between them is the dispatch and the wake-up, and
the gap between `max` and `mean` is the imbalance. A `bind` line, when
the solve bound its stores to a pattern, is that one-time binding: the
scatter stream's build, with its sort and its cut into one range per
thread.

A `seq` root prints a line saying so in place of the rows. The
same is in `LmResult::threads` for a caller that would rather read it
than parse it, and `ThreadReport::fell_back` is the one-line test.

The stores are what a solve keeps across solves of one model, through a
`Context` or an `LmSession`; see [What a solve
keeps](#what-a-solve-keeps----context-and-lmsession).

### The linear solve

Threads reach the block supernodal route three ways: independent
subtrees of the elimination tree run on separate threads; a panel with
enough update work is shared between the threads in windows of its
columns, and its triangular solve in windows of its rows; and the dense
kernels of the few panels larger still take the pool, size-gated. The
threaded factor matches the sequential one to rounding. The solve
phase's triangular solve runs on one thread. faer's scalar
factorization and triangular solve are handed the thread count and faer
splits what it can.

The Schur reduction runs on arael's own workers: the reduced system's
block columns are cut into one range per thread by the number of
products that land in them, and each thread forms its own columns and
its own part of the reduced right-hand side. The back-substitution
recovers the marginalized blocks in ranges the same way.

Of the analysis, the exact pricing of the routes runs its three
symbolic factorizations side by side, and the whole-system block route
prices its candidate orderings the same way; the rest is sequential.

In a solve on a GPU the threads run what stays on the host: the
analysis, and the assembly when it does not run on the GPU.

Threading has overhead. Whether it helps depends on the model and its
number of parameters: measure.

## Solving on a GPU

`SparseFaer` can run a solve on a GPU: the assembly of the gradient and
the Hessian, the damped solves and the LM loop's vector work. The host
analyses the structure once and then, with the assembly on the GPU,
mostly waits. A GPU pays where the
factor is large: bundle adjustment from a few hundred cameras up, SLAM
maps with landmarks. Pose graphs, and whole systems of tens of thousands
of tiny blocks, solve faster on the host. A GPU solve agrees with a host
solve to rounding.

### Enabling it

A cargo feature compiles in a GPU runtime:

| Feature | GPU | Needed at run time |
|---|---|---|
| `gpu-cuda` | NVIDIA | the CUDA driver library and NVRTC (CUDA's run-time compiler) |
| `gpu-hip` | AMD | the HIP runtime and hipRTC, from ROCm |
| `gpu-metal` | Apple silicon | macOS |
| `gpu-cpu` | none: the CPU device runs the GPU kernels on the host, for tests | a C++ compiler (`ARAEL_CPU_CXX`, else `CXX`, else `c++`; GCC or Clang) |

The runtimes load their libraries at run time, so a build needs no CUDA
toolkit or ROCm, and a program built with a runtime runs on machines
without that GPU. Several features can be on at once.

```toml
[dependencies]
arael = { version = "0.9", features = ["gpu-cuda", "gpu-hip"] }
```

A solve runs on a GPU when its backend is given one:

```rust,ignore
use arael::simple_lm::{GpuDevice, MixedPrecision, SparseFaer, SparseFaerOptions};

let gpu = GpuDevice::detect();   // the machine's GPU; None without one, and the solve stays on the host
// Metal has no f64: an f64 solve on it factorizes in f32 and refines
let mixed = gpu.as_ref().is_some_and(|g| !g.has_f64()).then(MixedPrecision::default);
let opts = SparseFaerOptions::default().with_gpu(gpu).with_mixed_precision(mixed);
let r = model.solve_with(&mut SparseFaer::from_options(&opts), &cfg)?;
if let Some(gpu) = &r.gpu {
    println!("gpu: {gpu}");   // what ran on it, see below
}
```

`SolverKind::Sparse` takes the same options.

### Choosing a GPU

| Call | The GPU |
|---|---|
| `GpuDevice::cuda(i)`, `hip(i)`, `metal(i)` | device `i` of that runtime |
| `GpuDevice::cpu()` | the CPU device (feature `gpu-cpu`): slow, for tests on machines without a GPU |
| `GpuDevice::detect()` | the machine's preferred GPU, `None` without one |
| `GpuDevice::available()` | the machine's GPUs as `GpuInfo` values (runtime, index, name, integrated), preferred first; `GpuInfo::device()` makes one a `GpuDevice` |
| `GpuDevice::shared(backend, i)` | one value per device for the whole process, for callers that cannot keep a value (the C, C++ and Python interfaces use it) |

The preferred GPU is a discrete one before an integrated one, then CUDA
before HIP before Metal, then the lower index. So a program built with
`gpu-cuda` and `gpu-hip` runs on an NVIDIA card where there is one, else
on an AMD one. `available` and `detect` ask only the runtimes compiled
in, open no device and do not list the CPU device. `detect` logs what it
found, and why a runtime listed nothing:

```text
[arael INFO] gpu: detected cuda:0 NVIDIA GeForce RTX 4090, preferred to hip:0 AMD Radeon Graphics (integrated)
[arael INFO] gpu: detected hip:0 AMD Radeon Graphics (integrated) -- cuda: the CUDA driver library is not installed
```

A `GpuDevice` value holds its GPU's state, created by the first solve:
a queue and the GPU memory. Solvers built from clones of one value share
them, and the memory a solver gives back is handed out again, so solve
after solve of the same structure allocates nothing
(`GpuDevice::reserved_bytes`, `buffers_created`). The memory is held
until the last clone is dropped. Separately created values have their
own state, and their solves can run at the same time.

### What runs on the GPU

Two of `SparseFaer`'s routes run on a GPU: the block supernodal
Cholesky, of the reduced system or of the whole one, and conjugate
gradients on a formed reduced system (`with_iterative_schur`). On them
the GPU does each damped solve: it forms the system, solves it and
recovers the eliminated blocks.

The other routes run on the host, and a GPU given to them is not used:
the solve logs a warning naming the route, and the report says
`GpuWork::Unused`. They are the envelope factorization and the narrow
band (`EnvelopeMode::Auto` can choose the envelope for a reduced system
in natural order), faer's scalar Cholesky (`BlockSupernodalMode::Never`,
or a model without block structure) and `with_implicit_schur`. To keep a
reduced system on the GPU:

```rust,ignore
SparseFaerOptions::default()
    .with_gpu(GpuDevice::detect())
    .with_envelope_schur(EnvelopeMode::Never)
```

How much of the solve runs on the GPU depends on the model:

- When every constraint of the root has a device form (below), the GPU
  also assembles the gradient, the Hessian and the trial costs, and
  keeps the LM loop's vectors (`GpuWork::Loop`). From the first
  linearization on, the parameters stay on the GPU; an attempt waits
  once, for a few numbers; the parameters come back at the end of the
  solve.
- Otherwise the host assembles and the Hessian's values are uploaded for
  each damped solve (`GpuWork::Solves`). So it is also with
  `with_gpu_assembly(false)`, and for an f64 root on a GPU without f64
  (Metal).

The host always does the analysis: the structure, the ordering and the
symbolic factorization. `SolveOrdering::Auto` can order a reduced system
differently on a GPU than on the host, as fits a GPU's factorization;
`with_ordering` fixes the ordering, and `SchurPlan::ordering` reports it.

### Device forms

`#[arael(root)]` generates the GPU's assembly kernels from the same
constraint bodies as the host's assembly. A constraint has a device form
unless it:

- writes `coo` pairs, or
- reads a value Rust code computes: a `compute =` field, a component's
  field computed in its `update` (a stored value inside a component
  among them), or a `#[arael::function]` with a numeric body.

Everything else has one: the rotation, pose, direction and heading
parameters, `symbolic =` and `deriv =` fields, parameters' stored
values, every built-in function of arael-sym, guards, robust losses,
index selects and `#[arael::function]`s with a symbolic body. One
difference: where the host's index select panics on an index outside its
arms, the GPU's values are NaN.

`LmProblemInternals::device_unsupported` says why a root has no device
form, and so does a solve's report (`GpuWork::Solves(HostAssembly::NoDeviceForm(why))`).

### What ran on the GPU -- `LmResult::gpu`

`LmResult::gpu` (on a solver, `SparseFaer::gpu_report`) is a `GpuReport`
of what the solve did on its GPU; `None` for a solve without one. It
prints on one line:

```rust,ignore
if let Some(gpu) = &r.gpu {
    println!("gpu: {gpu}");
    // gpu: cuda:0 NVIDIA GeForce RTX 4090 (sm_89), wide f64 -- assembly, damped solves and the LM loop on the device, f64 factor
    // gpu: cuda:0 -- not used: the solve took a route that does not run on a device
}
```

| Field | Meaning |
|---|---|
| `backend`, `index` | the runtime and the device, as the `GpuDevice` named them |
| `name`, `target` | the device's name from its driver, and what the kernels were compiled for (`sm_89`, `gfx1036`); empty when the GPU was not used |
| `integrated` | the GPU shares the host's memory: an integrated GPU, or the CPU device. False when the GPU was not used |
| `wide` | the wide scalar, f64 or df32 (see [Precision](#precision)) |
| `solve` | `Some(GpuSolve::Factor { f32 })` with whether the factor is f32, `Some(GpuSolve::ConjugateGradients)`, or `None` when the GPU was not used |
| `f32_system` | the system the GPU formed and solved is f32 (see the table under [Precision](#precision)); otherwise it is in the wide scalar |
| `work` | how much of the solve ran on the GPU (`GpuWork`, below) |
| `reserved_bytes` | the GPU memory the `GpuDevice`'s state holds, in use or free |
| `buffers_created` | the GPU buffers the `GpuDevice`'s state has allocated over its life; unchanged across a solve when the solve allocated nothing |

`GpuWork` is one of:

- `Loop`: the assembly, the damped solves and the LM loop's vectors.
- `Solves(why)`: the damped solves; the host assembled and uploaded the
  Hessian's values for each. `why` is a `HostAssembly`: `Asked`
  (`with_gpu_assembly(false)`), `NoDeviceForm(reason)` or `NoF64` (an
  f64 root on a GPU without f64).
- `Unused`: the solve took a route that runs on the host (see
  [What runs on the GPU](#what-runs-on-the-gpu)); `SchurPlan` says which.

A solve also logs what it did: a warning when a GPU was given and not
used, with the route that ran instead, and under `verbose` the GPU and
its share of the work. The attempt lines then carry the GPU's part (see
[Verbose trace format](#verbose-trace-format)):

```text
[arael WARN] gpu: cuda:0 was given and is not used -- the reduced system is factored under its envelope, on the host. ...
[arael INFO] gpu: cuda:0 NVIDIA GeForce RTX 4090 (sm_89), wide f64 -- assembly, damped solves and the LM loop on the device, f64 factor
[arael INFO] 2/0: 273874->250676 / 23198.2, lambda=3.33333e-5 (step=9262, gpu=9201)
```

`LmResult::report` prints the report as a `gpu` block (see
[Reporting a solve](#reporting-a-solve----lmresultprint)).

### Observers and drivers on a GPU

With the LM loop on the GPU (`GpuWork::Loop`) the vectors stay there, so
an observer and a lambda driver do not get them unless the config asks:

| Field | Off | On |
|---|---|---|
| `LmConfig::device_observer_params` | `LmIter::params` is empty | the parameters are read back after each accepted step |
| `LmConfig::device_driver_vectors` | `grad`, `diagonal` and `delta` of `LambdaState` and `LambdaStep` are empty | they are read back for each attempt |

Each read costs a copy from the GPU. `LambdaStep::predicted_reduction`
works either way, and it is all the built-in drivers use.

```rust,ignore
let cfg = LmConfig::default().with_device_observer_params(true)
    .with_observer(|it: &LmIter<f64>| { draw(it.params); ControlFlow::Continue(()) });
```

### Precision

The GPU computes an f32 root's values in f32, as the host does. The rest
is in the GPU's wide scalar, chosen with `GpuDevice::with_wide`:

- `GpuWide::F64`, the default on a GPU with f64;
- `GpuWide::Df32`: two f32 whose sum is the value, about 48 bits of
  significand, in f32 arithmetic only. For GPUs whose f64 arithmetic is
  slow, which most consumer cards are. It is the only choice on Metal.

In the wide scalar are an f64 root's values (with df32 they compute in
48 bits instead of the host's 53), an f32 root's cost with `cost_f64`, a
`MixedEulerAngleParam`'s reference, the `with_wide` scopes of constraint
bodies, the norms the stopping rules read and the refinement's residual.
The GPU sums without atomic operations, in trees: `cost_kahan`,
`grad_kahan` and `grad_f64` have no effect on its assembly.

The system the GPU forms and solves (the reduced one, or the whole one)
and its factor are, by solve:

| Solve | System | Factor |
|---|---|---|
| f32 | f32 | f32 |
| f32, `MixedPrecision` | wide scalar | f32, solves refined against the system |
| f64 | wide scalar | f64 (not on Metal) |
| f64, `MixedPrecision` | wide scalar | f32, solves refined against the system |
| conjugate gradients, f32 | f32 | none |
| conjugate gradients, f64 | wide scalar | none |
| conjugate gradients, f64, `MixedPrecision` | f32 (the refinement settings are unused) | none |

`MixedPrecision` with `max_refinements: 0` on an f32 solve forms the
system in the wide scalar and refines nothing. `GpuReport::f32_system`
and `GpuReport::solve` say which ran. With an f32 root, the f32 factor
and df32, nothing on the GPU runs in f64:

```rust,ignore
SparseFaerOptions::default()
    .with_mixed_precision(Some(MixedPrecision::default()))
    .with_gpu(Some(GpuDevice::cuda(0).with_wide(GpuWide::Df32)))
```

Metal has no f64 (`GpuDevice::has_f64`): its wide scalar is df32, an f64
solve that factorizes needs `MixedPrecision` (without it the solve fails
with `SetupError::SolverUnavailable`), and an f64 root assembles on the
host.

### Conjugate gradients on a GPU

With `with_iterative_schur` and a GPU, the GPU forms the reduced system,
factors its diagonal blocks for the preconditioner and iterates, with
the same `CgParams` and stopping rule as on the host. It holds no
factor. It pays where factorizing the reduced system is most of a damped
solve, as on a bundle adjustment of many hundred cameras. The step is
inexact.

The iterations are sent to the GPU in batches, sized from the solve
before, and the host waits once per batch; the verbose line counts them
(see [Verbose trace format](#verbose-trace-format)). Under
`MixedPrecision` an f64 solve forms and solves the reduced system in f32
on the GPU, while its assembly, costs and step stay f64. On the host,
conjugate gradients ignore `MixedPrecision`, with a warning.

```rust,ignore
SparseFaerOptions::default()
    .with_policy(SchurPolicy::Force)
    .with_iterative_schur(CgParams::default())
    .with_gpu(GpuDevice::detect())
```

### Failures

A GPU whose runtime is not compiled in fails the solve with
`SetupError::SolverUnavailable`, the parameters untouched, as other
optional backends do. A GPU that fails during the solve, setup included,
fails it with `SolveFailureKind::Device` (see
[Solve failures](#solve-failures)).

### Driving the solver by hand

A program that drives the solver through `LmSolver` (`compute`,
`extract_diagonal`, `solve_damped`) in place of `lm_solve` keeps the
assembly and the damped solves on the GPU too. Each `compute` reads the
gradient and the diagonal back, and each `solve_damped` damps with the
diagonal and the scale it is given, solves on the GPU and reads the step
back:

```rust,ignore
let mut solver = SparseFaer::<f64>::from_options(&SparseFaerOptions::default().with_gpu(Some(GpuDevice::cuda(0))));
let mut ctx = arael::Context::new();
let mut matrix = solver.new_matrix(n);
let cost = solver.compute(&mut model, &x, &mut grad, &mut matrix, &mut ctx)?;
solver.extract_diagonal(&matrix, &mut diagonal);
// damping: diagonal[i] + lambda * scale[i], here with the diagonal as the scale
let solved = solver.solve_damped(n, &mut matrix, &diagonal, &diagonal, lambda, &grad, &mut step)?;
```

### The kernels: compiling and caching

The GPU runs C++ kernels: the solver's own (the arael-gpu crate) and the
model's assembly, generated by `#[arael(root)]`. They are compiled for
the GPU at first use, by NVRTC on CUDA and hipRTC on HIP, and kept in a
disk cache found again by their source, the GPU and the compiler's
version. So only a program's first run on a machine compiles, and a
rebuilt program whose kernels did not change compiles nothing. Metal
keeps its own cache instead.

| Variable | Meaning |
|---|---|
| `ARAEL_KERNEL_CACHE` | the cache's directory, or `off`. Unset, it is `arael/kernels` in the user's cache directory (`~/.cache/arael/kernels` on Linux). A program can set it with `arael_gpu::set_kernel_cache` (crate arael-gpu) before its first GPU solve |

HIP runs on AMD GPUs with 32-lane and with 64-lane wavefronts; the
kernels are tuned for 32 lanes.

### Reading the kernels -- `cargo arael kernels`

`cargo arael kernels` writes the C++ a crate's roots generate, for
reading what the GPU runs for a constraint. It needs no GPU. It builds
the crate (with `--all-targets`; arguments after `--` replace that),
empties its output directory and writes each root to
`target/arael-kernels/<crate>/<Root>.cu`, or why the root has no device
form. `--out <dir>` writes elsewhere.

```bash
cargo arael kernels
cargo arael kernels -- --features gpu-cuda --test fit
```

A file lists the root's entity types, their parameters and its data
fields, each with its model path; then each constraint's kernels, under
a banner with its source line and its attribute. Every value a kernel
reads is commented with its model path:

```cpp
// C4: a constraint of `SFrine` (tests/device_assembly.rs:3687), 2 rows
//   #[arael(constraint(hb, parent = lm, {
//       let mr2w = pose.ea.rotation_matrix();
//   ...
        const F __l0 = COL(B, F, C4_D0)[o]; // sfrine.camera_pos.x
```

The comments are part of the compiled source, so moving a constraint to
another line compiles its root's kernels again.

`ARAEL_KERNEL_DUMP=<dir>` makes every GPU assembly write the whole source
it compiles there, as `<Root>.<target>.<wide>.cu` (`SPath.sm_89.df32.cu`):
the line numbers of a compile error refer to that file.

### Checking the GPU assembly -- `ARAEL_DEVICE_CHECK`

`ARAEL_DEVICE_CHECK=1` checks the GPU assembly of every model a program
solves against the host's. The first solve of each root type with a
device form assembles that root on a GPU near its starting point, in
both wide scalars, and panics where the cost, the gradient or the
Hessian's values differ from the host's beyond rounding: 1e-10 relative
for an f64 model, 1e-4 where the root or an entity is f32. The cost is
also allowed the rounding of the host's sum.

It needs a GPU feature and runs on every solve, also one without a GPU.
The check uses device 0 of the first runtime compiled in: CUDA, else
HIP, else Metal, else the CPU device. `ARAEL_DEVICE_CHECK_LOG=<file>`
appends a line per root type: checked, with the largest differences, or
why the root has no device form. A root whose host assembly panics on
the model's data (an index outside a select's arms) is not checked at
that solve, and the log says so. Over a test suite it checks every model
the suite solves:

```bash
ARAEL_DEVICE_CHECK=1 ARAEL_DEVICE_CHECK_LOG=check.log cargo test --features gpu-cuda
ARAEL_DEVICE_CHECK=1 cargo test --features gpu-cpu   # without a GPU
```

### Staging

The GPU assembly copies the model's data to the GPU: the structure once,
the values every solve. By default (`GpuStaging::Pinned`) it writes them
into pinned host memory, whose copies run while the host goes on.
`GpuStaging::Pageable` writes them into ordinary memory, which the
runtime copies itself, for a platform where copies from pinned memory
are slow:

```rust,ignore
.with_gpu(Some(GpuDevice::cuda(0).with_staging(GpuStaging::Pageable)))
```

## Where the log goes

`arael::log` sets the level and the destination of every `info!` / `warn!` /
`error!` arael emits: the `verbose` trace, the backend's report of what it
chose, the warnings from saturated guards, and on a GPU the line
`GpuDevice::detect` logs and the warning for a GPU given and not used.

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
r.pretty_print();   // color and glyphs, to stdout

let s: String = r.report();          // the same text print() writes
let s: String = r.pretty_report();   // the same text pretty_print() writes
let s: String = r.render(Style { color: true, unicode: false });  // pick both
println!("{r}");    // Display is report()
```

`report()` is pure ASCII with no escape sequences, so it is safe in a log or a
file. `pretty_report()` carries ANSI color and box glyphs and is for a terminal.
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
            fill ratio 0.45, ordered by Amd, factored by the block supernodal Cholesky
```

The timing block appears only when `gather_timing` was set, a threads block
when either half had more than one thread, and the backend line only when the
backend reported a plan. A solve given a GPU adds a `gpu` block last (see
[What ran on the GPU](#what-ran-on-the-gpu----lmresultgpu)), and with its loop
on the GPU the timing rows are the host's and the GPU's sides of it (see
[On a GPU](#on-a-gpu----lmtiminggpu)).

## Profiling a solve -- `LmTiming`

Set `gather_timing: true` to find out where a solve spends its time. The
result's `timing` is then `Some(LmTiming)` (otherwise `None`, and the clock
is never read):

```rust,ignore
pub struct LmTiming {
    pub total: Duration,          // whole solve
    pub analysis: Duration,       // the backend's one-time structural work
    pub setup: Option<SetupTiming>,   // ...and where it went (the sparse backend)
    pub gpu: Option<GpuTiming>,   // the GPU's own time, on a GPU -- see below
    pub assembly: Duration,       // residual + Jacobian + Hessian, all iters
    pub first_assembly: Duration, //   ...of which the first
    pub linear_solve: Duration,   // damped numeric factorization + solve, all attempts
    pub first_linear_solve: Duration,  // ...of which the first
    pub cost_eval: Duration,      // trial-point cost (residual only)
    pub first_cost_eval: Duration,
    pub advance: Duration,        // post-step re-centering
    pub first_advance: Duration,
    pub covariance: Duration,     // the covariance the config asked for, after the solve; not in total
    // plus a *_count for each phase

    pub steps: Vec<LmStep>,       // the per-iteration timeline -- see below
}
```

### What the setup costs -- `LmTiming::analysis`

`analysis` is the backend's one-time work on the structure, run inside
the first `compute`: finding the sparsity pattern and each block's place
in it, finding the blocks to marginalize, weighing the Schur reduction,
ordering, and the symbolic factorization. `assembly` is the model's
residual and Jacobian work, on every iteration including the first; a
backend reports how much of its `compute` was assembly and the solver
takes the remainder, so the two do not overlap. `linear_solve` holds
only numeric factorizations and solves.

```text
    analysis          0.08 ms   14.4%  ###.................    1 call
    assembly          0.05 ms   10.0%  ##..................   10 calls, first 0.01 ms
```

`LmTiming::setup` is where the analysis went, from the sparse backend
(`None` from the others): a `SetupTiming` of the parts it times. A part a
route does not have is zero, and they add up to at most `analysis`.

| Part | What it times |
|---|---|
| `blocks` | finding the model's blocks and the ones to marginalize |
| `structure` | the Hessian's block pattern, the blocks' positions in it and, when reducing, the reduced system's pattern |
| `gpu_open` | on a GPU, opening it: the first solve through a `GpuDevice` value creates its state, later ones nothing |
| `ordering` | ordering the system that is factorized |
| `weighing` | pricing the reduced route against the whole system; zero when the comparison was skipped |
| `factor` | the symbolic factorization and the factor's buffers; on a GPU, the GPU factor's plan and kernels |
| `gpu_plan` | on a GPU, setting the reduction up there |
| `gpu_layout` | on a GPU that assembles, setting the GPU assembly up: the model's data written for the GPU, and its kernels |

The report prints the parts under the analysis row, with what is left of
the analysis as `other`:

```text
    analysis         17.95 ms   25.8%  #####...............    1 call
                  blocks 0.85, structure 7.45, ordering 1.46, factor 1.43, gpu plan 2.17, gpu layout 4.58, other 0.01 ms
```

### On a GPU -- `LmTiming::gpu`

`LmTiming::gpu` is the GPU's own time in a solve, a `GpuTiming`; `None`
without a GPU. It is needed because the host's phases show waits, not
work: with the assembly on the GPU the host queues a linearization and
goes on, and waits once per attempt, so `assembly` and `cost_eval` read
near zero and `linear_solve` holds the GPU's assembly and trial cost with
its solve.

| `GpuTiming` field | Meaning |
|---|---|
| `assembly`, `assembly_count` | the linearizations on the GPU; zero when the host assembles (`GpuWork::Solves`) |
| `attempts`, `attempt_count` | the damped attempts on the GPU; when the host assembles, from the upload of the Hessian's values |
| `cg` | the part of `attempts` spent in conjugate gradients, the waits between their batches included; zero for a solve that factorizes |
| `cg_rounds` | the batches of conjugate gradient iterations, each a wait for the GPU |

With the loop on the GPU (`GpuWork::Loop`) the report names the host's
rows for what the host does there:

| Row | Field | What the host did |
|---|---|---|
| `queueing` | `assembly` | queued the linearizations, and waited for the first |
| `device wait` | `linear_solve` | waited for the GPU, once per attempt |
| `on the device` | `LmTiming::gpu` | the GPU's own time |

and leaves out `cost eval` and `advance`, which the GPU does:

```text
  time      69.59 ms
    analysis         17.95 ms   25.8%  #####...............    1 call
                  blocks 0.85, structure 7.45, ordering 1.46, factor 1.43, gpu plan 2.17, gpu layout 4.58, other 0.01 ms
    queueing          0.75 ms    1.1%  ....................   10 calls, first 0.68 ms
    device wait      50.57 ms   72.7%  ###############.....   19 calls, first 2.78 ms
    on the device    50.62 ms  assembly 3.16 ms in 10 calls, attempts 47.46 ms in 19 calls
```

With the assembly on the host the rows keep their names, and `on the
device` has the attempts alone. With conjugate gradients it adds their
part of the attempts:

```text
    on the device    52.12 ms  assembly 4.80 ms in 9 calls, attempts 47.33 ms in 16 calls, of them conjugate gradients 30.12 ms in 23 rounds
```

Timing a GPU solve adds a little GPU work per attempt; an untimed solve
adds none. `GpuTiming` does not split an attempt into its stages
(reduction, factorization, solve, back-substitution, trial cost). For the
time of each kernel, trace the solve: build with the `gpu-trace` feature
and set `ARAEL_GPU_TRACE=<file>` and `ARAEL_GPU_TRACE_TIME=1`. Every GPU
operation of the process is then written to the file, each kernel launch
with its time on the GPU. The repository's `tools/devmap` shows a trace
by phase, stage and kernel. Timed launches run one at a time, so a timed
trace gives the kernels' times, not the solve's:

```bash
# a program that solves on a GPU, built with --features gpu-cuda,gpu-trace
ARAEL_GPU_TRACE=solve.trace ARAEL_GPU_TRACE_TIME=1 ./my_program
tools/devmap/gen/devmap.py solve.trace --id my-case --title "my model, f32, cuda"
```

### The timeline -- `LmTiming::steps`

One record per attempt, damping retries included, so
`steps.len() == LmResult::iterations`. The totals add a rejected attempt and
an accepted one together; the timeline keeps them apart. With the loop on a
GPU the times are the host's, as in the totals: `assembly` is the queueing
and `linear_solve` the wait.

```rust,ignore
pub struct LmStep {
    pub iter: usize,          // 1-based, counts retries -- matches the verbose trace
    pub inner: usize,         // retry index at this linearization; 0 = first attempt
    pub accepted: bool,
    pub factorization_failed: bool,   // damped Cholesky said not positive definite

    pub lambda: f64,          // the damping this attempt used
    pub cost: f64,            // before
    pub new_cost: f64,        // at the trial point; NaN if the factorization failed
    pub step_norm: f64,       // |delta|_2 -- how far it proposed to move
    pub grad_max: f64,        // max|g_i| here; constant across the retries that share it

    pub time: Duration,           // the whole attempt, without `assembly` and `analysis`
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


Every phase records `{total, first, count}`. Each `first_*` is part of its
phase total; it is recorded apart so a steady-state mean can leave the first
call out. The one-time structural work is not in any first call: it is
`analysis`. Phase times are wall clock, whatever the thread count, and cover
only the work inside each phase, so they sum to slightly less than `total`.

For the steady-state cost per iteration, use the `mean_*` methods:
`mean_assembly()`, `mean_linear_solve()`, `mean_cost_eval()`,
`mean_advance()`. Each returns `(phase - first) / (count - 1)` as an
`Option<Duration>`, or `None` when only the first call ran.

## Graduated optimisation

When the residual groups of a model have very different weights (inverse
sigmas), start with the tight constraints loose and tighten them over
several solves. The cost surface is smoother at low weights, so from a
noisy initial estimate this converges more reliably than solving the
tight problem at once.

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
    let result = path.solve_sparse(&cfg)?;
    // ...
}
```

See `loc_demo.rs` / `slam_demo.rs` for the full pattern. Keep the
scale on a single root field and set it to the desired absolute
value each pass -- the constraint body reads the current value, so
the sigma you see is always the sigma in effect.

## Pre-solve centering

For models with root-level rigid-transform parameters (translation +
rotation applied to every entity), a coarse pass that freezes all
per-entity parameters and optimises only the global ones can absorb
systematic offset in the initial estimates before the main sweep
begins. See `Path::optimise_center` in
[loc_global_demo.rs](../examples/loc_global_demo.rs):

1. Set `pose.pos.optimize = false` / `pose.ea.optimize = false` for every pose.
2. Solve (only root-level params vary).
3. Bake the result into poses via `recenter()`, resetting globals to identity.
4. The Param constructors in `recenter` reset `.optimize = true` automatically.

## Damping-schedule drivers

A `LambdaDriver` decides the damping: the LM loop asks it for every
lambda and reports each attempt's outcome as a `LambdaStep { lambda,
cost, new_cost, grad, diagonal, delta, predicted }`. The gradient, the
Gauss-Newton Hessian diagonal and the step let a driver compute its own
measures of how well the model predicted the step;
`LambdaStep::predicted_reduction()` is the predicted cost reduction,
from which the Nielsen driver takes its gain ratio. `start` is called
after the first assembly with a `LambdaState { cost, grad, diagonal }`,
so the initial lambda can be chosen from the problem's scale. With the
LM loop on a GPU, `grad`, `diagonal` and `delta` are empty unless
`LmConfig::device_driver_vectors` is set (see
[Observers and drivers on a GPU](#observers-and-drivers-on-a-gpu)).

- `DefaultLambdaDriver` -- the classic fixed-multiplier schedule and
  what every plain entry point uses: divide lambda by 5 on acceptance
  (clamped to `LmConfig::lambda_floor`), multiply by 10 on rejection or
  factorization failure, give up when a rejection would pass 1e10.
- `NielsenLambdaDriver` -- the gain-ratio adaptive schedule: on
  acceptance lambda scales by
  `max(1/3, 1 - (2 rho - 1)^3)`, on rejection it multiplies by an
  escalating `nu` (2, 4, 8, ... reset to 2 by the next acceptance).
  Use it on strongly nonlinear problems, such as bundle adjustment,
  where dividing lambda by a constant after every acceptance leads into
  the next rejection.

Custom drivers implement the four-method trait (`start`, `accepted`,
`rejected`, which receives the `LambdaStep`, and `factorization_failed`,
which receives the current `LambdaState`). All three step hooks return
`Option<T>`, and `None` always stops the solve. Which status comes back
says whether a step survived:

| Hook returns `None` | Status | The step |
|---|---|---|
| `accepted` | `DriverTerminated` | kept: it is already in the parameters and the cost |
| `rejected` | `LambdaCeiling` | none was produced; the last accepted one comes back |
| `factorization_failed` | `LambdaCeiling` | none was produced; the last accepted one comes back |

`accepted -> None` is the hook for a stopping rule the config cannot
express: a step-norm test (`step.delta`), a gradient-norm test
(`step.grad`; both need `device_driver_vectors` with the loop on a GPU),
an external deadline, a good-enough cost. Besides an observer's `Break`
it is the only way to stop on an accepted step, and it overrides
`min_iters`. The built-in schedules never use it.

A driver is a `#[derive(Clone)]` type; attach it to the config with
`LmConfig::with_driver(...)` and every solve entry point (`lm_solve`,
`solve_sparse`, ...) picks it up from `config.driver`.

## Iteration observer

Watch a solve without touching the damping schedule: an `LmObserver`
rides on the config and is called once per damped attempt -- accepted,
rejected, and factorization-failed alike -- with an `LmIter`:

```rust,ignore
pub struct LmIter<'a, T> {
    pub iter: usize,             // 1-based attempt counter (= LmResult::iterations)
    pub inner: usize,            // retry index within this linearization
    pub accepted: bool,
    pub factorization_failed: bool, // no trial point; new_cost is NaN
    pub cost: T,                 // best accepted cost before this attempt
    pub new_cost: T,             // this attempt's trial cost
    pub lambda: T,
    pub accepted_total: usize,
    pub params: &'a [T],         // current best state; copy out what you keep
}
```

Return `ControlFlow::Break(())` to stop the solve: the current best
state is kept and the result reports `LmStatus::ObserverTerminated`
(an `Ok` termination, `is_success()`). Closures qualify:

```rust,ignore
use std::ops::ControlFlow;

// Progress + cancellation for a UI thread.
let cancel = Arc::new(AtomicBool::new(false));
let c = cancel.clone();
let cfg = LmConfig::conservative().with_observer(move |it: &LmIter<f64>| {
    if it.accepted {
        let _ = tx.send(it.params.to_vec());   // snapshot for redraw
    }
    if c.load(Ordering::Relaxed) { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
});
```

`params` borrows the solver's state for the duration of the call. To
draw the model's geometry, `deserialize` a snapshot into a display copy
of the model; the solver owns the model during the solve. With the LM
loop on a GPU, `params` is empty unless `LmConfig::device_observer_params`
is set (see [Observers and drivers on a GPU](#observers-and-drivers-on-a-gpu)).
Observers are cloned per solve like the driver, so a shared config
starts every solve with a fresh copy; state that must survive the solve
lives behind a channel, `Arc`, or `Rc` handle.

The observer is independent of the `LambdaDriver`: the observer watches
or cancels, the driver schedules damping, and the two combine without
wrapper types.
