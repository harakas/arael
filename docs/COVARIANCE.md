# Parameter Covariance

After a solve, the covariance of the estimated parameters says how certain the
solution is. arael recovers it from the same Hessian the solver already builds,
without ever forming the dense inverse. This document covers the API, the three
assembly modes and when to pick each, the block types you can query, and the
gauge requirement.

## The convention

At the solution the parameter covariance is `Sigma = (J^T J)^-1`. arael assembles
`H = 2 J^T J` (the cost is `sum r^2`, so `add_residual` carries a factor of 2),
so `Sigma = 2 H^-1`. Covariances come back in **local tangent coordinates**:
rotation parameters are already minimal 3-DOF retractions (a small delta around a
re-centered reference rotation), so there is no manifold projection to undo -- a
6-DOF pose block is `[translation, rotation-delta]`.

## The API

Bring the trait into scope and assemble at the solution:

```rust,ignore
use arael::covariance::{Covariance, CovMode};

model.solve_sparse(&cfg)?;                       // solution written back into the model
let cov = model.assemble_covariance(CovMode::AllMarginals)?;
```

`assemble_covariance(&mut self, mode) -> Result<CovAssembly, CovError>` is on the
`Covariance` trait, which every `#[arael(root)]` model implements. It re-linearizes
`H` at the current parameters and prepares it per `mode`. The returned
`CovAssembly` is an owned value -- querying it does not borrow the model, so the
two lines above borrow-check cleanly.

Query per-entity blocks by passing the entity itself. Any `Model` reports its
live-parameter span (`collect_param_blocks`), so a single pose, a single
landmark, or a whole `refs::Vec` collection are all valid arguments:

Each query returns a `Result`: `Err(CovError::NotPositiveDefinite)` for a
singular (unobservable) block, `Err(CovError::UnsupportedQuery)` for a
query the assembled mode cannot answer.

```rust,ignore
let sd = cov.std_dev(&model.poses[0])?;            // Vec<f64>, one per scalar param
let m  = cov.marginal_cov(&model.landmarks[3])?;   // DMatrix<f64>, its covariance block
let c  = cov.conditional_cov(&model.poses[0])?;    // DMatrix<f64>, others held fixed
let x  = cov.cross_cov(&model.poses[0], &model.landmarks[3])?; // joint off-diagonal
let n  = cov.dim();                                // number of optimized scalars
```

## Modes

`CovMode` is chosen once, at assembly, and decides how much work is done up front.

### `PerQuery`

Factor `H` once; answer each query by solving for that entity's columns of the
inverse. faer picks a supernodal factor where it pays off. The factorization is
shared across queries, so cost is roughly one factor plus one triangular solve
per queried entity. **Use it for a handful of entities.**

### `AllMarginals`

Also run a *selected inverse* up front: a block Takahashi recursion over the
supernodal factor (BLAS-3 dense kernels) that computes every covariance entry
inside the factor's sparsity pattern -- which always includes every diagonal
block. Every marginal, and every cross block that lands in-pattern, then becomes a
lookup. **Use it when you want many or all marginals** (an uncertainty ellipse per
pose, a covariance per landmark). It computes them all in one pass at a cost that
does not grow with how many you read back.

### `TriDiagonal`

For a block-tridiagonal `H` -- a localization pose chain against a fixed map, with
no loop closures, so consecutive poses couple and nothing else does. It runs a
forward Schur pass over the band with **no factorization at all**: the last pose's
covariance is `2 S_last^-1`, free once the forward pass reaches the end (and the
solve already needs that pass). Querying an interior pose triggers a single
backward pass, cached, after which interior marginals are `2 (S_i + R_i - D_i)^-1`.
Assembling errors with `NotTriDiagonal` if any off-band block couples non-adjacent
entities (a loop closure or a free landmark) -- use `PerQuery` or `AllMarginals`
there. This backend answers only single-block entity queries: `cross_cov`, and
a query spanning several blocks or none, error with
`CovError::UnsupportedQuery`. A singular (unobservable) block errors with
`CovError::NotPositiveDefinite`.

The [loc benchmark](../benchmarks/loc/README.md) recovers the last pose (the
localization query) in constant time as the trajectory grows, where the general
modes and the other libraries scale with it.

## How it factorizes

`assemble_covariance_with(mode, &opts)` spells out what
`assemble_covariance(mode)` leaves to the defaults. The covariance is the same
either way; `CovOptions` changes only what it costs to produce.

```rust
use arael::covariance::{CovMode, CovOptions, CovOrdering, Covariance};

let opts = CovOptions::auto().with_ordering(CovOrdering::NestedDissection);
let cov = model.assemble_covariance_with(CovMode::PerQuery, &opts)?;
```

**`ordering`** picks the elimination order. `Auto` (the default) prices minimum
degree against nested dissection over the model's block graph and keeps
whichever factors in fewer flops -- the same determination the solver makes for
a solve. `Amd` and `Natural` force the choice; `NestedDissection` forces
dissection, which is what a trajectory that revisits wants (a loop closure, a
figure-8 crossing), since poses far apart in the ordering are coupled there and
minimum degree has no separator to find.

Pricing is not free: it builds a symbolic analysis per candidate and discards
the loser. Where minimum degree wins regardless, naming `Amd` skips that and
halves the setup. A block graph of many small blocks is the case -- a bundle
adjustment with tens of thousands of 3-DOF points, where minimum degree wins
every Ladybug dataset. A trajectory that revisits is the opposite case, and
there the comparison pays for itself.

Ordering over the block graph rather than `H`'s scalar columns is the same
coupling with the entity sizes divided out: a 6-DOF pose against a 3-DOF
landmark is one edge there and 18 scalar ones in `H`.

**`block_supernodal`** factorizes in block form with arael's supernodal
Cholesky instead of faer's scalar one, which also skips the scalar triplets and
the scalar CSC. `AllMarginals` ignores it and stays on the scalar factor: its
selected inverse reads faer's supernode panels. `CovAssembly::took_block_route`
reports which route an assembly actually took.

A model with no block structure, or whose entities are all one scalar wide,
declines both -- there is nothing to divide out, and the scalar path is the
better answer.

## Block types

- **`marginal_cov`** -- the entity's covariance with every *other* parameter
  integrated out. This is the covariance you usually want: it folds in the
  uncertainty of the variables the entity couples to.
- **`conditional_cov`** -- the entity's uncertainty with every other parameter held
  fixed, `2 (H_ee)^-1`, the inverse of its own information block. Never larger than
  the marginal; `O(dof^3)` with no factor solve. An entity with no self-information
  yields infinities.
- **`cross_cov(a, b)`** -- the off-diagonal `A x B` block of the joint covariance,
  the correlation between two entities. Available in `PerQuery` and `AllMarginals`.
- **`std_dev`** -- the square root of the marginal diagonal, one value per scalar
  parameter. Convenience over `marginal_cov`.

## The gauge

`H` must be non-singular. If the problem has unobservable directions -- a
free-gauge pose graph (absolute position and heading undetermined), a bundle
problem defined only up to a similarity -- `H` is rank-deficient and
`assemble_covariance` returns `CovError::NotPositiveDefinite`. Fix the gauge
before recovering covariance: anchor a pose (hold its parameters constant), or add
a prior. The covariance is then relative to whatever the gauge fixes.

Other errors: `Empty` (the model has no optimizable parameters) and
`NotTriDiagonal` (above).

## Cost

The covariance modes are benchmarked against Ceres, GTSAM and g2o -- how the cost
scales from one marginal to all of them, on real problems -- in the covariance
section of the [slam](../benchmarks/slam/README.md),
[loc](../benchmarks/loc/README.md) and [bal](../benchmarks/bal/README.md)
READMEs. In short: `PerQuery` wins for a few, `AllMarginals` for many or all, and
`TriDiagonal` for the localization last-pose query.
