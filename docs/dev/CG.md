# Iterative Schur: state and what is left

Conjugate gradients on the Schur-reduced camera system, instead of factorizing
it. `SchurSolve::Iterative` on `SparseFaer`, off by default. This file is the
working plan; the user-facing description is in the crate docs.

## Where it stands

`arael-faer::cg` runs preconditioned CG over a `SparseBlockColMat`, with block
Jacobi (the Cholesky factor of each diagonal block) as the preconditioner and
f64 reductions whatever the storage type. `bsc::mul_symmetric_upper` is the
matrix-vector product. The reduction, the reduced right-hand side, the damping
and the back-substitution are the existing Schur code, unchanged.

Measured on the bundle benchmark (`BAL_CG_TOL=1e-3`, nested dissection):

| dataset | schur full-iter | schur-cg full-iter | schur peak | schur-cg peak |
|---------|----------------:|-------------------:|-----------:|--------------:|
| Ladybug-49 | 11.17 ms | 18.86 ms | - | - |
| Ladybug-372 | 199.38 ms | 115.28 ms | 356.3 MB | 308.3 MB |
| Ladybug-1723-clean | 1801.38 ms | 292.62 ms | - | - |

The crossover follows how much of the iteration the factorization is:
`schur_stats` measures that as 12% at 49 cameras, 43% at 138, 81% at 372, 96%
at 1723. Below roughly a hundred cameras CG has nothing to win.

Against Ceres on Ladybug-372: arael is faster per iteration (115.3 vs 151.9 ms)
and reaches a lower cost, but takes **308.3 MB against Ceres's 191.0**. Memory
is the gap, and the next two sections are why.

## 1. The factorization machinery CG never uses

`compute()` builds the reduced route the same way whatever `schur_solve` says
(`simple_lm.rs:5504-5542`): the scalar CSC pattern of S, a values buffer for
it, a fill-reducing ordering, a symbolic Cholesky, and then `size_llt_buffers`
sizes the factor and its scratch. The CG path reads none of it -- it multiplies
by the block form directly.

At Ladybug-372 (S is 3348 wide, 39.8% dense, 2.23M stored scalars in the upper
triangle):

| allocation | size | used by CG |
|------------|-----:|-----------|
| `l_vals` (the factor) | 73.6 MB | no |
| `s_vals` (S flattened to scalar CSC) | 17.8 MB | no |
| `s_row_idx` | 17.8 MB | no |
| `factor_mem`, `solve_mem` | faer scratch | no |
| S itself, block form | ~17.8 MB | yes |

That is about 109 MB of the 308.3 MB peak, against a 117 MB gap to Ceres. It
also explains the setup cost: Ladybug-1723's first iteration is 740 ms against
a 350 ms steady-state one, and the symbolic factorization alone is 164 ms there
(`schur_stats`).

**DONE.** The pattern, the ordering, the symbolic and the buffer sizing are
skipped when `schur_solve` is `Iterative`; `s` and the Schur symbolic, which CG
does need, stay. Measured at Ladybug-372:

| | peak before | peak after | 1st-iter before | after |
|---|------------:|-----------:|----------------:|------:|
| f64 schur-cg | 308.3 MB | 234.6 MB | 186.9 ms | 144.9 ms |
| f32 schur-cg | 212.2 MB | 166.9 MB | 140.8 ms | 104.4 ms |

73.7 MB on f64, which is `l_vals` (73.6 MB) and nothing else -- `s_vals` and
`s_row_idx` never showed up in `VmHWM`, so peak lands during the reduction
rather than after it. `full-iter` is unchanged, as it should be: this is setup
only. f32 now sits under Ceres's 191.0 MB; f64 is 43.6 MB above it, which is
mostly S itself.

## 2. Implicit S (stage 2)

Never form S. Each product applies `B x - E C^-1 (E^T x)` by walking the block
Hessian, which is what Ceres's `iterative_schur` does by default.

It removes S's storage and the reduction, but it does not remove the work --
it trades ONE reduction for ONE product per CG iteration, and an implicit
product is the more expensive of the two. It walks E twice (once for `E^T x`,
once for `E u`) plus a triangular solve per eliminated block, against a single
pass over S. So the trade turns on how many CG iterations a solve takes:

| | S nnz | E nnz | CG/solve | explicit | implicit |
|---|------:|------:|---------:|---------|----------|
| Ladybug-372 | 2.23M | 5.52M | ~35 | 51 + 35x1.3 = **96 ms** | 35x~3.3 = 115 ms |
| Ladybug-1723 | 9.7M | 18.3M | ~6 | 183 + 6x5 = 213 ms | 6x~11 = **69 ms** |

(matvec costs scaled from the measured 1.3 ms per S product at 372; the
implicit figures are estimates, not measurements.)

So implicit **loses on 372 and wins roughly 3x on 1723**, and the crossover is
CG-iterations-per-solve rather than problem size -- which at `tol=1e-3` is ~35
on 372 against ~6 on 1723. It is therefore a THIRD option, not a replacement
for the explicit route, and the solver would need a rule to pick between them.

**Prerequisite DONE**: the per-block Cholesky factors of the eliminated blocks
are cached in `SchurContext` by `schur_reduce`, and `schur_backsub` uses them
instead of factoring the same tiles again (`schur.rs`). That removes 47423
redundant 3x3 Choleskys per solve at Ladybug-372. An implicit product needs
those same factors per matvec, so this is the shared groundwork.

Still to build: the implicit product itself (`B x - E C^-1 (E^T x)` over the
flattened observer arrays the symbolic already carries), a block-Jacobi
preconditioner built from S's diagonal blocks without forming S, an operator
abstraction so `cg::solve` takes either form, and the rule that chooses.

## 3. The matrix-vector product

The hot loop. At 372 it is ~35 iterations per damped solve over 2.23M stored
scalars, and it went from 2.7 ms to ~1.3 ms per product when the inner loop
started accumulating in registers over slices instead of indexing twice per
element. Remaining ideas, in the order I would try them:

- **Fixed-size tile kernels.** Every tile in a BAL reduced system is 9x9. The
  reduction already does this (`schur.rs` `FIXED_SHAPES` / `gemm_sub_fixed`,
  measured 2x against the runtime-dimension loop) and the same dispatch would
  apply here. Pose graphs would want 3x3 and 6x6.
- **Parallelism.** Block-columns are independent except that both `y[rows]` and
  `y[cj]` are written, so a naive rayon split races. Per-thread output buffers
  reduced at the end, or a coloring of the block-columns, would work. The
  reduction has the same structure and is unparallelized for the same reason
  (TODO.md), so one solution serves both.
- **Skip the zero padding.** Diagonal tiles store a strictly-lower half that is
  always zero. It is skipped in the product already, but it still occupies
  cache lines and bandwidth.

## 4. Preconditioning

Block Jacobi is the floor, and it is what Ceres calls `SCHUR_JACOBI` -- so
matching it is not an advantage, only parity. Options above it:

- Ceres offers `CLUSTER_JACOBI` and `CLUSTER_TRIDIAGONAL`, which group cameras
  by visibility before inverting. More setup per solve for fewer iterations.
- `BlockJacobi::build` allocates fresh vectors on every damped solve. The
  structure never changes within a solve, only the values, so the buffers
  should be reused. Small, but it is allocation churn in the inner loop.

The forcing sequence (`(Q_i - Q_{i-1})/Q_i < eta/i`, Ceres's rule) was
implemented and **measured worse on both 372 and 1723**: fewest CG iterations
of anything tried, but 3-13 more outer steps and worse final cost. Removed. A
flat relative tolerance of 1e-3 beat it and beat 1e-6. Intermediate values are
worse than either end -- 1e-4 and 1e-5 cost 3-5x the outer steps of 1e-3, which
is deterministic and not understood. Worth a look with a per-iteration damping
trace, since it suggests the inexact step interacts with the gain-ratio driver.

## 5. Single precision

f32 schur-cg is the fastest row on Ladybug-372 (84.4 ms, 212.2 MB) and
converges there. On Ladybug-1723-clean it does not: 3.64M against f64's 771k.

That is **not** a CG problem. The plain `sparse` route, which does no reduction
and no CG, fails identically (4.53M), so the loss is upstream of all the linear
algebra -- in the assembly or the residual evaluation. All three f32 rows keep
their geometry (camera-centre RMSE 1.3e-4, inside even the tight gate) and lose
the cost, so the points or the intrinsics are what drift.

Chase that before adding precision machinery to CG. `CgOptions::restart_every`
exists for the recurrence-drift case and is untested on a problem that needs
it; the f64 CG vectors and f64 residual recompute from the original plan were
deliberately not built, because they answer a question f32 is not currently
failing.

## Open

- Ladybug-1723 needs a tighter shared termination criterion before its numbers
  are a fair race at all; no system converges there under the 1e-5 tolerances
  (TODO.md).
- `SchurPolicy::Auto` prices a factorization, so it cannot choose the iterative
  route. Leaving CG opt-in until there is a cost model for it.
- Covariance is unavailable on this route (no factor). It errors rather than
  returning something wrong, but the error could name the route.
