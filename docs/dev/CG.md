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

**DONE and measured.** `SchurSolve::IterativeImplicit`, benchmark route
`schur-cg-implicit`. On Ladybug-372 it reaches the explicit route's cost
bit-identically (225347.2179) at the same 9(16) iterations -- the operator is
exact through a whole solve -- and loses on time as predicted:

| Ladybug-372, f64 | full-iter | total ms |
|------------------|----------:|---------:|
| schur (factorize) | 206.28 | 3908.78 |
| schur-cg (explicit) | 116.92 | 1735.84 |
| schur-cg-implicit | 203.90 | 2956.07 |

Worse than the estimate above, though: ~5.5 ms per implicit product against
~1.5 ms explicit, so 3.7x rather than the projected 2.5x. Two candidates, both
untested: `schur_apply` walks S's structure for the `B x` part and so visits
fill tiles it then skips, and it calls `gemm_sub` once per observer per
product where the explicit route pays that once per solve.

On Ladybug-1723-clean, the case it was built for, it wins:

| Ladybug-1723-clean, f64 | full-iter | total ms |
|-------------------------|----------:|---------:|
| schur (factorize) | 1766.22 | 47634 |
| schur-cg (explicit) | 314.83 | 9583 |
| schur-cg-implicit | **261.12** | **8322** |

1.21x per iteration and 1.15x total over the explicit route, same cost
bit-identically at the same 23(32) iterations. So the crossover is real and
sits where the products-per-solve model put it -- but the margin is 1.21x, not
the 3x projected, because the implicit product costs more than estimated (see
above). Against Ceres iterative_schur there (393.97 ms/iter, 9269 ms) it is
1.51x per iteration and 1.11x overall.

Not yet measured: peak memory on this route, which should be below the
explicit one's since S is never allocated. Both 1723 runs above used
BAL_NO_MEM.

The pieces, all done and each pinned against the explicit route: `schur_apply`
(the product) and `schur_factor_eliminated`, checked column by column against
`schur_reduce` + `mul_symmetric_upper`; `schur_prepare_implicit` (the reduced
rhs and S's diagonal blocks), checked against `schur_reduce`'s own; `cg::solve`
over a closure rather than a matrix; `BlockJacobi::from_diagonal_blocks`.

Where the extra cost may be, if it is worth chasing: `schur_apply` walks S's
structure to place H's kept-kept tiles, so it visits fill tiles the elimination
created and then skips them -- a kept-kept-only tile list would not. And it
calls `gemm_sub` once per observer per product, where the explicit route pays
that once per solve.

## 3. The matrix-vector product

The hot loop. At 372 it is ~35 iterations per damped solve over 2.23M stored
scalars, and it went from 2.7 ms to ~1.3 ms per product when the inner loop
started accumulating in registers over slices instead of indexing twice per
element. Remaining ideas, in the order I would try them:

- **Fixed-size tile kernels -- DONE, +3%.** Widths 3, 6 and 9 take a
  specialized inner loop. Measured on Ladybug-372, three interleaved
  alternations, all in the same direction: full-iter 114.68 -> 111.22 ms.
  Much less than the 2x the reduction's GEMM kernels got, which is the first
  hint that this loop is not compute-bound.

  The row-spanning arguments must arrive as `&[T; NR]`, not `&[T]`. A const
  trip count over a runtime-length slice still bounds-checks every element,
  because nothing tells the compiler the slice is `NR` long -- that version
  measured 4% SLOWER than no specialization at all. Typing them as arrays is
  the entire difference between -4% and +3%.

- **The product looks memory-bound, so kernel work has a low ceiling.** At
  Ladybug-372 one product reads S's 2.23M scalars = 17.8 MB in ~1.3 ms, about
  13.7 GB/s, which is a plausible single-core figure for this machine.
  Consistent with it: f32 schur-cg runs 85.95 ms full-iter against f64's
  112.18 -- roughly what halving the bytes buys, and not what halving nothing
  (f32 has the same flop count) would buy.

  If that holds, the levers are bytes and passes, not kernels: f32 storage,
  a stronger preconditioner (fewer full passes over S), or not forming S.
  nano-gemm does not help here -- `schur.rs` already records that its
  function-pointer dispatch costs more than the arithmetic at these sizes,
  and with `wb == 1` there is no reuse for a microkernel to exploit anyway.
  **Worth confirming with a bandwidth probe before acting on it.**
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
