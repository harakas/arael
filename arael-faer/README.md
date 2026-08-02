# arael-faer

faer extensions. Everything here is built on faer's public API and laid out the
way it would sit in faer itself; arael depends on it, but nothing in it is
arael-specific.

The block-structured pieces a large sparse solve needs:

- **Block CSC** (`bsc`) -- sparse matrix storage over a *variable* block
  partition. A Hessian assembled from entities (6-wide poses, 3-wide points)
  is a matrix of small dense tiles, and storing it that way means one index
  lookup per tile instead of per scalar, and dense kernels inside the tile.
- **Schur complement** (`schur`) -- eliminate a set of mutually uncoupled
  blocks from a block-CSC matrix and factorize only what is left. This is the
  landmark/point marginalization that makes bundle adjustment and SLAM
  tractable, and it needs the block structure to be cheap.
- **Conjugate gradients** (`cg`) -- solve a symmetric system by repeated
  multiplication instead of factorizing it, preconditioned by the Cholesky
  factor of each diagonal block. No fill and no factor to store, at the price
  of an inexact solution; the reductions run in f64 whatever the storage type.
  The operator is a closure, so it can be a matrix that was never formed.
- **Envelope Cholesky** (`envelope`) -- factorize a block-CSC matrix in natural
  order directly in block form, fill confined to each column's envelope. A
  trajectory's Hessian, and its reduced pose system, keep a narrow one, so this
  needs no fill-reducing ordering and no symbolic phase.
- **Nested dissection** (`nd`) -- a fill-reducing ordering for matrices with no
  band and no small degrees, where minimum degree has nothing to chew on. faer
  offers AMD, natural, or a custom permutation; this computes the custom one.

## bsc -- block CSC

`SymbolicSparseBlockColMat<I>` is the structure (row/column partitions, which
tiles exist), `SparseBlockColMat<I, T>` adds the values. Upper-triangle storage
is the convention for symmetric matrices; diagonal tiles carry only their own
upper triangle.

| | |
|---|---|
| `SymbolicSparseBlockColMat::from_scalar_coords` | build the structure from scalar (row, col) coordinates and a partition |
| `SparseBlockColMat::zeroed` / `new` | allocate values for a structure |
| `block(b)` / `block_mut(b)` | a tile as a faer `MatRef` / `MatMut` -- dense, so faer's kernels apply |
| `PositionResolver` | scalar (i, j) -> offset in the value array; build a scatter map once, assemble by index forever after |
| `csc_pattern` / `csc_vals_into` / `to_csc` | hand the matrix to a scalar sparse factorization |
| `to_dense` | expand, for tests and debugging |

## nd -- nested dissection

A fill-reducing ordering. Cholesky fill spreads along paths in the matrix's
graph; nested dissection cuts them -- find a set of vertices whose removal
splits the graph in two, order each half first and the separator last, and no
fill can reach from one half to the other. Recurse.

```text
order(V) = order(A) ++ order(B) ++ S      S separates A from B
```

This is what bundle adjustment needs and minimum degree cannot give it: a 3D
point seen by k cameras makes a k-clique among them, and AMD drowns in cliques.
On the 1723-camera Ladybug problem AMD leaves 83.1M values in the factor and
faer takes 4.7 s over it; this ordering leaves 46.9M and takes 2.3 s.

`NestedDissection::of_blocks` dissects the graph of BLOCKS -- one node per
camera, not per parameter -- so a block's parameters stay contiguous and the
factor keeps its supernodes; the graph is also 9x smaller, and the ordering runs
in 50 ms instead of 916. It returns a permutation faer takes as-is
(`SymmetricOrdering::Custom`).

It is NOT a general win, and the caller must know which matrix it has:

- **banded** (a SLAM trajectory's reduced system) -- the natural order is
  already at the fill limit, and dissecting it is 3.4x SLOWER.
- **very sparse graphs** (a pose graph) -- AMD wins outright.
- **cliquey, no band** (bundle adjustment) -- this is the one.

Nested dissection normally comes from METIS, a C library -- it is what CHOLMOD
uses. faer's `SymmetricOrdering` offers `Amd`, `Identity` and `Custom` and
nothing else, so there was no Rust route to this ordering. So a pure Rust
implementation was written here; it computes the `Custom` permutation.

## schur -- Schur complement

Given `H` in block CSC and a set of blocks to eliminate, `S = Hkk - Hke Hee^-1
Hek`. The eliminated blocks must be mutually uncoupled (no tile joins two of
them), which makes `Hee` block-diagonal and lets the reduction decompose into
one independent contribution per eliminated block.

| | |
|---|---|
| `schur_symbolic(h, eliminated)` | analyse once: what S looks like, where every contribution lands. Returns `SchurSymbolic`, or `SchurError` if the set is not eliminable |
| `schur_reduce(sym, h, rhs, ctx, s, rhs_out)` | numeric: fill S and the reduced right-hand side from a (damped) H |
| `schur_backsub(sym, h, rhs, x_kept, ctx, x_full)` | recover the eliminated blocks once the reduced system is solved |
| `SchurSymbolic::{kept_size, kept_bandwidth, reduce_flops, pair_count}` | what the reduction will cost and how big S is -- free, from the symbolic pass, for deciding whether to reduce at all |
| `SchurContext` | reusable workspace across iterations; `enable_timing` breaks a reduction down by stage |
| `FIXED_SHAPES` / `has_fixed_kernel` | the tile shapes with a fully unrolled GEMM kernel. Anything else works, through the nano-gemm fallback, at about 1.2-1.4x |
| `SchurSymbolic::gemm_shapes` | which shapes a given problem needs, and how many calls each carries -- so a caller can see whether it is on the slow path |

S does not have to be formed. The same operator applied instead of built lets
conjugate gradients solve the reduced system with no S to store or factorize,
which is what a bundle problem wants once S stops fitting comfortably in a
factor:

| | |
|---|---|
| `schur_prepare_implicit(sym, h, rhs, ctx, ...)` | the reduced right-hand side and S's diagonal blocks (all the preconditioner needs), neither of which requires S. Leaves the eliminated blocks' factors in `ctx` |
| `schur_apply(sym, h, ctx, x, y)` | `y = (B - E C^-1 E^T) x` over the kept numbering, without forming S -- one pass per call, so it pays when the solve takes few enough products |
| `schur_factor_eliminated(sym, h, ctx)` | the eliminated blocks' diagonal factors on their own, when a caller wants them without a reduction |

The shapes with unrolled kernels are the ones SLAM systems actually use --
observers 3 (a 2D pose), 6 (a 3D pose), 7 (a similarity), 9 (a camera with
intrinsics); marginalized 1 (inverse depth), 2 (a 2D point or a bearing), 3 (a
3D point), 4 (a 3D line or a 2D segment) -- cross-checked against g2o's vertex
dimensions and GTSAM's variable dimensions. The list also carries those widths
one column wide: the reduction's right-hand side and back-substitution both
update a single column per observer, which runs as often as there are
observations. A shape outside the list is not an error, only slower, and
`gemm_shapes` will say so.

When every observer of an eliminated block has the same width and storage
orientation -- the ordinary case, since the entity marginalized out is usually
seen by one kind of entity -- the whole triangular pair loop runs under one
shape dispatch, with the tile shape a compile-time constant throughout.

The caller factorizes S itself (`csc_pattern` + faer's sparse Cholesky, or any
other solver), then calls `schur_backsub`. See `arael::simple_lm::SparseFaer`
for the whole loop.

## cg -- conjugate gradients

Solve `A x = b` for symmetric positive-definite `A` by repeated multiplication:
no factorization, so no fill and no factor to store. The step it returns is
inexact by construction -- it stops when the residual has fallen far enough --
which suits a damped solve, where the step is a trial anyway.

| | |
|---|---|
| `solve(apply, m, b, x, opts, w)` | `apply` is the operator as a closure, so `A` can be a matrix held in block CSC (`mul_symmetric_upper`) or one that is never built at all |
| `BlockJacobi::build` / `from_diagonal_blocks` | the preconditioner: the Cholesky factor of each diagonal block, from a matrix or from the blocks directly |
| `CgOptions` / `CgStats` | tolerance and iteration cap in, iterations and final residual out |
| `CgWorkspace` | the four vectors, reused across iterations |

The reductions run in f64 whatever the storage type. Paired with `schur_apply`
this solves the reduced camera system without ever forming it, which is what
bundle adjustment wants once the camera system stops fitting comfortably in a
factor.

## envelope -- envelope (profile, skyline) Cholesky

Given a symmetric positive-definite block-CSC matrix in natural order, factor
`R^T R = S` in block form. Cholesky preserves each column's envelope, so the
factor's pattern follows from the matrix: no fill-reducing ordering, no symbolic
analysis, no scalar-CSC round trip. Works on the whole Hessian (a pose graph or
localization system) or on a reduced Schur system -- anything left in its
natural order.

| | |
|---|---|
| `EnvelopeSymbolic::new(sym)` | analyse once: the factor's envelope pattern and the map back to the matrix's values. The envelope falls out of the block structure; the caller supplies nothing |
| `EnvelopeSymbolic::with_panel_width` | override the panel the factorization works in; `None` derives it from the mean envelope height |
| `envelope_factorize(sym, s, factor)` | numeric: `R^T R = S` into a factor buffer. Left-looking by block column, reusing schur's unrolled tile kernels |
| `envelope_solve(sym, factor, rhs)` | solve `S x = rhs` in place from the factor |
| `envelope_flops(sym)` | what the factorization will cost, in one pass over the pattern -- for deciding whether to take this route at all |
| `EnvelopeError` | the matrix was not positive definite |

It holds less than an ordered sparse factor while the envelope stays narrow, and
more once it widens, so the choice is worth pricing. `arael::simple_lm::EnvelopeMode`
prices it for the reduced Schur system; `SparseFaer::with_narrow_band` takes the
whole Hessian and warns when its band is too wide to pay.

## License

MIT.
