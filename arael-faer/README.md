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
- **Band Cholesky** (`band`) -- factorize a block-CSC matrix that is banded in
  natural order directly in block form, fill confined to each column's
  envelope. A trajectory's Hessian, and its reduced pose system, are banded, so
  this needs no fill-reducing ordering and no symbolic phase.
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
| `FIXED_SHAPES` / `has_fixed_kernel` | the tile shapes with a fully unrolled GEMM kernel. Anything else works, through a generic loop, at roughly half the speed |
| `SchurSymbolic::gemm_shapes` | which shapes a given problem needs, and how many pair contributions each carries -- so a caller can see whether it is on the slow path |

The shapes with unrolled kernels are the ones SLAM systems actually use --
observers 3 (a 2D pose), 6 (a 3D pose), 7 (a similarity), 9 (a camera with
intrinsics); marginalized 1 (inverse depth), 2 (a 2D point or a bearing), 3 (a
3D point), 4 (a 3D line or a 2D segment) -- cross-checked against g2o's vertex
dimensions and GTSAM's variable dimensions. A shape outside the list is not an
error, only slower, and `gemm_shapes` will say so.

The caller factorizes S itself (`csc_pattern` + faer's sparse Cholesky, or any
other solver), then calls `schur_backsub`. See `arael::simple_lm::SparseFaer`
for the whole loop.

## band -- narrow-band Cholesky

Given a symmetric positive-definite block-CSC matrix banded in natural order,
factor `R^T R = S` in block form. Cholesky preserves each column's envelope
(George-Liu), so fill stays inside the band: no fill-reducing ordering, no
symbolic analysis, no scalar-CSC round trip. Works on the whole Hessian (a pose
graph or localization system) or on a reduced Schur system -- anything banded
in its natural order.

| | |
|---|---|
| `BandSymbolic::new(sym)` | analyse once: the factor's envelope pattern and the map back to the matrix's values. The half-bandwidth falls out of the block structure; the caller supplies none |
| `band_factorize(sym, s, factor)` | numeric: `R^T R = S` into a factor buffer. Left-looking by block column, reusing schur's unrolled tile kernels |
| `band_solve(sym, factor, rhs)` | solve `S x = rhs` in place from the factor |
| `BandError` | the matrix was not positive definite |

Worth it only when the band is narrow: in benchmarks a wide band factorizes
faster as a general sparse matrix (faer's supernodal Cholesky).
`SparseFaer::with_narrow_band` wires it into the LM loop and warns when the band
is too wide to pay.

## License

MIT.
