# arael-faer

faer extensions, staged for upstreaming. Everything here is built on faer's
public API and laid out the way it would sit in faer itself; arael depends on
it, but nothing in it is arael-specific.

Two things faer does not ship:

- **Block CSC** (`bsc`) -- sparse matrix storage over a *variable* block
  partition. A Hessian assembled from entities (6-wide poses, 3-wide points)
  is a matrix of small dense tiles, and storing it that way means one index
  lookup per tile instead of per scalar, and dense kernels inside the tile.
- **Schur complement** (`schur`) -- eliminate a set of mutually uncoupled
  blocks from a block-CSC matrix and factorize only what is left. This is the
  landmark/point marginalization that makes bundle adjustment and SLAM
  tractable, and it needs the block structure to be cheap.

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

## License

MIT.
