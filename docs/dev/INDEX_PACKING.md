# Packing the assembly scatter map

The indexed sparse assembly used to keep one scatter index per scalar Hessian
contribution. On a bundle problem that map was about the same size as the
Hessian it filled. Blocks with a static tile shape now keep one tile origin
and stride instead, and the map is gone for them.

Nothing here is specific to a linear solver. The map was built once per setup
and consumed by `LmProblem::calc_grad_hessian_sparse_indexed`, so every indexed
route paid it: the plain sparse factorization, all three Schur routes, Eigen
and CHOLMOD alike.

## What it cost

| dataset | scatter positions | as `usize` | Hessian values |
|---------|------------------:|-----------:|---------------:|
| Ladybug-372 | 5,822,022 | 46.6 MB | 5,977,683 |
| Ladybug-1723-clean | 19,335,705 | 154.7 MB | 19,867,041 |

Roughly one stored index per Hessian value on both.

## The scheme

Every position in a stored tile is
`origin + (col - col_start) * stride + (row - row_start)`, so one pair of
numbers describes the whole tile. `SelfBlock` and `CrossBlock` carry that pair
in a `TilePosition { base: u32, stride: u32 }` beside the parameter indices
they already hold, and recover `row_start` / `col_start` from those indices:
each is the block's smallest live index, which is the first non-fixed slot
because parameters serialize in declaration order.

Both resolvers reduce to that form, so one field pair serves every backend:

- `bsc::PositionResolver` (block CSC): `base + (j - col_start) * row_w + (i - row_start)`
- `ScalarCscResolver` (scalar CSC): `col_ptr[j] + prefix + (i - row_start)`, and
  `col_ptr` is affine inside a block column, so the stride is that column's height

Both gained a `resolve_tile` returning the position and the stride.

The stride has to be stored, not derived. For the block CSC it is the row
entity's live width, which the indices give; for the scalar CSC it is the whole
block column's height, which they do not.

## Two binders

`Model::bind_hessian_positions64` takes a `HessianBinder`:

- `Tiled` -- the pattern is tile-expanded, so a block keeps its `TilePosition`
  and pushes nothing.
- `Scalar` -- a COO-built pattern stores only the coordinates that occur, so a
  block's entries are not contiguous and there is no tile to walk. Blocks push
  one position per entry and read them back through a cursor, as before.

`TripletBlock` always pushes: its pattern is only known after a compute. A zero
stride selects the map path at assembly time, which also covers an all-fixed
block (it scatters nothing either way).

## Rebinding

The pattern is solver state, but the scatter targets are now model state. A
warm re-solve can hand a kept pattern a different model instance whose blocks
were never bound to it, so the backends rebind once at solve entry, in
`configure` / the first `compute`. That costs one lookup per block, against one
per scalar to build the old map.

## Measured

Ladybug-372, interleaved before/after, minimum across alternations. The map
goes to zero -- this model has no `TripletBlock`.

Peak memory, MB:

| route | before | after |
|-------|-------:|------:|
| schur | 356.3 | 309.2 |
| schur-cg | 234.6 | 187.4 |
| schur-cg-implicit | 217.6 | 170.4 |

47.2 MB off every route, against the 46.6 MB the map occupied.

Phases, schur-cg, 15 iterations, ms:

| phase | before | after |
|-------|-------:|------:|
| assembly | 157.0 | 154.3 |
| analysis | 36.1 | 21.3 |
| first assembly | 19.5 | 18.9 |
| total | 2051.5 | 2041.1 |

Assembly was the risk: it gains a multiply-add per element where it had a bare
scatter. It did not get slower -- the 47 MB stream it no longer reads pays for
the arithmetic. Analysis drops by 41%, which is the map build going from one
resolver call per scalar to one per block.

Ladybug-1723-clean, same method, f64:

| | schur-cg before | after | implicit before | after |
|---|---:|---:|---:|---:|
| peak MB | 779 | 632.2 | 705 | 558.3 |
| analysis, ms | 144.7 | 113.2 | 147.3 | 112.7 |
| steady assembly/iter, ms | 54.3 | 55.3 | 55.2 | 54.8 |
| 1st-iter, ms | 494.8 | 476.3 | 473.5 | 439.4 |

146.7 MB off each route: the 154.7 MB map, less 6.7 MB for the new fields
(836k blocks). Assembly is unchanged at this scale too. Totals moved by less
than the run-to-run spread of the same binary, so only memory, analysis and
the first iteration are worth reading.

The rebind has to be skipped after a setup that already bound the blocks --
otherwise every solve pays a second bind on its second iteration, which
measured 99.4 ms against 78.0 for that iteration's assembly at 1723.
`tests/lm_session.rs::each_solve_binds_the_blocks_once` pins it.

## Left

- The fallback map is still `usize`. It now survives only for `TripletBlock`
  and COO-built patterns, where narrowing it to `u32` would still be free of
  the assembly-time risk: positions index a value buffer that would need 4e9
  entries (34 GB of `f64`) to overflow.
- Ladybug-1723-clean is unmeasured here; 154.7 MB of a 705 MB peak is the
  prediction.
