# Packing the assembly scatter map

The indexed sparse assembly keeps one scatter index per scalar Hessian
contribution. On a bundle problem that map is about the same size as the
Hessian it fills. This is a plan to replace it with one index per TILE, stored
in the block objects that already carry the per-instance indices.

Nothing here is specific to a linear solver. `positions` is built once per
setup and consumed by `LmProblem::calc_grad_hessian_sparse_indexed`, so every
indexed route pays it: the plain sparse factorization, all three Schur routes,
Eigen and CHOLMOD alike.

## What it costs

`VERBOSE=1` reports it per solve (`simple_lm.rs`, the reduced-route setup):

| dataset | scatter positions | as `usize` | Hessian values |
|---------|------------------:|-----------:|---------------:|
| Ladybug-372 | 5,822,022 | 46.6 MB | 5,977,683 |
| Ladybug-1723-clean | 19,335,705 | 154.7 MB | 19,867,041 |

Roughly **one stored index per Hessian value** on both, so the map costs about
what the Hessian costs. At Ladybug-1723 it is 154.7 MB of a 705 MB peak.

## The idea

Every position is `val_ptr[b] + local_col * row_width + local_row`, so a whole
tile is derivable from one base. And the blocks already hold per-instance
indices:

```rust
pub struct SelfBlock<A, const N: usize, const M: usize, T> {
    indices: [u32; N],
    hessian: [T; M],      // upper triangle
    ...
}
pub struct CrossBlock<A, B, const NA: usize, const NB: usize, const P: usize, T> {
    indices_a: [u32; NA],
    indices_b: [u32; NB],
    cross_hessian: [T; P],
    ...
}
```

So the base belongs there -- one more `u32` beside the indices it complements,
rather than a parallel array the assembly walks with its own cursor.

At Ladybug-1723 that is ~836k blocks x 4 B = **3.3 MB against 154.7 MB**, and
it removes the side vector rather than shrinking it. Narrowing the existing map
from `usize` to `u32` would only halve it, so this supersedes that.

## What has to be answered first

1. **Do the index arrays stay dense when params are fixed?**

   The scheme rests on `local_row == i`, the slot in `indices_a`. That holds if
   a fixed param is simply absent from the array. It does not if the array
   keeps all `NA` slots and marks fixed ones with a sentinel -- `NA` is a const
   generic and cannot shrink per instance.

   This decides the design: dense means the stride is the const `NA` and the
   base alone suffices; sentinel-padded means the live width is per instance
   and has to be stored or looked up. **Read the macro's index emission before
   anything else.**

2. **`accumulate_hessian_positions` takes `&self`.** Writing a base into the
   blocks needs `&mut self`, which is a trait signature change reaching the
   macro and every backend.

3. **Assembly gains arithmetic where it currently has none.** Today the inner
   step is `vals[positions[k]] += v`, a pure scatter. It becomes
   `vals[base + col * w + row] += v`. Assembly is ~60 ms of a 264 ms iteration
   at Ladybug-1723 and is the tightest loop in the pipeline, so this trades
   memory for time in the worst available place. It may still win -- the map is
   a 155 MB stream competing for the same bandwidth -- but that is a
   measurement, not a deduction.

4. **A fully-fixed block has no tile at all** and must be skipped, not given a
   base.

5. **`TripletBlock` and extended constraints have no static tile shape.** Their
   Hessian pattern is only knowable after a compute, so the scalar map stays as
   the fallback. This is a fast path for statically-shaped blocks, not a
   replacement.

## Staged

**A. Settle question 1.** Read how the macro fills `indices` / `indices_a` for
a model with fixed params, and write a test that pins the answer either way.
Everything downstream is contingent on it, and it is an afternoon at most.

**B. Add the base, keep the map.** Fill the new field during setup and assert
against the existing positions on a real model -- ideally in the bal or slam
benchmark, where the shapes are not toys. Peak memory goes UP while both live;
that is fine, the point is to prove agreement before removing anything.

**C. Switch and measure.** Move assembly onto the base, drop the map, and take
interleaved before/after on BOTH axes: peak memory and assembly time. Question
3 is the reason to expect a time cost, and the change is only worth making if
the memory saved is not paid back in the assembly loop.

**D. Then reconsider `u32` for whatever remains.** If the fallback map survives
for `TripletBlock` models, narrowing it is still free of question 3's risk --
positions index a value buffer that would need 4B entries (34 GB of `f64`) to
overflow `u32`.

## Why bother

Ladybug-1723 peak, f64, by route: 1388 MB factorizing, 779 MB explicit CG,
705 MB implicit CG. Against Ceres's `iterative_schur` at 618 MB. The solver
side is done -- nothing is held for the linear solve beyond the Hessian itself
-- so the remaining gap is the Hessian and the machinery around it, and this
map is the largest single piece of that machinery.
