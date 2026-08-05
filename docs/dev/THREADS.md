# Threading the block supernodal Cholesky

Research and staged plan. Written 2026-08-05, against the supernodal
route as it stands in `arael-faer/src/supernodal.rs`.

The short version: **the original roadmap had the order backwards.**
`BLOCK.md`'s tail says "etree-level scheduling, then switch `Par` into
the dense kernels". Measured on the trees we actually factor, tree
parallelism is capped at 1.1-3.3x and is worth *nothing* on the shapes
where the factorization is most expensive, while the dense kernels
carry 87-100% of the work there and scale ~3x at four threads. Dense
kernels first, tree second, and the second one may never be worth it.

## Where the solver stands

*(As found, before tier 1.)* `BlockSupernodalMode::Auto` took the
supernodal route only when `num_threads == 1`; a threaded solve was
handed to faer's scalar Cholesky, whose dense kernels use the threads.
Every kernel call inside `supernodal_factorize` passed `faer::Par::Seq`.
Threading and the block route were mutually exclusive, which is why
`tests/threading.rs` had to pin a route to compare thread counts at all.

Numeric factorization is left-looking over supernodes:

```
for s in 0..ns:
    zero and seed panel s from A          (tile scatter)
    for each descendant d of s:           (apply_pair / apply_bucket)
        panel_s -= L_d * L_d^T            (GEMM into scratch, scatter back)
    cholesky(panel_s top q x q)           (faer llt)
    trsm(panel_s below)                   (faer triangular solve)
```

Panel `s` reads its descendants' panels (all earlier) and writes only
its own. `SupernodalContext` holds one shared scratch set (`upd`,
`blk_row`, `runs`, `a_cat`, `b_cat`, `chol_mem`).

## What the trees look like

Dumped from real runs (`SupernodalSymbolic::supernode_dims` and
`supernode_parent`), panel cost modelled as
`q^3/3 + (h-q) q^2 + (h-q)^2 q`.

| problem | route | supernodes | work/span | coarse P=4 | top work | q med | q max |
|---|---|---:|---:|---:|---:|---:|---:|
| slam figure-8, 1200 poses | whole H, ND | 3958 | 2.4x | 2.1x | 30.9% | 3 | 1293 |
| slam S-curve, 900 poses | reduced S | 39 | 1.0x | 1.2x | 76.3% | 108 | 1122 |
| slam S-curve, 300 poses | reduced S | 6 | 1.0x | 1.1x | 87.3% | 162 | 978 |
| pgo parking garage | whole H, AMD | 709 | 4.0x | 3.3x | 7.3% | 12 | 120 |
| pgo sphere2500 | whole H, AMD | 1505 | 2.1x | 1.7x | 44.1% | 6 | 600 |
| Ladybug-372 | reduced S, ND | 17 | 1.3x | 1.4x | 63.7% | 45 | 2142 |
| Ladybug-372 | whole H | 47442 | 1.3x | 1.3x | 66.1% | 3 | 1674 |

- **work/span** is the best any tree schedule can do with unlimited
  threads: total work over the longest root-ward chain.
- **coarse P=4** cuts the tree so no subtree exceeds a quarter of the
  work, runs those subtrees on four threads and the top sequentially --
  the "few big chunks" shape. **top work** is that sequential remainder.

Two facts dominate everything below. **A reduced Schur system is a
chain**: work/span 1.0 at both slam sizes, so tree parallelism has
nothing to offer -- there is only one path. And **the work concentrates
in a handful of huge panels**: the top ten panels are 63% of the work at
figure-8 1200, 98% at Ladybug-372 reduced, 100% at S-curve 300. The
biggest single panels are 1233x2028, 1122x1122, 1674x1674, 2142x2142.

Work by panel width:

| problem | q < 32 | 32-128 | 128-512 | q >= 512 |
|---|---:|---:|---:|---:|
| slam figure-8 1200 | 7.2% | 14.6% | 41.0% | 37.2% |
| slam S-curve 900 | 0.0% | 58.8% | 29.9% | 11.3% |
| pgo parking garage | 90.1% | 9.9% | 0.0% | 0.0% |
| Ladybug-372 reduced | 0.4% | 10.7% | 25.2% | 63.7% |

pgo is the outlier in both tables: small panels, no concentration, and
the only tree with real width. It is also the cheapest problem we run.

## Does faer scale at these sizes?

Measured on this machine (8 cores, `RAYON_NUM_THREADS=8`, f64), sizes
taken from the dumps above.

| `matmul` C[m,n] -= A[m,k] B[k,n] | 1 thread | 2 | 4 |
|---|---:|---:|---:|
| 795x1233x1233 (figure-8 top panel) | 43.6 ms | 1.89x | **3.06x** |
| 1674x1674x333 (Ladybug-372) | 32.5 ms | 1.94x | **3.45x** |
| 400x300x200 (mid-tree) | 0.82 ms | 1.58x | 1.92x |
| 60x40x30 (leaf) | 3 us | 0.86x | 1.00x |

| `solve_lower_triangular_in_place` | 1 thread | 2 | 4 |
|---|---:|---:|---:|
| 795x1233 | 23.4 ms | 1.45x | 1.89x |
| 1809x333 | 4.3 ms | 1.57x | 1.80x |
| 852x162 | 0.59 ms | 0.89x | 0.85x |
| 54x48 | 4 us | 1.02x | 1.02x |

| `llt cholesky` of the q x q diagonal | 1 thread | 2 | 4 |
|---|---:|---:|---:|
| q = 1233 | 13.3 ms | 1.44x | 1.84x |
| q = 600 | 1.74 ms | 0.93x | 1.02x |
| q = 162 | 60 us | 0.99x | 0.96x |
| q = 48 | 4 us | 0.99x | 1.00x |

GEMM scales well and is where the work is. The triangular solve scales
half as well. The dense Cholesky barely scales at all and *regresses*
below q ~ 600 -- it is a small share of panel cost, so this costs little,
but it must not be threaded blindly. Everything small loses to the pool.

## Tier 1 as built [DONE 2026-08-05]

Measured end to end, f64, interleaved, `BENCH_THREADS` driving both the
core pin and `num_threads`. Costs and iteration counts identical at every
thread count on every row.

| problem | 1 thread | 2 | 4 | speedup |
|---|---:|---:|---:|---:|
| Ladybug-372, reduced S | 192.0 ms | 155.6 | 134.8 | **1.43x** |
| slam figure-8 1200, whole H | 314.5 ms | 249.0 | 232.8 | **1.35x** |
| slam S-curve 900, reduced S | 172.7 ms | 172.1 | 173.0 | 1.00x |
| pgo parking garage | 3.71 ms | -- | 3.72 | 1.00x |

Bundle adjustment and the figure-8 gain; the pose graph is untouched
(its panels are all below the gate, which is the gate working); the
S-curve gains nothing, which the kernel numbers did not predict.

**Why the S-curve is flat.** Its factorization is a smaller share of the
iteration than the model assumed -- the Schur reduction that *forms* the
1122-wide S is the larger half, and it is sequential. Amdahl, not a
failure of the kernels: the same panels in Ladybug-372, where the
reduction is comparatively cheaper, deliver the 1.43x. This is the
strongest argument yet for option F below.

The gate thresholds were measured, not guessed, and the guess in the
first draft of this document (2e6) would have been actively harmful:
faer parallelises from about 1e6 and *loses* until about 2e7.

| kernel | work | gate | below the gate | above |
|---|---|---:|---|---|
| `matmul` | m*n*k | 2e7 | 0.22-1.0x | 1.6-2.8x |
| `solve_lower_triangular` | rows*q^2 | 1.5e8 | 0.15-0.98x | 1.5-2.0x |
| `llt cholesky` | q^3 | 5e8 | 0.64-1.04x | 1.2-2.4x |

Bit-identity held, so `BlockSupernodalMode::Auto` no longer branches on
the thread count: the block route runs at any `num_threads`, the scalar
fallback is gone, and `tests/threading.rs` is back on the default route.

## Expected gain (written before tier 1; kept for the record)

Blending the kernel scaling against the work distribution, at four
threads:

- slam reduced S (chain, all work in wide panels): **~2.2-2.5x**, all of
  it from tier 1.
- slam figure-8 1200 whole H: **~2.5x** from tier 1; tier 2 could add
  the 13% that sits in narrow panels, worth maybe 0.1x.
- Ladybug-372 reduced: **~2.5-3x** from tier 1, the cleanest case.
- pgo: **~1x from tier 1** (90% of work in panels under 32 wide, where
  the kernels do not scale), up to 3.3x from tier 2.

So tier 1 covers slam and bundle adjustment, tier 2 covers pose graphs
and nothing else. Given the target applications, tier 1 is the whole
feature and tier 2 is optional.

## Options considered

**A. Par into the dense kernels, size-gated.** Thread the 18 kernel
calls, each behind a work threshold. No scheduling, no shared mutable
state, no unsafe: the calls are already sequentialised by the loop.
Cheap to build, cheap to revert, and it is where the measured work is.
*Taken -- tier 1.*

**B. Coarse subtree parallelism.** Cut the tree so each thread owns one
subtree, run the top sequentially. Few big chunks, which is the
requested shape. Subtrees are self-contained (a panel reads only its own
descendants), so the parallel writes are provably disjoint -- but
`factor` is one `&mut [T]`, so handing N disjoint panel ranges to N
threads needs raw pointers and an invariant argument. Also needs
per-thread scratch. *Taken -- tier 2, optional, pgo-only.*

**C. Per-supernode task DAG.** A task per supernode, dependencies from
the etree, work-stealing. Standard for large supernodal solvers. Rejected:
the median supernode is 3-12 wide, so this is thousands of tasks of a few
microseconds -- exactly the overhead the target scale cannot absorb, for a
ceiling (work/span) of 1.0-4.0x that B already reaches with a fraction of
the machinery.

**D. Fan-in parallelism over descendants.** Parallelise the descendant
loop for one panel: each thread accumulates into a private update buffer,
then reduce. This is the only option that helps a panel with thousands of
tiny descendants (Ladybug-372 whole H: 47442 supernodes). Rejected for
now: costs P copies of the largest update buffer, needs a reduction pass,
and the shapes it would help are ones where the reduced route -- which we
take by default -- has already collapsed the problem to 17 supernodes.
Reconsider only if a real problem shows up with a wide, shallow tree.

**E. 2D / block-cyclic panel distribution.** Distributed-memory
technique. Rejected: we are one process on one machine, and the panels
are already handed to a threaded BLAS3 by A.

**F. Thread something else instead.** The factorization is 30-60% of an
iteration on the problems above; assembly, the Schur reduction and the
solve are all sequential. Threading the Schur reduction (an independent
GEMM per landmark block) is plausibly a bigger win than tier 2 and is
entirely separate work. Out of scope here, worth its own study.

## Plan

### Tier 1 -- Par into the dense kernels [DONE]

1. `SupernodalParams` gains `par: faer::Par` (default `Par::Seq`), or
   `supernodal_factorize` takes it as an argument. The argument is
   cleaner: the symbolic is cached across solves and the thread count is
   a per-solve property.
2. Add a size gate. From the tables: parallelise a `matmul` when
   `m*n*k` is above ~2e6 (400x300x200 = 2.4e7 gained 1.92x; 60x40x30
   lost), a triangular solve when `rows*q` is above ~5e5 (852x162 =
   1.4e5 lost 15%), and the diagonal Cholesky only above q ~ 800.
   One helper, `fn gate(work: usize, par: Par) -> Par`, used at every
   call site so the thresholds live in one place.
3. Thread `par` through `apply_pair`, `apply_bucket`, the panel factor
   and the panel solve. Nothing else changes -- no scratch changes, no
   unsafe, no ordering change, so the result stays bit-identical to the
   sequential run at any thread count (each kernel is deterministic; faer
   does not reassociate across threads for these routines -- **verify
   this before relying on it**, it is the one assumption in tier 1).
4. `simple_lm`: pass the solver's `Par` down, and drop the
   `sn_take()` restriction to `Par::Seq` so `Auto` takes the block route
   when threaded. Delete the corresponding line in the `BlockSupernodalMode`
   docs and the note in `docs/SOLVERS.md`'s Threads section.
5. `tests/threading.rs` goes back to the default route and regains the
   case it lost when it was pinned to scalar (see `BLOCK.md`'s threads
   item), provided step 3's bit-identity assumption holds. If it does
   not, the test keeps a pinned route and the docs say results depend on
   thread count.

Benchmarks: slam 300/900 reduced S and figure-8 1200, bal-372 reduced,
pgo garage, at 1/2/4 threads, interleaved. Accept if 4 threads beats 1
by >1.8x on slam and bal and does not regress pgo by more than the
noise band.

### Tier 2 -- coarse subtree parallelism (optional, pgo-shaped problems)

1. Symbolic: after the postorder, compute subtree work and cut at
   `total/P` to get the chunk list plus the sequential top. Store it on
   `SupernodalSymbolic` (it depends on P, so either store the subtree
   work and cut at factorize time, or recompute the cut when P changes).
2. `SupernodalContext` becomes per-thread: one scratch set per worker,
   allocated once and reused. Note this multiplies scratch memory by P
   -- `max_update` is the largest buffer, so document the cost.
3. Parallel section: each worker takes chunks from the list and runs the
   existing per-supernode body over its subtree, writing only panels in
   its own subtree. Then a barrier, then the top sequentially with tier
   1's kernels.
4. Safety: `factor` is handed to workers as a raw pointer with a
   documented invariant -- panel ranges are disjoint by construction
   (`val_ptr[s]..val_ptr[s+1]`), and a subtree's descendants are inside
   the subtree. Add a debug-only assertion that each worker's writes stay
   inside its chunk's ranges, and a test that a two-thread run over a
   known tree is bit-identical to the sequential one.

Benchmarks: pgo garage and sphere2500 at 1/2/4 threads. Accept only if
garage beats 3x at 4 threads; below that the machinery is not worth it,
and the honest outcome is to stop after tier 1.

## Risks

- **Bit-identity.** Tier 1 rests on faer's threaded kernels producing
  the same values as the sequential ones. If they do not, `Auto` cannot
  silently switch on threads and the docs must say the answer depends on
  the thread count. Verify first, before writing any of tier 1.
- **The pool is shared.** `rayon`'s global pool is the application's, and
  arael is a library. Nested parallelism (a caller solving several models
  on their own threads) would oversubscribe. Tier 1 inherits whatever
  `num_threads` says; tier 2 would want `rayon::in_place_scope`.
- **Scratch memory scales with P** in tier 2. On Ladybug-1723 the update
  buffer is not small.
- **Tier 2 may measure worse than the model.** The cut is static and the
  subtree costs are estimates; a bad cut leaves one thread with the tail.
  The bound is 3.3x on the only problem it helps, so there is little
  headroom for scheduling loss.

## What not to do

Do not build a task DAG over supernodes. Do not thread the diagonal
Cholesky below q ~ 800. Do not parallelise the tile seeding or the
scatter -- they are memory-bound and already the smallest phase. Do not
make threading the default: `num_threads` stays 1, and the route stays
whatever the caller asked for.
