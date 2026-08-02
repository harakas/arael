# Supernodal block Cholesky: design and staged plan

A sparse Cholesky factorization that works directly on a
`SymbolicSparseBlockColMat` -- supernodes built from whole blocks, dense
column-major panels, no scalar-CSC round trip -- as a new `arael-faer`
module and a `SparseFaer` route beside the envelope one. This file is
the working plan. Aim: a faster (and leaner) factorization than
flatten-to-scalar + faer, on the matrices arael actually produces.

## Why

Every factorized route today flattens the block matrix to scalar CSC
and hands it to faer. What that costs, per damped attempt and at setup:

- **The flatten.** `csc_vals_into` copies every stored scalar of S (or
  H) into a scalar value array -- 2.23M scalars (17.8 MB) per attempt
  at Ladybug-372.
- **faer's own copy.** `factorize_numeric_llt` starts by carving
  scratch for a full permuted copy of A -- values, a column pointer and
  a row index array (faer 0.24.4, cholesky.rs:3945 and the scratch at
  :3841) -- then assembles it into the factor panels one scalar at a
  time through an n-sized scatter map. The matrix is copied twice more
  after we already copied it once.
- **The scalar symbolic.** faer's analysis walks the scalar pattern:
  164 ms at Ladybug-1723 (first iteration 740 ms against 350 steady).
  The block graph has 3-9x fewer nodes and tiles instead of scalars for
  edges; the block-level nested dissection already runs in 50 ms where
  the scalar-coordinate work took 916.
- **Duplicate storage.** The sparse route holds S twice (block form +
  `s_vals`) plus `s_row_idx` plus the factor: ~35 MB of scalar-CSC
  duplicates at Ladybug-372, on routes where peak memory is a headline
  number.
- **The memory-bound small-supernode regime.** Instrumented faer at
  slam-300 whole-H (landmarks-first): rank-3 landmark updates run at
  6.6-7.2 GFLOP/s against 40-60 for the big trailing panels -- 37-44 ms
  of a 60-66 ms factorization. It is arithmetic intensity, not kernel
  quality: a rank-3 update streams the whole m x n output panel for 6
  flops per entry. Only a structural change (batching updates per
  target) attacks it, and that needs the block picture.

And two strategic reasons that are not per-attempt milliseconds:

- **faer has no supernode-level parallelism.** Its supernode loop and
  descendant loop are strictly sequential; `Par` only reaches the dense
  kernels (verified over the whole of cholesky.rs). An etree-scheduled
  block factorization is an open multicore win faer does not occupy.
- **Owning the factor format.** A block factor we control opens
  fused assembly (model tiles scattered straight into panels), a block
  selected inverse for covariance, and incremental refactorization --
  none reachable through faer's opaque factor.

What is NOT on the table: the big-panel GEMM regime. faer's supernodal
runs at the machine's dense rate there (~50 GFLOP/s, measured parity
with CHOLMOD supernodal on the reduced slam system). On matrices
dominated by big panels the numeric ceiling is parity, and the win is
everything around the GEMMs.

## Where it would run

Stage-0 measurements (below) ranked the targets; the table reflects
them:

| target | today | measured gain surface |
|--------|-------|-----------------------|
| whole H, pose graph (pgo) | scalar AMD + faer supernodal, factorization ~35% of the iteration | 19-35% of the numeric is deletable copy/assembly/index overhead; small panels run at 13-47 GFLOP/s, so amalgamation has headroom on top |
| whole H, BA-fill (slam non-default) | landmarks-first + faer | 55% of the numeric is memory-bound small-width updates at 11 GFLOP/s -- the batching attack, bold and uncertain |
| reduced S, BA-shaped (bal) | block ND + scalar faer; the S factorization is 81% of the iteration at 372, 96% at 1723 | per-attempt only 3-8% (dense-rate GEMM dominates); the win is setup (sym 66 ms at 1723c) and memory (116 MB of scalar-CSC duplicates) |
| reduced S, trajectory-shaped (slam) | envelope route (already block-native) | none -- envelope keeps this regime unless the supernodal matches it |
| f32 twins of the above | same scalar round trip in f32 | same deletions; kernels already f32-capable |

## faer's design, and what we take from it

The full technical picture lives in faer 0.24.4
`src/sparse/linalg/cholesky.rs`; the shape we inherit:

- **Symbolic pipeline:** elimination tree + column counts in one
  Liu-style pass over the upper triangle (:588); fundamental supernodes
  by the three-condition adjacency test (:2459); relaxed amalgamation
  merging a supernode into its immediately preceding parent under a
  zero-fill budget (`relax` table, default `(4,1.0) (16,0.8) (48,0.1)
  (MAX,0.05)`; :2529-2726); patterns filled by an up-looking reach over
  the supernodal etree, emitted in increasing column order so every
  pattern is sorted for free (:1417); the supernode adjacency
  transposed and deduplicated into a row-major graph giving each
  supernode its sorted descendant list (:2894).
- **Numeric: left-looking, not multifrontal.** No frontal stack, no
  extend-add; scratch is one n-sized scatter map plus the largest
  single descendant update. Per supernode: assemble A into the panel,
  apply each descendant`s update (two binary searches into its sorted
  pattern split it into mid/bot, relative indices map rows into the
  target panel), dense-factor the diagonal block, triangular-solve the
  panel below it. The factor is one flat value array; `split_at_mut` at
  the supernode`s value offset separates read-only history from the
  panel being built (:3133-3269).
- **Panels are plain dense column-major**, diagonal block stored full
  (upper half wasted) so unmodified dense kernels apply.
- **Its one trick we cannot take:** `spicy_matmul` fuses the scatter
  and the diagonal scaling into the GEMM microkernel, so the update
  matrix is never materialized. It is `pub(crate)` -- unreachable. Our
  compensation is that block-run scatters are contiguous-segment adds,
  not per-scalar, and the mid part lands contiguously by construction.

Deviations, all block-motivated:

- **The symbolic runs on the block graph.** Etree, counts, supernode
  detection and amalgamation over block columns; a block is a born
  supernode (its scalar columns share one tile pattern by
  construction). Fundamental detection then only decides which BLOCKS
  chain; amalgamation accounts zero-fill in scalar units (widths are
  known) with the relax table as a tunable.
- **The permutation lives in the scatter map.** The symbolic takes an
  optional block permutation (from `nd::order_graph`, block-AMD, or
  natural) and bakes it into the tile-to-panel source map. No permuted
  copy of S, ever -- the seed scatter IS the permutation. RHS
  permutation is one gather per solve.
- **Assembly is per tile, not per scalar.** The source map stores, for
  every stored tile, its target offset + stride in the factor panel
  (the `s_src` pattern from envelope, generalized): the seed pass is
  per-tile column memcpys (transposing upper tiles into the
  lower-panel convention), and damping writes through precomputed
  diagonal positions.
- **Relative indices are per block row, not per scalar row** -- the
  index lists shrink by the block width, and each entry names a
  contiguous run in the target panel.

## What exists to reuse (arael-faer)

- `bsc`: the storage and its invariants (sorted block rows per column,
  upper-tile convention with full-square diagonal tiles, tile-local
  column-major stride), `from_scalar_coords`, `PositionResolver`.
- `envelope`: the working precedent for everything but fill -- the
  precomputed source map with a `NO_SRC` sentinel, seed-zero-then-
  scatter, strided panel windows wrapped as faer `MatRef`/`MatMut` via
  `from_raw_parts` feeding `matmul` and `solve_lower_triangular_in_place`,
  the `with_panel_width(sym, Option<usize>)` constructor shape, exact
  flop pricing, `EnvelopeError` as the one failure mode.
- `schur`: `gemm_sub` (crate-visible dispatch to 29 unrolled shapes +
  nano-gemm fallback -- contract is packed contiguous tiles, which the
  update scratch satisfies), `llt_in_place`, `llt_solve_panel`,
  `SchurContext`'s grow-once workspace pattern, `SchurTiming`'s
  zero-cost staged stopwatch, the stamp-array + touched-list symbolic
  technique.
- `nd`: `Graph::of_blocks` + `order_graph` (public) give the
  block-level elimination order directly.
- Test house style: LCG builders, golden dense comparisons with
  `rel_resid`, structural assertions, thread-local dispatch counters,
  microbenchmarks as `examples/` with results checked into `results/`.

**Nothing exists for:** elimination tree, column counts, supernode
detection, amalgamation, factor patterns, descendant graphs -- in
either crate. That is the bulk of the new work.

## Design decisions (taken now, revisable at stage boundaries)

- **Module** `arael-faer/src/supernodal.rs`, sibling of `envelope`.
- **LLT only, lower-L convention.** Panels store L column-major as
  faer does; the seed transposes each upper tile into place (tiles are
  cache-resident, the transpose is free against the copy). LM damping
  owns conditioning, so no LDLT, no pivoting, no dynamic
  regularization; a non-positive pivot returns an error and the LM
  loop raises lambda -- exactly the envelope contract.
- **Dense work through faer's public kernels** (`matmul`,
  `triangular::matmul`, `linalg::cholesky::llt::factor::cholesky_in_place`,
  `triangular_solve`) for panels, `gemm_sub`/`llt_in_place` where a
  small fixed shape is hot. Updates: GEMM into a packed scratch sized
  by the largest update (symbolic-derived bound), then block-run
  scatter-add into the panel.
- **Generic over `SchurReal`** (f32 + f64) from the first numeric
  commit, like envelope.
- **`SparseIndex`/`ValueIndex` (u32) throughout**, checked conversions
  via `value_index`; factor value counts checked at symbolic time (the
  4e9-value ceiling is a 34 GB f64 factor, past any target).
- **Damping by re-seed.** The factor is overwritten in place each
  attempt; the pristine values stay in the block matrix (S is rebuilt
  by `schur_reduce` each attempt anyway; H is re-damped through
  `bdiag_pos` as today). Seed cost is one streaming pass over the
  factor, which the envelope route already pays.

## Staged plan

### Stage 0 -- ground truth and go/no-go [DONE 2026-08-02]

Measured with `block_stage0` bins in benchmarks/bal and benchmarks/pgo
(uncommitted, as is the `[patch.crates-io]` faer override they build
against: the instrumented clone at `~/.local/opt/faer-rs`, which
self-reports each numeric factorization's internal split when
FAER_DBG_TIMING=1). Single core pinned, min of 5, this dev VM. The
"deletable" column is copy (faer's internal permuted copy of A) +
asm/idx (its scalar panel assembly and per-pair index lists) + the
`csc_vals_into` flatten where the route pays one.

Factorized-Schur route, BAL (block ND ordering, supernodal forced):

| dataset | reduce | flatten | nd | sym | numeric | solve | deletable | of numeric |
|---------|-------:|--------:|-----:|-----:|--------:|------:|----------:|-----------:|
| Ladybug-49 | 5.9 | 0.02 | 0.05 | 0.4 | 0.9 | 0.04 | ~0.2 | ~20% |
| Ladybug-138 | 19.5 | 0.12 | 1.3 | 1.9 | 8.8 | 0.18 | 1.3 | 14% |
| Ladybug-372 | 53.1 | 0.86 | 8.5 | 10.5 | 110.8 | 1.2 | 9.2 | 8% |
| Ladybug-1723c | 182.6 | 4.4 | 46.0 | 66.3 | 1470.5 | 16.9 | 47.4 | 3% |

The 9-wide camera blocks make every supernode big: zero small-width
updates, GEMM at 75-86 GFLOP/s (the dense rate), and the dense
diagonal factor + panel solve is another 32-40%. Nothing but the
overhead is addressable, and the overhead is 3-8% where it matters.
The flatten specifically is 0.3-0.8% -- noise. Scalar-CSC duplicate
storage: 26.7 MB at 372, 116 MB at 1723c.

Whole-Hessian route, pgo (scalar AMD, supernodal forced):

| dataset | blocks | tiles | sym | numeric | solve | deletable | of numeric |
|---------|-------:|------:|----:|--------:|------:|----------:|-----------:|
| m3500 | 3500 | 8953 | 2.1 | 1.6 | 0.27 | 0.44 | 27% |
| city10000 | 10000 | 30687 | 7.1 | 8.3 | 0.90 | 1.50 | 19% |
| garage | 1661 | 7936 | 2.9 | 2.4 | 0.27 | 0.85 | 35% |
| sphere2500 | 2500 | 7449 | 3.7 | 14.7 | 0.73 | 1.07 | 7.5% |

Small pose blocks change the picture: a fifth to a third of the
factorization is copy/assembly/index overhead, and the big-panel GEMM
runs at only 13-47 GFLOP/s (the panels AMD + scalar amalgamation
produce are too small to reach the dense rate) -- so the block
amalgamation levers apply on top of the deletions. garage, the one
pose-graph dataset where g2o/CHOLMOD still beats us, is the worst
offender: 0.85 ms of a 2.4 ms numeric, ~13% of its ~6.6 ms iteration.

Whole-Hessian route, slam-300 landmarks-first (from the real solver,
SLAM_ARAEL_SOLVER=faer): numeric ~56 ms = copy 2.2 + small-regime
asm/idx 0.9 + small GEMM 30.9 (1561 updates at 11 GFLOP/s) + big GEMM
8.5 + dense factor 13.1. The memory-bound small-update regime is 55%
of the factorization; overhead deletions 5.7%. The batching attack is
the only lever that moves it.

slam default route (Schur): FAER_DBG_TIMING counts zero faer numeric
calls -- the envelope route already owns S there. No supernodal target.

**Gate: GO.** The per-attempt case on the BAL S route is weak (3-8%),
but a default route (pgo whole-H) shows 19-35% of its factorization
deletable before any kernel work, the BA-fill batching target is
confirmed at 55%, and the setup and memory wins stand everywhere the
scalar round trip runs today. Priority order for the numeric work:
whole-H with small blocks first (pgo shapes), BA-fill batching second,
BAL S last (parity there is acceptable; its win is setup + memory).

### Stage 1 -- block symbolic [DONE 2026-08-02]

Shipped as `arael-faer/src/supernodal.rs`: `SupernodalSymbolic::new`
over a block matrix and an optional block order. One Liu-style pass
gives the block etree plus column counts in block AND scalar units;
fundamental detection and faer's relaxed-amalgamation budget run on
those (the budget in scalar units); patterns fill by the up-looking
reach; the descendant graph carries a precomputed pattern split per
(target, descendant) pair so the numeric loop never searches; the tile
source map bakes in the permutation and the upper-to-lower transpose
per tile. Verified: with amalgamation off, structural fill matches an
independent scalar-reference symbolic EXACTLY across natural, reversed
and nested-dissection orders (three seeds, four sizes); invariants and
u32-overflow rejection tested. Original plan below.

`supernodal.rs`: `SupernodalSymbolic::new(&sym, perm: Option<&[I]>,
params)` producing:

- block etree + block column counts (one Liu-style pass over the
  permuted upper tile pattern -- the permutation applied via index
  translation, not a permuted copy);
- fundamental block supernodes (three-condition test on block
  columns), relaxed amalgamation with scalar-unit zero accounting and
  a tunable relax table;
- per-supernode sorted block patterns (up-looking reach, emitted in
  ascending order), the transposed + deduplicated descendant graph;
- panel layout (value offsets, scalar dims), the tile source map
  (tile -> panel offset/stride, `NO_SRC` for fill), diagonal scalar
  positions for damping, `factor_val_count`, exact flop count.

Tests: against faer's scalar symbolic on the expanded pattern under
the same permutation -- with amalgamation disabled the scalar fill
counts must match exactly; amalgamation only ever grows the factor and
never splits a block; patterns sorted, contiguity invariants;
u32-overflow paths.

### Stage 2 -- numeric factorization + solve, sequential [DONE 2026-08-02]

`supernodal_factorize` / `supernodal_solve`, generic over `SchurReal`
(f32 + f64). Left-looking; the update is one triangular GEMM (the
mid-by-mid lower half -- computing the full rectangle instead cost 21%
at Ladybug-1723c, the top separators' updates being nearly all
triangle) plus one rectangular GEMM into packed scratch, then a
block-run scatter; the diagonal block factors with faer's dense LLT on
the strided panel, the panel below with its triangular solve. Tests:
dense-reference residual < 1e-10 over seeds x sizes x three orders x
relax on/off; f32 at 1e-3; indefinite matrices error cleanly.

Measured on the stage-0 matrices (same bins, ROUNDS=5 min, solutions
match faer's to 1.7e-8 or better). "attempt" = factorize + solve vs
faer's flatten + numeric + solve; "sym" = our block symbolic vs faer's
scalar one, both once per structure:

| matrix | attempt speedup | sym ms ours/faer | notes |
|--------|----------------:|-----------------:|-------|
| pgo garage | 1.32x | 0.4 / 2.8 | block-AMD ordering |
| pgo m3500 | 1.15x | 0.8 / 1.6 | amd+sym vs sym |
| pgo city10000 | 1.08x | 4.2 / 7.0 | |
| pgo sphere2500 | 1.04x | 0.5 / 3.5 | |
| bal S 49 | 1.24x | 0.0 / 0.4 | ND on both routes |
| bal S 138 | 1.17x | 0.2 / 2.0 | |
| bal S 372 | 1.00x | 1.2 / 10.8 | +18% factor values from relax padding |
| bal S 1723c | 1.02x | 6.4 / 68.7 | factor itself 1467 -> 1449 ms |

The block-AMD question (stage 4) largely answered itself: faer's amd
run on the block adjacency gives the same fill as scalar AMD on all
four pgo datasets, runs in 0.1-2.2 ms, and keeps blocks whole by
construction. The stage-2 gate (ahead per attempt, within 10% raw
numeric) passes everywhere. Original plan below.

`supernodal_factorize(&sym, &s, &mut factor)` and
`supernodal_solve(&sym, &factor, &mut rhs)`:

- seed: zero panels, scatter tiles through the source map (transposed
  into lower-L), damp through the diagonal positions;
- left-looking loop: per descendant, two binary searches on its block
  pattern, GEMM (triangular for the mid part) into packed scratch,
  block-run scatter-add; then dense LLT of the diagonal block + panel
  triangular solve; error -> `NotPositiveDefinite`;
- solve: forward/backward sweeps, dense per-panel kernels, block-run
  gather/scatter; RHS permutation at entry/exit.

Tests: golden dense comparisons (LCG builders; `rel_resid` < 1e-10
f64, < 1e-3 f32) over banded, BA-shaped and random-fill structures;
solution agreement with faer LLT on the same matrix and permutation;
an f32 drift integration test in `tests/` mirroring
`schur_f32_accuracy.rs`.

Microbenchmark (`examples/`, results checked in): factor + solve vs
`csc_vals_into` + faer numeric + faer solve on the stage-0 matrices,
interleaved runs, codegen-units=1.

**Gate:** whole-attempt (seed+factor+solve vs flatten+faer) ahead on
the S matrices, and raw numeric within ~10% of faer where big panels
dominate. If the raw numeric loses more than that, the scatter path
needs work before integration, not after.

### Stage 3 -- the SparseFaer route [DONE 2026-08-02]

Shipped, opt-in: `SparseFaer::with_block_supernodal(true)` (and the
same on `SparseFaerOptions`). Two arms, both mirroring the envelope
precedent exactly:

- **Reduced S** -- a fourth branch at the `solve_damped` seam, taken
  when the envelope declines (iterative and envelope keep precedence);
  block order from the solver's own ordering decision (the nested
  dissection's block order, now kept by `NestedDissection::block_order`;
  block-AMD via the new `supernodal::amd_block_order` otherwise;
  natural passes through). The route allocates none of the scalar-CSC
  machinery.
- **Whole H** -- `setup_whole_supernodal`, reached when the solve does
  not reduce and the narrow-band route did not take the system first.

`SchurPlan::block_supernodal` reports the route. Tests
(`tests/block_supernodal.rs`): S route and whole-H route each match
the scalar route's optimum, the options struct carries the flag, and
the f32 twin solves. Verified: 731 workspace tests, eigen / cholmod /
lapack / no-default builds, every benchmark crate checks, rustdoc
clean. Defaults are untouched (the flag is off), so no benchmark
regression is possible; sweeps come with the default flip at stage 6.
Deferred to stage 6 with the rest of the surface work: the C++/Python
mirror of the new option and plan field (the C plan struct copies
fields explicitly, so the Rust-side addition is invisible there until
mirrored). Original plan below.

- A fourth arm at the `solve_damped` seam (beside iterative /
  envelope / scalar-sparse), plus the whole-H analog of
  `setup_whole_band`; symbolic built once at setup beside the envelope
  gate; the route allocates none of the scalar-CSC machinery
  (`s_vals`, `s_row_idx`, `l_vals`, faer scratch).
- Opt-in first: `SparseFaerOptions::with_block_supernodal(bool)`
  mirroring the narrow-band precedent, `SchurPlan` reporting which
  route ran. Ordering: reuse the existing block-ND decision; natural
  order and explicit permutations pass straight through.
- cxx/python: mirror the new option where the sparse options surface
  is already exported.

Verification: full test suite, feature-gated backends, benchmark
sweeps (pgo, bal, slam, loc) before/after with interleaved binaries.
**Gate:** no regression on any default route; opt-in wins recorded
here. README/charts only move when a default flips (later stage).

### Stage 4 -- measured pushes (each keep-or-kill on the benchmarks)

- **Relax-table sweep** [DONE 2026-08-02, no change]. Swept none /
  faer-default / single-rung zero caps (2%, 5%, 10%) / a 4x-widened
  table, on all four pgo matrices and all four bal S matrices
  (block_stage0 bins). The faer-default table wins or ties on attempt
  time everywhere: the small size-capped rungs earn their keep on
  3-wide blocks (m3500 1.18x vs 0.95x for a bare zero cap), and
  nothing beats it on the 9-wide bal blocks either. One alternative
  worth remembering: a single `(MAX, 0.02)` rung matches the default's
  speed on bal S while holding 8 MB less factor at 372 (41.3 vs 49.1)
  -- a memory-lean choice if the route ever wants a per-class table.
  Amalgamation off is 10-20% slower on every matrix with enough
  structure to merge. Default stays.
- **Fused tiny updates** [MEASURED AND KILLED 2026-08-02]. A scalar
  axpy path writing small updates straight into the target panel (no
  GEMM dispatch, no scratch) was 40-80% SLOWER on the pgo matrices
  (m3500 factor 1.14 -> 2.05 ms): without packing, the destination
  segments are re-streamed once per depth step, while the GEMM path's
  scratch round trip IS the packing that lets faer's microkernels
  touch everything once. Reverted.
- **Fixed-shape update kernels** through `gemm_sub`: not attempted --
  the fused-update result applies. `gemm_sub`'s contract is packed
  contiguous tiles; the update operands are strided panel windows, so
  the pack step would cost what the GEMM path already pays.
- **Update batching** [SHIPPED 2026-08-02, `batch_ratio` default 1.5
  since direct accumulate removed the ratio's memory cost -- see the
  log; the paragraph below records the pre-direct-accumulate sweep].
  Consecutive descendants of a target whose zero-padded joint update
  stays within `batch_ratio` of their individual flops are packed
  (depth-capped at 16-wide members, 512 total) into one A/B pair over
  the union target span -- one GEMM, one dense subtract, one pass over
  the shared region instead of one per member. The ratio is the whole
  game, measured interleaved on slam-300 whole-H (block-AMD order):
  1.2 / 1.5 / 2.0 give the same -4-5% ms/iter (91 -> 87-89) and -4-6%
  full-iter, while 3.0 LOSES 9% -- past ~2, the padding's arithmetic
  outruns the traffic it saves. Neutral on bal S (194 ms/iter, parity)
  and a shade ahead on pgo garage (4.2 -> 4.1; both measured at 1.2).
  The default is 1.2, the memory-lean end of the flat optimum: batch
  buffers cost +0.1 MB peak at slam-300 against 1.5's +4.0, 2.0's
  +6.6 and 3.0's +12.4. Possible follow-up: a marginalize-first block
  order on the whole-H route would make landmark neighbors adjacent in
  elimination order and tighten the unions block-AMD scatters.
- **Direct accumulate** [SHIPPED 2026-08-02, kept for simplification].
  Batched updates now GEMM straight into their contiguous target
  sub-panel (`Accum::Add`, no product buffer, no subtract pass -- the
  batch product scratch is gone entirely), and single-pair updates
  whose target rows and columns are each one contiguous range take the
  same two-GEMM direct path. Measured: garage factor 1.76 -> 1.68 ms
  (~5%), everything else neutral within this VM's noise. Kept: less
  code, one less buffer, one less pass -- but the scratch traffic it
  deletes is measurably small at the default batch ratio, which also
  further deprioritizes a `private-gemm-x86` (x86-only fused scatter)
  experiment: what it would save is what this measured as small.
- **Block-AMD** [DONE at stage 2]: matches scalar AMD's fill on every
  pgo dataset, 0.1-2.2 ms on the block graph; shipped as the
  supernodal route's default ordering.
- **Seed fusion for whole-H**: bind model scatter positions directly
  into pristine panels, skipping block-CSC H where no reduction needs
  it. OPEN, small expected gain (the seed is one tile-wise pass).
- **Solve path**: ours trails faer's supernodal solve ~10% on bal-1723
  (17.4 vs 15.7 ms) and is at parity elsewhere; ~1% of the attempt
  there. OPEN, low value.

### Stage 5 -- parallel numeric (deprioritized 2026-08-02: threads are
### a fringe deployment for the target applications; see the roadmap's
### tail)

Etree-level scheduling: process independent subtrees across threads,
switch `Par` into the dense kernels for the big trailing panels.
Behind the existing `rayon` feature and `num_threads`; sequential
remains the default and the reference. Gate on demand -- nothing else
in the pipeline is threaded today, so this pays only where the
factorization dominates (bal).

### Stage 6 -- consolidation

- Auto route selection priced from the symbolic flop/fill counts (the
  `EnvelopeMode` pattern), replacing the opt-in.
- Envelope subsumption question: natural-order supernodal vs envelope
  on banded systems -- measure; envelope stays if it stays ahead.
- Later candidates, recorded not planned: block selected inverse for
  covariance over this factor; incremental refactorization.
- Docs (arael-faer README + lib.rs, SOLVERS.md), CHANGELOG at the next
  release, benchmark README refresh where defaults changed.

## Performance and memory roadmap (2026-08-02)

The open improvements, prioritized. Supersedes the "still open"
scraps in stage 4: everything live is here. Performance is the primary
axis; memory items are marked, and an opt-in that buys ~30% memory for
some speed is considered worth shipping. Each item follows the
stage-4 rule -- measured on the benchmark suite, keep or kill.

### P1. Persistent numeric workspace, panel-local zeroing
### [DONE 2026-08-02, except the scratch cap]

Shipped: `SupernodalContext` on the `SchurContext` grow-once pattern
(factorize scratch + solve vectors), owned by `SparseFaer` and passed
through the public entry points (signature change); the whole-factor
upfront `fill(ZERO)` replaced by per-panel zero-and-seed at each
supernode's own turn, with the tile map grouped by target supernode in
the symbolic to make that possible.

Measured (same-run comparisons, our factor times): bal-372 111.5 ->
105.2 ms (-5%), bal-1723c ~1455 -> 1418 (-2.5%), solve 1.7-2.1 -> 1.5
and 15.5-17.4 -> 15.1 there; pgo factors -3-7% (garage 1.68 -> 1.57,
city 7.38 -> 7.13, sphere 13.66 -> 12.95); slam whole-H a wash (its
factor is GEMM-dominated), +0.4 MB resident from the buffers now
persisting across attempts.

Remaining from this item: chunk oversized updates by row block so
`upd` (largest single update; big on bal's separators) is bounded --
memory-only, fold into the P5/P6 memory work.

### P2. The supernodal against the envelope on the reduced banded S
### [MEASURED 2026-08-02; default flip deferred to stage 6]

Head-to-head on the slam reduced system (natural order on both routes,
so the comparison is purely the factorization strategy; interleaved at
300, single runs at 60/120; this VM):

| size | envelope full-iter f64/f32 | supernodal f64/f32 | peak f64 env/sn |
|------|---------------------------:|-------------------:|----------------:|
| 60 | 1.70 / 1.48 | 1.74 / 1.37 | 9.8 / 10.4 |
| 120 | 5.23 / 3.78 | 4.96 / 3.71 | 16.5 / 17.7 |
| 300 | 39.4-39.5 / 26.8-27.9 | 36.0-38.4 / 23.9-24.7 | 53.3 / 62.6 |

The supernodal matches or beats the envelope on full-iter at every
size (up to ~5-9% at 300), and at 300 the f32 rows differ in QUALITY,
not just speed: the envelope route rejected a step in both interleaved
rounds (2 accepted of 4 attempts, final cost 24243.923) while the
supernodal accepted 3(3) and reached 24243.910, tracking the f64
optimum. Costs: +5-7 ms first iteration at 300 (the envelope's
symbolic is nearly free) and +9.3 MB peak f64 (+17%) from
amalgamation padding and the update scratch.

loc-1000 (whole-H, kd=11): a wash -- band 16.3, narrow_band 17.1,
faer 17.4, supernodal 17.0 ms full-iter; the iteration is
assembly-dominated and the tuned scalar band solver keeps its
default.

Verdict: the supernodal is the faster factorization for the flagship
slam route, and the sturdier one at f32; the envelope keeps a real
edge in setup cost and peak memory. The default flip is a stage-6
decision and should be taken together with P6 (memory-lean relax) and
the P5 scratch cap, which attack exactly the +17% peak that argues
for the envelope.

### P3. Postorder the block elimination tree
### [DONE 2026-08-02 -- shipped as insurance, measured ~neutral]

Shipped: after the Liu pass the block etree is postordered (children
before parents, sibling subtrees ascending) and composed into the
elimination order; the etree and both column counts relabel directly
(a postorder is a topological order of the tree), so no second pass.
`SupernodalParams::postorder`, default on; off exists for measurement.

What it guarantees: in a postorder every only child sits immediately
before its parent, so fundamental detection can never lose a merge to
the ordering. The targeted test (two dense block cliques eliminated
alternately) shows the difference: 2 supernodes against 12+ without
the postorder, at identical fill.

What it changes on the real benchmark orderings: almost nothing. bal's
nested dissection already emits separators contiguously -- identical
supernode counts at 49-372, and at 1723c a wash (128 vs 127
supernodes, factor 302.3 vs 303.5 MB, times within same-run noise).
The instructive negative: the +18% padding against faer's scalar
analysis at bal-372 is NOT lost adjacency -- pure chains are not
fundamentally mergeable at all (a tridiagonal's column patterns do not
nest); that padding is intrinsic to block-granular amalgamation.
Kept because it is one O(nblk) DFS, symbolic time unchanged, and it
removes a whole class of ordering-quality hazards for orders we do
not control (user-supplied ones included).

### P4. Hint-derived block order for the whole-Hessian route

`setup_whole_supernodal` always orders by block-AMD (or ND on
request). Two consequences: on BA-fill systems AMD drowns in point
cliques (the reason the scalar route ships landmarks-first there), and
batching's unions are scattered because AMD separates the landmark
neighbors that see the same poses. The model's marginalize hint --
already available on this path -- gives an eliminate-first block order
for free: landmarks consecutive, then the trajectory.

Do: when the model names or detects an eliminable set and the solve
does not reduce, order those blocks first (their natural order), the
rest after; fall back to block-AMD otherwise. Measure on the slam and
bal whole-H rows. Expected: tighter batch buckets (the -5% batching
win was measured under block-AMD's scattered order, so there is
headroom), and a sane bal whole-H ordering. Risk: low -- ordering
choice only, correctness unaffected.

### P5. f32 factor under an f64 solve, opt-in (the memory headline)

Store the FACTOR in f32 while the matrix, right-hand side and step
stay f64: half the factor buffer -- the dominant allocation of every
factorized route (299 -> 150 MB at Ladybug-1723c, 41.5 -> 21 at 372)
-- and the update GEMMs run at twice the SIMD width, so it may not
even cost time. The LM step is a damped trial, inexact by nature (the
CG routes ship far looser steps); one refinement pass -- residual
`r = b - S x` at f64 through `mul_symmetric_upper`, correction solved
through the f32 factor, `x += dx` -- restores most of the lost digits
for two extra solves' worth of work.

Do: type-split `supernodal_factorize`/`solve` over (matrix T, factor
F); an opt-in knob (`SparseFaer` + options + benchmark env); refinement
on by default under it; measure accuracy against the f64 route on the
bal and slam suites (final cost and step-quality, the
`schur_f32_accuracy` methodology) and peak `VmHWM`. Expected: factor
memory halved, speed neutral-to-better; the risk is conditioning --
ill-conditioned S may need the refinement pass or a fallback, which is
why it ships opt-in. Route-peak effect from the measured numbers: the
factor is 49.1 of the 233 MB peak at bal-372 (~10% of peak saved) and
303 of the ~1110 MB peak recorded for the factorized route at 1723
(~14%); combined with P6 and P1's scratch cap the option lands in the
20-30% band on factor-heavy systems.

### P6. Memory-lean amalgamation preset (measured, shelf-ready)

The stage-4 sweep already measured it: a single `(MAX, 0.02)` relax
rung matches the default table's speed on wide-block S while holding
16% less factor (41.3 vs 49.1 MB at bal-372). Ship it as a named
preset (`SupernodalParams::memory_lean()` or an auto-pick when the
mean block width is >= ~6), opt-in beside P5. Expected: -10-16%
factor on wide-block systems, ~0 speed cost there; do NOT auto-apply
to narrow-block systems, where the small rungs earn 10-20% speed.
Risk: none, numbers exist.

### P7. Solve-path polish

The solve trails faer ~10% on big-supernode shapes (17.4 vs 15.7 ms at
bal-1723c; ~1% of the attempt) and allocates per call (folded into
P1's context). The factorization's direct-accumulate trick applies
here too: a supernode whose pattern is one contiguous run can GEMV
straight against the x segment, no gather/scatter. Expected: small;
do it opportunistically when touching the solve for P1/P5.

### P8. Batch-acceptance tuning

The acceptance charges each candidate its column SPAN rather than its
exact mid width (over-counts individual flops, so some good batches
are refused), and `BATCH_DEPTH_MAX`/`BATCH_K_MAX` (16/512) were set,
not swept. P4's ordering change re-shapes the buckets anyway, so
tune after it lands. Expected: small; measure on the slam whole-H
row.

### Tail, in this order and no earlier

- **Threads** (was stage 5): a fringe deployment for the target
  applications; do at the very end. First the cheap tier (pass the
  solver's `Par` into the dense kernels -- until then, multithreaded
  users should prefer the scalar faer route, which does), then
  etree-level tasking if demand exists.
- **`private-gemm-x86`**: maybe never. x86-only, unsafe raw-pointer
  contract, version-lockstep with faer -- and direct accumulate
  measured the traffic it would save as small at our defaults. ARM is
  the deployment target; faer's own fallback there scatters
  scalar-by-scalar while ours moves block runs.
- **Scalar micro-kernels of any kind**: killed by measurement (stage
  4, twice). The GEMM path's packing IS the win; do not revisit
  without new evidence.

## Risks

- **Parity ceiling on big panels.** The GEMM regime is at the
  roofline; if stage 0 shows the deletable overhead is small
  everywhere, the case rests on memory + setup + batching + threading
  alone. That is what the stage-0 gate is for.
- **No fused scatter.** faer's microkernel-fused scatter is
  unreachable; our per-update materialize + block-run add can lose on
  scattered patterns with tiny runs. Mitigation: runs are whole blocks
  by construction; measure at stage 2, not after integration.
- **Symbolic subtlety.** Amalgamation zero accounting and pattern
  emission order are easy to get quietly wrong; the scalar cross-check
  tests (exact fill-count equality with relax off) are the safety net.
- **Scope.** The symbolic machinery is roughly a schur.rs-sized body
  of code with nothing to reuse. The stages are ordered so the
  expensive part only starts after stage 0 proves the target.

## Log

- 2026-08-02: P3 shipped (`SupernodalParams::postorder`, default on):
  block-etree postorder composed into the elimination order, etree and
  counts relabeled in place. Guarantees no fundamental merge is lost
  to the ordering (clique test: 2 supernodes vs 12+); measured
  ~neutral on the benchmark orderings, kept as free insurance. Learned
  on the way: the bal-372 padding gap vs scalar analysis is intrinsic
  block-granularity amalgamation, not lost adjacency -- pure chains
  are not fundamentally mergeable.
- 2026-08-02: P2 measured. The supernodal matches or beats the
  envelope on the slam reduced system's full-iter at every size and is
  markedly sturdier at f32 (no rejected steps, better final cost at
  300), for +5-7 ms setup and +17% peak at 300. loc whole-H is a wash
  across all routes. Default flip deferred to stage 6, coupled to the
  memory items that would erase the peak argument.
- 2026-08-02: P1 shipped: `SupernodalContext` (grow-once workspace for
  factorize and solve, a signature change on both entry points) and
  panel-local zero-and-seed replacing the whole-factor memset. Factor
  -2.5-7% across bal and pgo, solve visibly down, slam neutral. The
  update-chunking scratch cap deferred to the memory items.
- 2026-08-02: Performance and memory roadmap written (P1-P8 + tail),
  superseding stage 4's leftover bullets. Directives recorded: threads
  land last (fringe for the target applications), `private-gemm-x86`
  is maybe-never. New items from the re-survey: the whole-factor
  upfront zeroing and per-attempt scratch allocation (P1), the
  never-run supernodal-vs-envelope measurement on the banded reduced
  system (P2), the missing block-etree postorder (P3), the
  hint-derived whole-H ordering (P4), and the f32-factor-with-
  refinement opt-in as the memory headline (P5, ~halves the factor
  buffer). P6 ships the already-measured memory-lean relax preset.
- 2026-08-02: Batch default 1.2 -> 1.5. Direct accumulate deleted the
  batch product buffer, and with it the entire ratio-dependent memory
  overhead (peak at slam-300 is 61.5 MB flat at every ratio, where 1.5
  used to cost +4.0 MB and 3.0 +12.4). Re-measured interleaved
  post-change: 1.5-2.0 lead 1.2 by ~3% full-iter (81.4 vs 84.1 ms), so
  the memory reason for 1.2 is gone and 1.5 wins on speed.
- 2026-08-02: Stage 4, second pass: update batching shipped
  (`SupernodalParams::batch_ratio`, default 1.2 -- the memory-lean end
  of the 1.2-2.0 flat optimum, +0.1 MB of batch buffers against 1.5's
  +4.0) -- -4-5% ms/iter on the slam-300 whole-H route,
  neutral-to-positive on pgo and bal, tests pinning batched ==
  unbatched. User control: `SparseFaer::with_block_supernodal_batching`
  (and the options twin) sets the ratio or disables batching with
  `None`; the default flows from `SupernodalParams::default()`. The
  whole-H supernodal baseline measured on the way: parity with the
  landmarks-first faer row on ms/iter, -11% first iteration, peak 75.4
  -> 61.2 MB. Ratio 3.0 measured 9% WORSE -- the acceptance ratio is
  the entire trade. Still open, both small: whole-H seed fusion,
  solve-path tuning.
- 2026-08-02: Stage 4, first pass: relax sweep run on all eight
  matrices -- the faer-default table confirmed as the right default
  (kept); fused tiny-update path measured 40-80% slower and killed
  (the GEMM scratch round trip is the packing); fixed-shape kernels
  ruled out by the same mechanism; block-AMD already shipped. Left
  open: update batching (the big one, own session), whole-H seed
  fusion and solve-path tuning (both small).
- 2026-08-02: Benchmark wiring: `ARAEL_BLOCK_SUPERNODAL=1` (one
  cross-benchmark name, like ARAEL_LAMBDA_FLOOR) flips the arael rows
  of pgo, bal, slam and loc onto the route; CG rows ignore it (they
  never factor). `ARAEL_BLOCK_SUPERNODAL_BATCH` tunes the route's
  update batching in the same four runners: a ratio overrides the
  default, `0`/`off` disables, a typo aborts the run. Real-row spot checks, interleaved, this VM: pgo
  garage f64 ms/iter 5.8 -> 4.2, 1st-iter 8.6 -> 4.9, peak 32.2 ->
  26.1 MB; bal-372 schur ms/iter 197 -> 195 (parity as measured at
  stage 2), 1st-iter 281 -> 264, peak 280.5 -> 233.1 MB. Same costs
  and iteration counts everywhere.
- 2026-08-02: Stage 3 shipped: `with_block_supernodal` on `SparseFaer`
  and `SparseFaerOptions`, both the reduced-S and whole-Hessian arms,
  `SchurPlan::block_supernodal`, `NestedDissection::block_order`,
  `supernodal::amd_block_order`. Opt-in; defaults untouched. 731
  workspace tests green, feature builds and benchmark crates check,
  rustdoc clean. Next: stage 4 (relax sweep, small-shape kernels,
  update batching, solve-path tuning), stage 5 (parallel numeric),
  stage 6 (auto-pricing, default flip, cxx/python mirror, docs).
- 2026-08-02: Stages 1 and 2 shipped (`arael-faer::supernodal`,
  symbolic + sequential numeric + solve, 7 tests). Faster than the
  scalar faer route per attempt on all eight stage-0 matrices
  (1.00-1.32x), with a 2-10x cheaper symbolic phase. Two findings: the
  mid-by-mid triangular GEMM split is worth 21% at bal-1723c, and
  block-AMD (faer's amd on the block adjacency) matches scalar AMD's
  fill on every pgo dataset while keeping blocks whole. Open for stage
  4: the relax table pads bal's 9-wide blocks (+18% values at 372);
  the solve is a shade slower than faer's on many-small-supernode
  shapes.
- 2026-08-02: Stage 0 measured and closed, gate GO. The plan's
  expected-gain table corrected by the measurements: BAL S
  factorization is all dense-rate GEMM + big-panel dense factor
  (deletable 3-8%), while pgo whole-H is 19-35% deletable overhead
  with small panels far off the dense rate, and slam BA-fill whole-H
  is 55% memory-bound small updates. Numeric priority reordered to
  pgo-shaped whole-H, then batching, then BAL S. Scaffolding left in
  place for stages 1-2: `block_stage0` bins (bal, pgo), faer
  `[patch.crates-io]` overrides in bal/pgo/slam benchmark Cargo.tomls,
  all uncommitted.
