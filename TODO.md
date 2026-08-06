# TODO

- **arael: price `CovOrdering::Auto` without building both symbolics.** Auto
  picks between minimum degree and nested dissection by building a full
  `SupernodalSymbolic` per candidate and reading `flops()` off it, then keeps
  one. On a BAL shape minimum degree wins every dataset and the discarded
  dissection candidate costs more than the whole winning path: at Ladybug-49 a
  single camera marginal is 56.7 ms under Auto against 28.1 ms under `Amd`.
  Column counts over the elimination tree would price the candidates far
  cheaper, but they are NOT equivalent -- they count the scalar factorization,
  while `flops()` counts the supernodal work including the zeros inside relaxed
  panels, and that padding differs between the two orderings. Adopting the
  cheap proxy needs evidence it still picks the same winner on both a BAL shape
  and a revisiting trajectory (slam's figure-8, where dissection wins and the
  pricing pays for itself). Not done for 0.8.2: benchmarks/bal names `Amd`
  outright instead, which sidesteps the pricing without changing the library
  default.

- **arael: gate `EnvelopeMode::Auto`** -- DONE (2026-07-30). Auto prices the
  envelope against the ordered sparse factor of the same reduced system and
  takes it below `ENVELOPE_FLOP_MARGIN`. Cheap shape statistics of the envelope
  do not separate the cases -- a figure-8 at 300 poses has an envelope holding
  exactly what the pattern stores, and still regresses 18% -- so the comparison
  is against the competing factorization itself, which costs a symbolic
  analysis of `S` (reused when the gate declines).

- **arael**: narrow the value-buffer offset maps -- DONE (2026-07-28). Every
  offset into a matrix's values is now `arael::ValueIndex` (an alias, `u32`
  by default): the assembly scatter map, `CscMatrix::diag_pos`, the block
  Hessian's `bdiag_pos`, `BlockJacobi::at`, and the band factor's `s_src`.
  `arael_faer::value_index` is the one checked conversion, so widening the
  library to `u64` is the alias plus a rebuild -- verified by building and
  passing the whole suite both ways. NOT converted, and why: `CscMatrix`'s
  `col_ptr` (handed to the Eigen/CHOLMOD FFI as `*const i64`, relying on
  `usize == u64` layout), `s_col_ptr` / `s_row_idx` / `perm_fwd` / `perm_inv`
  (faer's symbolic API takes `usize`), `SymbolicSparseBlockColMat`'s own
  arrays (generic over faer's `Index`, instantiated as `usize` and shared
  with faer paths), and the nested-dissection graph (analysis scratch, freed
  before the solve). Worth 1.9 MB at Ladybug-1723-clean, where the scatter
  map is already gone and only `bdiag_pos` remains; the halving pays on
  `TripletBlock` and extended-constraint models, which still carry a map.

- **Unscented-transform utility for covariance mapping**. Pushing a
  parameter-space marginal through a nonlinear embedding currently
  linearizes: slam_demo_gm maps its inverse-depth landmark marginal
  (chart, rho) to world position via `J C J^T`, which pretends the
  banana-shaped along-ray uncertainty is a symmetric ellipsoid. A small
  utility -- 2n+1 scaled sigma points from (mean, cov) through a user
  closure, mean and covariance reconstituted -- captures the mean shift
  and the honest variance inflation. Shape:
  `unscented(mean, cov, f: impl Fn(&[f64]) -> Vec<f64>) -> (mean, cov)`,
  probably next to the covariance module. Not done yet because only
  display code consumes the mapped covariance so far.

- **Generic models** -- DONE (2026-07-22) for components, entities and
  constraint structs: one scalar type parameter bounded by `Float`, one
  instantiation per entity name per root (mixing `Vec<Pose<f32>>` and
  `Vec<Pose<f64>>` in one root is a macro error). Blocks route their
  precision at monomorphization through per-precision `Model` impls, so
  `SelfBlock<Pose<T>, T>` sorts itself per instantiation. Still concrete
  by design: `#[arael(root)]` (the root is where generated solver code
  becomes real, at one precision) and `#[arael(fit(...))]` (separate
  codegen path, unaudited for generics -- extend if a use case appears).
  Tests: tests/generic_component.rs, tests/generic_entity.rs,
  tests/generic_model_errors.rs.

- **`loss = |s| ...` in `fit(...)`** -- DONE (2026-07-17, REVIEW3 item 7).
  `fit(...)`/`fit64(...)` accept a trailing `loss = |s| rho(s)` block
  M-estimator over each point's squared residual; reuses the constraint loss
  pipeline; no-loss codegen byte-identical. Test:
  tests/fit_attr.rs::block_loss_cauchy_ignores_outliers. Original entry:
  `constraint(...)` attribute (arael-macros/src/constraint.rs). The `fit(...)`
  form has its own codegen (`generate_fit_impl` in lib.rs) and does not go
  through it, so a robust curve fit currently needs a hand-written constraint
  model. `fit` residuals are scalar (one per data element), so the block is
  just `s = r^2` -- add `loss = |s| <expr>` parsing to `parse_fit_inner` and the
  block_cost/weight emission (from the constraint path) to `generate_fit_impl`.
  This is the canonical robust-curve-fitting case (Ceres's robust_curve_fitting
  example is a fit with CauchyLoss).

- **Automatic marginalize detection** -- DONE (2026-07-12). The macro hands the
  solver the model's type-coupling graph and `SparseFaer` reads the
  marginalizable families off it, so no hint is needed. The quality guard this
  entry asked for exists and is stronger than the fill comparison it proposed:
  a cheap band/dense filter settles the obvious cases from the block structure
  alone, and only the unclear ones pay for the exact fill comparison against
  AMD. `SchurPolicy` overrides, `LmResult::solver` reports what happened.

- **Explicit blocked Schur-complement backend** -- DONE (2026-07-11/12). This
  entry judged it "only worth it if solve-dominated problems become the primary
  target"; it measured 2.1x per iteration at slam-300 and is now the default
  route whenever the model has something to marginalize. It is not a separate
  backend: `SparseFaer` reduces or factorizes the whole system, and picks.

- **SparseFaer::with_threads(n): opt-in multithreaded faer solve** -- DONE
  (shipped as `LmConfig::num_threads` + the `rayon` cargo feature; the
  factorization and triangular solve thread, assembly does not). Note the
  shipped form reversed this entry's "backend builder, NOT an LmConfig field"
  call: config-level won because `configure()` hands it to the backend and
  the other backends ignore it. Original entry:
  (discussed 2026-07). faer threads via rayon: every heavy call takes a
  `faer::Par` (`Par::Seq` / `Par::rayon(nthreads)`, 0 = all cores), and
  `SparseFaer::solve_damped` currently hard-codes `Par::Seq` at its four
  call sites (factor scratch, solve scratch, `factorize_numeric_llt`,
  `solve_in_place_with_conj`). Plan: hold a `par: faer::Par` field on
  `SparseFaer` (default `Seq`), expose a `with_threads(n: usize)`
  builder taking a plain usize so `faer::Par` stays out of the public
  API (1 = Seq, n > 1 = `Par::rayon(n)`, 0 = all cores). Needs a rayon
  import: we build faer with `default-features = false`, so `faer/rayon`
  is compiled out -- forward it through a new arael cargo feature (e.g.
  `rayon`) to keep default builds dependency-light. Deliberately a
  backend builder and NOT an LmConfig field: LmConfig is shared by all
  backends and never reaches `LmSolver::solve_damped`, the knob would be
  dead on Dense/Band/Eigen/LAPACK, and it would overpromise -- only the
  factorize+solve phase threads, not assembly/cost eval. Revisit a
  config-level `threads` only if assembly is ever parallelized. Payoff
  is large-problem-only (loc solve share ~4%: pointless; slam-300
  bracketed at 10-25%/iter by the threaded-BLAS cholmod accident;
  BAL-372+ the real case). Benchmarking it needs the `RAYON_NUM_THREADS=1`
  cap in the run.sh scripts lifted deliberately for that row.

- **lm_resolve(): warm re-solve for arael-sketch dragging** -- DONE
  (2026-07-18, superseded by `LmSession` -- see the SHIPPED entry further
  down; the matrix storage moved into the session instead of the solver, so
  the trait kept its shape). The sketch-integration notes below still apply
  to the pending sketch adoption. Original entry: (design agreed
  2026-07, shelved -- fringe for now; sketch systems are sub-millisecond
  today, so this is headroom, not a rescue). Adds
  `lm_resolve(x0, &mut solver, problem, cfg)` alongside `lm_solve`: same
  loop, but skips the `LmSolver::reset()` at entry so the sparsity
  pattern, COO->CSC position map, and faer symbolic factorization
  (roughly a third to half of a short small-system solve) survive across
  same-structure re-solves. Cold on a fresh solver, so callers need one
  code path. Prerequisite: matrix storage must move INTO the solver
  (drop the `Matrix` associated type; `new_matrix(n)` becomes
  `prepare(&mut self, n)`; compute/extract_diagonal/solve_damped/
  matrix_nonfinite_count lose the matrix parameter) -- lm_solve currently
  creates the matrix locally, so there is nothing to pick up from.
  Contract (documented, not auto-validated -- full pattern validation
  costs the work being saved): structure must be unchanged since the
  previous solve; `prepare` DOES cheaply panic on a parameter-count
  change ("was 800, now 806 -- use lm_solve or reset()"), which catches
  the most common violation (pose/point added); same-n pattern changes
  remain a documented logic error (optional debug_assert full check).
  Sketch integration notes: (1) Sketch implements LmProblem, so the
  persistent solver cannot be a field of the struct passed as `problem`
  (double &mut) -- hold it beside the sketch or Option::take() around the
  call; (2) reuse scope is PER DRAG GESTURE: auto-anchor toggles
  Param::fixed which changes the parameter count, so reset at gesture
  start, warm resolves per mouse-move, reset on release/edit; guard
  states must also be stable within the gesture. Optional follow-up:
  carry converged lambda between resolves (needs final_lambda in
  LmResult, REVIEW.md F4).

- **Residual scale audit for DOF / eigenvalue analysis**: The eigenvalue-based DOF calculation (`compute_dof` + friends) depends on the singular values of `J^T J` being well-separated from zero for real constraints and near-zero for truly redundant ones. Several residuals are deliberately multiplied by a sketch-scale factor (e.g. `arc.radius` on the sweep constraint, `mlen` on the parallel cross product) so their SVs track the sketch size and stay out of the DOF-noise floor even for very small or very large geometry. Constraints whose residuals are NOT scaled by a sketch-size quantity will produce naturally small singular values (think pure-angular residuals like `xangle` / `ArcLineParallel` / `ArcArcParallel`) and are candidates for false-positive DOF rejections on tiny sketches or false-negatives on huge ones. Concrete steps:
  1. Enumerate every `#[arael(constraint(...))]` in `arael-sketch-solver` and note, per residual row, whether it has a sketch-scale factor (radius, length, `mlen`, `constraint_isigma`-only, etc.).
  2. Build a test sketch in pathological scale regimes (1e-3 and 1e5 world units) and log the singular-value spectrum of `J^T J`. Flag residuals whose SV drops below the DOF-zero threshold at one scale but not the other.
  3. Specifically review the `xangle` (ArcRotation target) residual: it was originally `(arc.rotation - target) * arc.radius * isigma` to match the sweep pattern, dropped to `(arc.rotation - target) * isigma` in the v0.6.0 push because the radius factor let the solver collapse radius as a cheap way to zero out a distant target. The unscaled form might misbehave under eigenvalue-DOF at very large sketch scales. Concrete alternatives to try if it does, best-first:
     - **`(arc.rotation - target) * arc.radius_value * isigma`**. The `_value` suffix is the macro-synthesised shadow that reads the last-committed `.value` as a constant, with zero derivative in the generated Jacobian (same trick the SLAM drift regularizer uses for `pose.pos_value`). So this scaling: (i) tracks sketch size exactly like the old `* arc.radius` form, so the row's singular value stays at the usual scale; (ii) contributes zero gradient w.r.t. radius, so the solver *cannot* collapse radius to cheat on the rotation residual; (iii) the scaling factor evolves naturally across solves because `_value` updates from `.value` after each LM pass. Most likely the right answer -- recovers the DOF-floor benefit of the original formulation without its cost. Same move applies to `ArcArcParallel` if its pure `sin(a.rot - b.rot)` is too small at scale: multiply by `(arc_a.radius_value + arc_b.radius_value) / 2`.
     - `(arc.rotation - target) * min(arc.radius, arc.radius_b)`. Keeps the scale link but still leaves a non-zero radius gradient (just clamped to the smaller axis), so the collapse pressure is weaker but not eliminated.
     - An explicit soft ceiling on `|rotation - target|` that saturates above a tolerance so the solver can't trade radius for rotation regardless of scaling.
     Same review applies to ArcLineParallel (`... / mlen` -- uses the live `mlen`, same collapse risk at tiny line lengths; `mlen` could become a `_value`-pattern if needed) and ArcArcParallel (`sin(a.rot - b.rot)` -- pure dimensionless, no sketch-scale link at all).
  4. Document the convention in a short comment block above the constraint bodies so future contributors know why the radius / mlen factors are there and when to keep them.
- **arael-sketch conflict messages**: Enhance `check_constraint_conflict` so the "already parallel / perpendicular / equal / ..." error messages include the offending existing constraint's name (e.g. "conflict with C5: parallel L0 L1"). The plan (`yes-did-you-check-memoized-wren.md`) called this a single-site change, but the conflict messages are spread across many arms in `arael-sketch/src/conflicts.rs` and the current `pair_exists` helper does not expose the matching index. Adding a sibling helper that returns both the boolean and the index (or refactoring `pair_exists` to return `Option<usize>`) would let every arm include the nid. Skipped for now to keep the initial naming feature surface-area small.
- **Sketch editor**: Tag constraints owned by dimensions so they can be deleted logically instead of matching by approximate value comparison. Currently `RemoveDimension` in `arael-sketch/examples/editor.rs` searches for the underlying constraint (e.g. `distance_pl`) by floating-point distance match, which is fragile (e.g. signed vs unsigned mismatch for `PointLineDistance` — the constraint stores -5.0 but the dimension stores 5.0).
- **Sketch editor**: Add `MoveDimension` action to cleanly reposition dimension annotations (offset + text_along). Currently dimension dragging uses the `Drag` action which snapshots the entire sketch state. Alternatively, extend `AddDimension` to accept optional offset/text_along so the position can be specified at creation time.
- derived dimensions -- DONE
- hiding of intermediary PcN points -- DONE
- dragging hide cost != 0
- investigate arc negative radiuses -- DONE (ccw flag + sweep_sign + negative radius rejection)
- implement elliptic arcs
- rect tools etc
- circle creaing tools etc
- mirror tool
- fillet tool
- offset tool
- trim tool
- split tool
- scale tool
- mirror tool
- various circle tools
- text placement
- polygon tool
- **Duplicate constraint check**: `symmetry_pp` (point symmetry) skips duplicate detection because `resolve_as_point` creates helper points before we can check — need to compare semantic endpoints, not Ref<Point> values
- **Redundancy warning**: DONE -- constraints now checked for DOF reduction, rejected if redundant. Use `force` to override.
- Way to get the Jacobian for the system with constraints identifiable for more efficient SVD analysis of DOF in arael-sketch -- DONE (`#[arael(root, jacobian)]` + `#[arael(constraint_index)]`)
- **Document Jacobian feature**: DONE -- documented in lib.rs features, macro docs, README.
- **Degenerate tangent at shared endpoint**: DONE -- TangentLA uses dot-product formulation when tangent point is a shared endpoint (detected via coincident scan). The perpendicular-distance formulation has zero Jacobian at shared endpoints; the dot-product does not.
- **DOF computation ownership in Sketch**: Move async DOF lifecycle into Sketch itself. Mutations kick off async computation internally. `sketch.dof()` blocks and waits if result not ready. GUI reads `sketch.cached_dof` for non-blocking display. Eliminates the external async plumbing (dof_input/dof_output/dof_display, bincode serialize/deserialize copy, poll_dof).
- **arael-sym**: Implement `Mul<E> for f64`, `Mul<E> for i64`, etc. so `2.0 * expr` works (currently only `expr * 2.0` compiles). Same for Add, Sub, Div. -- DONE
- **arael-sketch**: when rotating arcs the arc radius dimension does not rotate -- DONE
- **arael-sketch**: implement sweep A0 driven/distance L0 driven -- make to current value and driven -- DONE
- **arael-sketch**: arc angles when rotating arc can drift? so I see things like arc sweep 480 degrees -- DONE
- **arael-sketch**: just help: add"Type help full for ..". Add Help button. Open command, expand half sketch, issue help full. -- DONE
- **arael-sketch**: dragging should keep hilight, not hilight others -- DONE
- **arael-sketch**: sometimes we somehow get stuck pasting into cmd input -- ??? browser/wasm issue?
- **arael-sketch**: make language more real so that you can do vector algebra
- **arael-sketch**: add keywords into language? stop assigning to anything?
- **arael-sketch**: dimension: distance between concentric circles
- **arael**: support single struct model+root. right now it does not function. -- DONE (SelfBlock<Self> on root + direct-composed sub-model fields now route through EntityLocation::RootSelf / EntityLocation::DirectField in arael-macros)
- **arael-sym**: parse_with_bag -- bag with functions and substitutions so func() gets current active function -- DONE
- **arael-sym**: cli calculator demo app, better than bc, define functions, etc. -- DONE
- **arael**: docs: document constraint has name= property
- **arael**: extend jacobi demo with constraint labels
- **arael-sketch**: clean up points obscuring everything, make line endpoint when creating explicit and clean -- cross on line snap, cross+box on point snap
- **arael-sketch**: auto-perpendicular constraint
- **arael-sketch**: switch to hdimension/vdimension at creation when moving mouse far -- switching commands during tool usage
- **arael-sketch**: hdimension/vdimension at creation when moving mouse far
- **arael-sketch**: toggle tags like driven, quiet, construction during line creation
- **arael-sketch**: robot.cmd scale to 0.1 takes us to "interesting" view
- **arael-sketch**: investigate chain misbehavior
- **arael-sketch**: when selecting a line/point/arc/etc hilight/hover all the constraints associated with it

- **arael**: geometry helpers deferred from the F6 math-coverage review (new feature surface rather than sym/runtime parity; nothing in the repo blocks on them): matrix2/matrix3 `inverse`, matrix -> quaternion conversion, SE(2)/SE(3) compose/between helpers, skew/hat operator (`[v]_x` from a vect3).
- **arael-sym**: quaternsym operations deferred from F6 stage E: `pow`/`log`/`exp`/`slerp`/`from_two_vectors`/`get_axis_angle`. All are branchy (sign flips, acos edge cases, zero-angle guards) and not residual-friendly; the runtime `quatern` has them for data preparation.
- **arael**: SparseSchur solver deleted (it lost to faer on every benchmark and its hard-coded 6-DOF-pose/3-DOF-landmark contract failed silently outside BA-shaped problems -- REVIEW B26). The explicit Schur-elimination math is the natural seed for F2 sliding-window marginalization; recover it from git history (src/simple_lm.rs prior to the deletion commit) if F2 lands.

- **arael-sym**: investigate a first-class `branch()`/piecewise construct instead of heaviside-multiplication. Multiplying by `heaviside(c)` compiles to a select (csel/cmov), never a real branch: LLVM cannot skip evaluating the "off" arm (both operands of the multiply are computed), and it cannot fold `0 * x` to 0 because x may be inf/NaN -- so `H(c) * f + (1 - H(c)) * g` always pays for BOTH arms and propagates NaN/inf out of the disabled one. A `branch(cond, then_expr, else_expr)` expression node could codegen an inline `if`, evaluate only one arm, stay NaN-safe, and differentiate piecewise (derivative of the selected arm, like clamp's pass-through philosophy). Would also give safe formulations of log-map-style residuals (Taylor fallback arm near the singularity) without epsilon biasing.

- **arael**: LambdaDriver follow-ups. The driver abstraction and NielsenLambdaDriver are DONE and are the benchmarks/bal default (zero rejections on Ladybug-49, 2.5x on Ladybug-138 where no fixed-schedule knob pair worked, floor rendered moot). Remaining: consider making NielsenLambdaDriver the LIBRARY default after validating it across the pgo and sketch workloads (it must not regress the pose graphs' 5-7-step solves or the sketch's tiny systems).

- **benchmarks/bal**: Ladybug-1723 needs a tighter shared termination criterion before it can be a fair race: with the common 1e-5 tolerances no system converges -- they stop at plateaus spanning 0.9% (765376 to 772215, arael's the HIGHER cost) and the mutual validation gate cannot pass. Its damping IS now tuned (both f64 routes and both Ceres rows report a clean full-iter), so the iteration is measurable; only the stopping rule is missing.

- **benchmarks/bal**: arael f32 cannot run Ladybug-1723 (0 accepted steps). NOT a range-overflow-in-J^T J problem, which is what this file used to claim -- measured 2026-07-14: the residuals are tiny (largest 4.5e3, largest r2 158, nothing near f32's range). It is the DATA: 199 observations lie behind the camera and 14 sit on the optical centre, where pc.z = R*p + t.z cancels from terms of order 1 down to 3.65e-9. f32 has 7 digits, that cancellation needs 9, and thirteen of them land on exactly 0.0 -- the perspective divide becomes 0/0 and arrives as a NaN on the Hessian diagonal. No damping value changes it, and a depth clamp is NOT the fix (a one-sided floor teleports the behind-camera points to just in front of the lens and the cost explodes: +47,000% at floor 0.1, 1e36 at 1e-5; a sign-preserving magnitude floor at 1e-5 is bit-identical on the other three datasets and would work, but arael has no way to express it -- `clamp` is two-sided on the value and its derivative passes through). Wants either the data cleaned for every system alike, or residual suppression in the model layer (the real gap: nothing lets a constraint say "this residual is undefined here").

- **arael**: explicit Schur/marginalization support for camera-landmark problems -- DONE. Superseded by the two entries at the top of this file: the macro hands the eliminable blocks to the solver, and `SparseFaer` marginalizes them behind `SchurPolicy` (`Auto` by default). benchmarks/bal measures both routes as permanent rows.

- **arael**: investigate a conjugate-gradient Schur solver (the route Ceres calls `iterative_schur`): CG on the reduced camera system, applying it by multiplication instead of forming and factorizing it. Measured motivation (benchmarks/bal): against arael's factorizing Schur route its iteration goes 2.8x more expensive on Ladybug-49, 1.3x on 138, then 1.4x CHEAPER on 372 and 4.5x on Ladybug-1723 (397 vs 1806 ms) at half the peak memory (621 vs 1345 MB). It crosses over, and the lead grows with size -- so a factorization is the wrong tool above a few thousand cameras. Needs a preconditioner to be worth anything (Ceres defaults to Jacobi) and a stopping rule for the inner solve.

- **arael**: detect BAL-like structure and pick nested dissection automatically. The reduced camera system of a bundle problem is a union of CLIQUES (a 3D point makes a clique of every camera that sees it), which is what AMD is worst at, and the default ordering (`FaerOrdering::Auto` = AMD) is wrong there -- benchmarks/bal has to ask for nested dissection by hand (`BAL_ORDERING=nd`). Measured on Ladybug-1723: AMD factorizes the reduced system in 4716 ms (`--bin schur_stats`), where under nested dissection the WHOLE Schur iteration is 1806 ms, and the fill ratio the Schur policy decides on moves from 1.21 to 0.55 -- so AMD makes `SchurPolicy::Auto` DECLINE a reduction that is in fact the right call. The detection wants to be structural (eliminable blocks whose elimination cliques the kept ones), not a name-based special case.

- **arael**: a deep-analysis flag on the solver: instead of deciding the plan from a cheap heuristic, actually BUILD and BENCHMARK the candidates -- Schur or not, the orderings, the backends -- on the real problem, keep the fastest, and hand the winning plan back in `LmResult`. The caller then passes it to the next solve and skips the analysis entirely (the same problem is usually solved again and again). Motivation: the current fill-ratio heuristic gets Ladybug-372 and 1723 wrong under AMD (item above), and no static rule is going to be right across pose graphs, localization bands and bundle cliques. Costs one slow first solve; buys the right plan and, thereafter, no analysis at all.

- **arael-faer**: re-measure the transposed tile GEMM on x86. The fallback used to hand nano-gemm a strided lhs for a transposed `C_a`, which made it pack the tile into a 64 KB stack frame -- a flat cost that on x86 f64 was ~60 ns whatever the widths, and made (5,3,7) transposed 1.4x SLOWER than the runtime loop it replaced. It now transposes into a stack buffer up to [`TRANS_PACK_MAX`] and passes a column-major lhs instead, which on aarch64 removed the floor entirely ((5,3,7) transposed 27.4 -> 16.9 ns, vs the loop's 47.3). The x86 side of that is unmeasured -- `--example gemmbench`, both with and without `x86-v4`.

- **arael-faer**: nano-gemm's `x86-v4` (AVX-512) feature is forwarded but measured to HURT on the one x86 machine we have: every tile shape above nano-gemm's width gate (m > 4 for f64, m > 8 for f32) got 1.2-2.7x slower, every shape below it was unchanged. Kept anyway -- one machine, one nano-gemm version. Worth re-measuring on other x86 hardware with `--example gemmbench`, and dropping if it never wins. Note it is NOT forwarded from the `arael` root crate, so downstream cannot reach it; wire that up only if it is ever shown to pay.

- **arael**: order the eliminated blocks last, so the Schur reduction never sees a transposed tile. The Hessian stores each tile once, and which of `C_a` or `C_a^T` it holds depends only on whether the eliminated block's index is below or above the observer's. Today that falls out of the root struct's FIELD ORDER: `Path { poses, landmarks }` puts the eliminated landmarks above the observing poses and every tile lands direct (measured: slam and bal both give trans=0, all 4859 and 31843 tiles respectively). Write `Path { landmarks, poses }` and every tile is transposed instead -- a silent tax, paid for a cosmetic choice the author had no reason to think mattered. Measured on the slam benchmark at 300 poses, full-iter: 49.0 ms poses-first vs 50.8 ms landmarks-first (f64), and it was 54.7 ms before the transpose-first kernel. So the penalty is down from 12% to 4%, but it should be 0: the elimination order is arael's to choose, and it should choose it so the tiles come out direct, whatever order the fields are declared in.

- **arael-faer**: the shape dispatch is hoisted out of the pair loop only when every observer of an eliminated block has the same width AND the same storage orientation (`gemm_tri`). When they differ the loop falls back to dispatching per tile. Not generalized because no model measured has a mixed slot: the entity marginalized out is seen by one kind of entity, so bal (cameras through a point) and slam (poses through a landmark) are both uniform. The general form is to sort each block's observers into maximal runs of equal (width, orientation) -- they are already in kept-id order, so runs are contiguous and every pair of runs is a rectangle of constant shape -- and dispatch once per run pair. It costs a run table in `SchurSymbolic` and a rectangle-major pair emission order. Worth doing when a model with mixed observer widths per marginalized entity actually shows up.

- **arael-faer**: batched tile GEMM. The reduction issues a long run of `dst -= C_a * Z_b` at the SAME shape -- one per observer pair -- and each is a separate call that re-enters the kernel, re-reads the widths and re-establishes its registers. Try widening it: hand 2, 3 and 4 tiles to one call and do them inlined, both sequentially (back-to-back in one body, so the loads of one overlap the arithmetic of the last) and in parallel (interleaved in the same loop, so independent chains fill the pipeline instead of stalling on a dependency). The unrolled kernels are the place to try it -- they already know the shape at compile time, so a batch of 4 is just a wider const-generic body. Measure each width: past some point register pressure spills and it turns into a loss, and where that point is has to be found, not guessed. `SchurSymbolic::gemm_shapes` already reports the pair count per shape, so the batching opportunity is known before the loop runs. nano-gemm offers nothing here: it has no batch API, and its `microkernels`/`millikernel` fields are private, so its indirect call cannot be hoisted out of a run either.

- **arael-faer**: measure the `gemm` crate (the general dense kernel) for the tile GEMM, as an alternative to nano-gemm and our const-generic kernels. Never measured.

- **arael**: look into whether faer can be made faster on the pose-graph benchmark problems. On the 3D datasets faer's factorize+solve dominates the LM step (~8.5 of arael's 10.6 ms/step on parking-garage; assembly is only ~1.9 ms) and trails GPL supernodal CHOLMOD (10.6 vs 8.9 ms/step garage, 19.4 vs 17.7 sphere2500) while clearly winning the 2D datasets. Things to investigate: whether we drive faer's supernodal path and threshold knobs optimally for 6-DOF block structure, fill-reducing ordering choices, reusing more of the symbolic work across damped retries, and whether upstream faer has (or would take) improvements for block-sparse SPD problems of this shape.

- **arael**: consider an opt-in supernodal CHOLMOD backend. Measured on the 3D pose-graph benchmark: Eigen::CholmodSupernodalLLT steps at 8.9 ms on parking-garage and 17.7 on sphere2500 vs faer's 10.6 / 19.4 (faer clearly wins the 2D datasets: 3.5 / 12.7 vs 5.0 / 20.0). Much stronger on the BA-fill SLAM benchmark (SLAM_POSES=300, 5400 params, 770k Hessian nnz): flipping the cholmod shim to CholmodSupernodalLLT drops arael f64 from ~112 to ~66 ms/iter (interleaved, same optimum) -- ahead of g2o's 86.7, and arael factorizes the FULL 5400 system where g2o Schur-reduces to 1800 first. The linear solve is ~96 of faer's ~109 ms/iter there, so the factorization is the whole gap: faer loses ~2x to supernodal on dense-fill BA structure. g2o's internal split (G2O_STATS=1 in the slam runner) per 86 ms iteration: residuals 2.4, assembly ~6, Schur formation ~29, CHOLMOD numeric factorization only ~10-12 on the reduced system, total linear solution ~42. Combining supernodal with the F2 Schur/marginalization item could plausibly land near ~40 ms/iter. Not adopted because CHOLMOD's Supernodal module is GPL (the simplicial module our `cholmod` feature binds is LGPL); shipping it even behind a feature flag makes the resulting binary GPL. If ever added it must be a loudly-documented separate opt-in (e.g. `cholmod-gpl`) that users enable knowingly.

- **arael**: rotation-param uniform read-back (the "correct fix"). The semi-correct fix landed 2026-07-10: all three SO(3) primitives share one contract -- `value` is the initial guess in and the optimized orientation out, synced only by `deserialize(&result.x)` (mandatory after a solve); the working state is solver-internal (`work` for Simple, `ref_rotation` for Euler, a private `ref_value` unit quaternion for Quaternion), fixed params seed their reference from `value` at serialize (they used to evaluate as identity in constraints -- tests fixed_{euler_angle,quaternion}_param_drives_constraints), `update_self` re-derives the working state from `value`, and QuaternionParam's deserialize folds the handed-back delta with the solver's own rotation-vector retraction instead of euler angles, without mutating the reference (idempotent). REMAINING: make the external 3-number form for the delta primitives a ROTATION VECTOR (log map of the reference) instead of a delta that is always zero -- gimbal-free (singular only at angle = pi, not pi/2) and its exp/log IS the solver's own retraction (`from_rotation_vector_small`), so serialize and deserialize would carry the actual orientation and `result.x` would become self-contained for rotation params (today the result is only reachable through the deserialized model, not from `result.x` alone). Also remaining, EulerAngleParam-specific: its deserialize folds the delta into `ref_rotation`, so repeated deserialize calls would double-fold a nonzero delta (dormant, the solver always hands back delta = 0), and its euler read-back keeps the inherent pitch = +-90 recomposition loss (`get_euler_angles`/`from_euler_angles` degenerate there with ~sqrt(eps) error).

- **security**: remove the RUSTSEC-2026-0194/0195 (quick-xml < 0.41 DoS) ignores from `.cargo/audit.toml` once `cargo update` can resolve quick-xml >= 0.41 across the tree. Blocked upstream as of 2026-07-04: wayland-scanner (latest 0.31.10) and zbus_xml (latest 5.1.1) both pin `^0.39`, and the zbus_xml 4.0 in our tree (via atspi/zbus-lockstep) pins 0.30. Exposure is build-time codegen and local-session-bus XML only, hence the ignore rather than a workaround.

- **arael-macros**: nested struct support in the model tree -- DONE (06e8f15 + 8ffd8ed). Block-bearing entities and constraints now codegen at any depth below the root, through block-less grouping sub-models and collections-of-sub-models (`Map { paths: Vec<Path> }` with `Path { poses, pose_pairs, frines }`). `resolve_nested_path` walks the registry to a multi-segment `AccessSegment` path (`EntityLocation::Nested`); self-block / cross / set_block_indices emission wrap the per-entity loop in the nested prefix (`nested_container` / `wrap_in_prefix`); `parent.<coll>` refs resolve against the containing sub-model, `root.<coll>` against the root; passive nested entities get their SelfBlock wired. One-hop emission byte-identical (verified via cargo expand on all model examples + the full arael-sketch-solver lib). Tests: nested_self_block, nested_cross_block; demo: slam2d_multi_demo. Unblocks R6.

- **arael**: port the COO-free first iteration to the other scalar-CSC backends -- DONE 2026-07-11. The first-call assembly is now one shared function (`assemble_first_csc` in simple_lm), used by SparseFaer, SparseEigen, SparseCholmod and SparseCholmodSupernodal alike: models with a statically-knowable pattern build their CSC and position map from the block structure, everything else falls back to COO. Interleaved at slam-300, first iteration: eigen 431.6 -> 416.2 ms, cholmod 433.4 -> 420.9, cholmod-gpl 111.3 -> 105.2; steady-state ms/iter unchanged on all three, so the tile-expanded pattern's ~1.2% explicit zeros cost the CHOLMOD and Eigen factorizations nothing measurable (the open question when this was deferred). First-assembly means from SLAM_TIMING confirm the mechanism: 12.92 -> 7.69 ms (cholmod-gpl), 13.16 -> 9.72 (eigen). Tests: eigen/cholmod/cholmod-gpl backends each land on the dense optimum (tests/block_assembly.rs, feature-gated -- they had no tests at all before).

- **arael-faer**: parallelize the Schur reduction's GEMMs. `with_threads` above
  buys faer's threads for the FACTORIZATION; it does nothing for the reduction,
  which is our own code and, at slam-300, 94% of `schur_reduce` (measured stage
  timings: seed 0.05, factor 0.17, panel 1.78, GEMM 36.5, rhs 0.31, finish 0.01
  ms -- the whole `Hee^-1` side, 1200 Cholesky factors and their triangular
  solves, is under 2 ms). So the reduction is a GEMM loop and nothing else, and
  it is the half of the Schur route faer cannot thread for us.

  It parallelizes almost perfectly in principle: `Hee` is block-diagonal, so
  every eliminated block's contribution is independent -- that independence is
  the whole reason the reduction decomposes per landmark. The catch is the
  destination: two landmarks seen from the same pair of poses accumulate into
  the SAME tile of S, so a naive `par_iter` over eliminated blocks races. Three
  ways out, in rising order of effort: (a) per-thread S buffers summed at the
  end -- trivial, but S is ~260 MB at 6000 poses so it costs memory per thread;
  (b) partition the eliminated blocks by their observer sets so no two threads
  target the same tile (a graph coloring, computed once in `schur_symbolic`,
  which already knows every pair target); (c) invert the loop and parallelize
  over S's tiles instead of over landmarks, each thread owning its own
  destinations and gathering the contributions -- no races, no extra memory,
  but it needs the transpose of `pair_dst` (tile -> contributing landmarks),
  which is also a one-time symbolic build. (b) or (c) is the real answer.
  Needs the same `rayon` cargo feature as `with_threads`.

- **arael**: threading for the Schur route's S factorization. INVESTIGATED 2026-07-11 (benchmarks/slam sfactor_bench): the earlier premise that faer trails CHOLMOD here was WRONG -- a misread G2O_STATS dump. Re-measured, g2o's CHOLMOD supernodal numeric on the same 1800-parameter reduced system takes 22.5 ms/iter against faer's 21.9: parity. faer's auto threshold already selects supernodal (identical to FORCE_SUPERNODAL; forcing simplicial is 7x slower), and at ~50 GFLOP/s it runs at the machine's dense-kernel rate, so there is no kernel headroom to reclaim single-threaded. A dense LLT on S is slower (37.3 vs 21.9 ms at 300 poses) despite S being 69% dense, because the sparse factor is half the size. What is left: more cores (faer's Par::rayon on the numeric factorization, tied to the with_threads item above) and fewer flops (a structurally different reduction). Not band+border: the wide landmarks make S dense, so there is no band to exploit.

- **arael-faer**: dedicated README.md -- DONE (arael-faer/README.md, mirrored in src/lib.rs)

- **benchmarks/bal**: decide whether to ship a permanent `arael LM f64/f32 schur` row -- DONE (2026-07-14). Shipped: every table carries four arael rows, f64/f32 x sparse/schur. The "loses badly at Ladybug-1723" reading that made this a question was an AMD artifact -- under nested dissection (the ordering entry below) the Schur route has the cheaper iteration on EVERY dataset, 1806 vs 2910 ms even at 1723. `--bin schur_stats` still reports S's size, density and factorization split per dataset, but it orders S with AMD, so its factorization times are not the benchmark's.

- **arael/arael-faer**: fill-reducing ordering for a LARGE SPARSE Schur complement -- DONE. `FaerOrdering::NestedDissection` exists and benchmarks/bal orders the reduced camera system with it. It is the whole ballgame on the clique structure a bundle problem's S has: AMD needs 4716 ms just to factorize S at Ladybug-1723, where under ND the entire Schur iteration -- assembly, reduction, factorization, solve -- is 1806 ms. What is NOT done is choosing it automatically; see the BAL-structure-detection entry above (arael's default is still AMD, and under AMD's fill `SchurPolicy::Auto` declines a reduction it should take).

- **arael**: keep the Schur structural analysis across solves. SHIPPED as `LmSession` (REVIEW3 item 3): the session owns the backend and its matrix storage, skips the entry `reset()`, and reuses the pattern, position map, ordering, symbolic factorization and Schur plan on every solve after the first. A parameter-count change invalidates by itself; `invalidate()` covers structure changes at the same count. Warm solves are bit-identical to cold ones (tests/lm_session.rs). NOT done: arael-sketch integration -- a session field on `Sketch` (take/put around `Sketch::solve()`) with action-level invalidation, so graduated stages and drag re-solves go warm; left out because it touches the sketch crate's action layer, a separate change from the library API.
- **arael-faer**: narrow-band solver for the reduced system. SHIPPED (opt-in, `arael_faer::band`). Envelope Cholesky over the reduced Schur system in block form: fill confined to each column's envelope, no scalar-CSC round trip, no symbolic/ordering pass; reuses schur.rs's unrolled tile kernels (`gemm_sub`). `SparseFaer::with_narrow_band(true)` routes any banded system through it, whatever the bandwidth -- the caller's explicit choice; faer is the fallback for non-banded systems. Applies to BOTH the reduced Schur system (when it reduces) AND the whole Hessian (when it does not -- pose graph / localization); `setup_whole_band` handles the non-reduced banded case. Loc benchmark (`LOC_ARAEL_SOLVER=narrow_band`): the whole-system block route is a WASH against the tuned scalar `Band` solver at kd=11 (17.4 ms/iter both at 1000 poses) and ~3-5% ahead of faer -- at this bandwidth it does not beat scalar, but it auto-detects bandwidth and needs no manual kd. `SchurPlan::narrow_band` reports whether it was taken; a `warn!` fires when the half-bandwidth exceeds `NARROW_BAND_WIDE_KD` (128). Slam benchmark hook: `SLAM_NARROW_BAND=1` (and `SLAM_SPAN=N` caps landmark span to make a narrow-band scene). Benchmarked (interleaved, this dev VM, f64 ms/iter): it WINS in the narrow regime and loses in the wide one -- band vs faer-supernodal ratio at slam-1200 was 0.86 (kd 60) / 0.97 (kd 120) / 1.05 (kd 240), holding at 6000 poses; first iteration is a bigger win (0.73 at kd 24) because that is where faer pays for symbolic + scalar-CSC setup. Crossover ~kd 130-150 (hardware-dependent). NOT done: the "+ border" case (wide landmarks, loop closures, global params -- SCHUR.md) for band-plus-a-few-violations systems; supernodal panel blocking to compete on WIDE bands (large, and largely re-implements what faer supernodal already does well -- low ROI); a cheaper low-effort squeeze is unrolling the 6x6/3x3 diagonal Cholesky + triangular tile solves. A general BAL-style camera set is NOT banded (Ladybug-1723's S loses), so the route is trajectory/local-feature-shaped only.

- **Schur: S is stored twice** -- once as blocks (what `schur_reduce` writes) and once as the scalar CSC values faer factorizes, ~260 MB each at 6000 slam poses. Tried removing the block form: give every S tile a strided view into the CSC values (a tile's rows are contiguous, its columns one block-column of nonzeros apart) so the reduction accumulates straight into the array the factorization reads. It works and is exact, but it is SLOWER, consistently, and was rejected: the GEMM's destination tile stops being contiguous, and the 6 columns of a 6x6 tile land ~14 KB apart. Interleaved A/B at 6000 poses -- f64 2310 -> 2344 ms/iter, f32 1612 -> 1721 (+6.8%); at 300 poses f64 49.2 -> 50.2. Memory did fall (f64 peak 2476 -> 2247 MB) but only for f64; the f32 peak did not move at all, because it is set during the symbolic phase, before S exists. The double storage is the price of handing S to a SCALAR sparse factorization, so the way out is not a cleverer copy -- it is the band/block solver above, which factorizes S in the block form the reduction already produces and needs no scalar CSC at all.

- **arael**: `LmConfig` presets instead of `Default` -- DONE (2026-07-18,
  f26fe6d): `conservative()` (the `Default`), WIP `well_conditioned()` /
  `ill_conditioned()` modeled on the slam / BAL benchmarks, `continue_from`,
  `with_nielsen`, per-field `with_*` builders. Examples and docs adopted the
  presets via the builder form (8333c28).

- **arael**: predictive, Ceres-style termination -- DONE (2026-07-17,
  REVIEW3 item 8): `LmConfig::predicted_reduction_tolerance: Option<T>`
  (default None), checked on accepted steps, reports
  `LmStatus::PredictedReduction`, respects `min_iters`. Tests in
  tests/termination_semantics.rs.

- **arael**: audit every panic / unwrap / expect on the solve path -- DONE
  (2026-07-16, REVIEW3 item 5), shipped WITHOUT the API break this entry
  expected: the seven structural panics surface as
  `LmStatus::SetupFailed(SolveError)` on the ordinary `LmResult` (parameters
  unchanged, NaN costs, zero iterations), entry points unchanged.
  `LmSolver::compute` and `CooMatrix::to_csc{_with_map}` return `Result`
  internally. Tests: tests/degenerate_model.rs, tests/sparse_faer_routes.rs.

- **arael macro**: multi-output constraint CSE -- RESOLVED (2026-07-16,
  REVIEW3 pre-check): the premise was wrong -- CSE already runs once across
  all outputs of a multi-output constraint. No change needed.

- **arael**: landmark-folding Schur for covariance. The `SparseFaer` solver already marginalizes landmark-like blocks (a Schur complement) when it is faster; `assemble_covariance` does not -- it factors the whole system. Investigate reducing over the landmarks for covariance too, so camera/pose marginals come from the smaller reduced factor (as g2o's `computeMarginals` does in the bal/slam benchmarks). Requested by the user 2026-07-16.

- **arael**: a single generic `model.solve(solver, &cfg)` -- DONE
  (2026-07-18, b313ef8): `SolverKind` enum (Dense / Band / BandLapack /
  Sparse(SparseFaerOptions) / Eigen / Cholmod / CholmodSupernodal),
  `LmProblem::solve(kind, &cfg)` via `BackendScalar` per-scalar dispatch;
  a backend not compiled in (or CHOLMOD at f32) returns
  `SolveError::SolverUnavailable` instead of failing to build. The old
  entry points remain.

- **arael-sym**: piecewise builtins beyond `min`/`max`/`sign`. `min`, `max`, and `sign` are implemented (branch/heaviside-based, in `FUNCTIONS`). REVIEW3 item 11 also listed `floor`, `ceil`, `round`, `hypot`, `powi`, `cbrt`, `rem`. Not done: `floor`/`ceil`/`round`/`rem` are piecewise-constant or discontinuous with a subgradient story that needs its own design (a plain 0 derivative is wrong for `rem`), and `hypot`/`cbrt`/`powi` are smooth conveniences that duplicate existing `sqrt`/`pow` compositions with no user having asked for them yet. Add on demand.

- **benchmarks/pgo + arael macro**: SE3 QuaternionParam compile time -- DONE
  (2026-07-17, c3dbee7 + 9c647db): not by shrinking the derivative but by
  making its construction cheap -- single-pass cached() substitution
  (cse::replace_many) and E::diff building raw with one identity-preserving
  simplify() at the end. pgo SE3 macro expansion ~37s -> ~7s, generated code
  byte-identical. Original entry: the SE3 model compiles 3x slower under `QuaternionParam` than under `EulerAngleParam` (pgo bench crate: 32 s -> 92 s incremental, measured 2026-07-13). The switch was made because the euler-angle delta undershoots on large initial rotation errors: on sphere2500 arael's first step cut the cost by 5% where factrs's exponential-map retraction cut it by 75%, costing arael a whole extra iteration (7 vs 6). `QuaternionParam` closed that gap exactly. The compile cost is the macro symbolically differentiating `QuaternionParam::rotation_matrix()` -- a much larger expression tree than the euler composition -- for every residual component against all 12 parameters of a pose-pose edge, twice (f64 and f32 models). Worth investigating whether the generated derivative can be shrunk: the rotation-vector retraction has a lot of shared structure (the delta is small by construction, and `ref_rotation` is a constant within an iteration), so common-subexpression extraction or treating `ref_rotation` as a factored-out constant may cut the tree substantially. 3x compile time for a 6-DOF pose model is steep enough that a user could reasonably choose the worse retraction to avoid it.

- **benchmarks/plane + arael**: sensor-extrinsic calibration example, ported
  faithfully from g2o's plane_slam demo (its actual subject: 3 orthogonal
  always-visible planes, wiggle trajectory, sensor SE3 offset optimized from
  a large initial error, offset marginal covariance as the output). Deferred
  because the observation becomes a THREE-variable constraint (pose, plane,
  offset) and component params were guarded off in multi-block/triplet
  span builders. That guard is LIFTED (2026-07-24: every span builder
  walks `param_slots`, so component params work in triplet, multi-cross,
  `root.<selfblock>` and `[hb, root.<triplet>]` forms --
  tests/macro_matrix_components.rs), so the example is now buildable
  (also a natural showcase for assemble_covariance). The plane benchmark
  keeps a degenerate all-visible env for the Schur-gate stress case
  meanwhile.

- **arael: Schur Auto gate misfires when a small block family is observed
  by everyone** -- DONE (2026-07-19). The shortcut now prices the
  alternative: it fires only when the reduced route (reduction +
  worst-case S factorization) stays within `obvious_flop_ratio` (25) of a
  fill-free floor under any whole-system factorization,
  `max(nnz(H), nnz(H)^2/n)`; and the exact comparison is a flop crossover,
  `reduce_flops + factor_flops(L_S) > flop_margin * factor_flops(L_H)`
  (sum-of-squared-column-counts from the symbolic factors; `flop_margin`
  1.5 replaces `fill_ratio_max`). The shared-plane scene declines (arael
  4.8 -> 0.5 ms/iter, slowest of 8 systems to fastest); slam and BAL
  routes unchanged, measured ratios in the SchurPolicy docs. Also fixed
  en route: block CSC patterns overcount the scalar triangle by the
  diagonal tiles' lower halves (the "101% dense" print) --
  `scalar_upper_nnz()` corrects every density the gate reads. Test:
  tests/selection.rs a_small_global_family_is_declined. Original entry:
  the `obvious_flop_ratio` shortcut compared worst-case S factorization
  against the reduction's own cost only, treating the reduction as sunk;
  a small family observed by everyone made both expensive while H stayed
  nearly banded, and the one comparison that would have declined was
  skipped.

- **arael-macros: a model field named like the root type (or like its own
  struct) broke emission** -- DONE (2026-07-20). The cause was NOT binding
  resolution as first filed: `rename_ident` (constraint.rs), which maps
  body-binding names to the emitted access (`<root_lc>` -> `self`,
  `<struct_lc>` -> `__item`), renamed EVERY matching identifier token,
  including ones in field position. A field `m2` under root `M2` emitted
  `__item.self.x` (rustc E0609 at the owner); a field named like its own
  struct's lowercase would have emitted `__item.__item.x`. Fixed by
  renaming only identifiers in VARIABLE position -- not after `.` (field
  or method) or `:` (path segment). Verified a no-op for existing models:
  generated code is byte-identical for the plane benchmark, loc_global_demo
  (root params + TripletBlock), slam_demo, m3500_demo and root_fit_demo.
  Test: tests/name_collisions.rs.

- **arael: `BArena` / `BRef` for collections past 16.7M elements** -- a
  `Ref` packs a 24-bit index and an 8-bit generation into one `u32`
  (`refs.rs`), so every collection tops out at `MAX_INDEX` = 16,777,215
  elements and an `Arena` slot repeats its generation after 256 recycles.
  Nothing in the tree is near either bound: the largest collections are
  BAL's points and pgo's city10000 poses, orders of magnitude under. The
  escape hatch, when something does need it, is a parallel `BRef` holding a
  full `u32` index and a full `u32` generation -- 8 bytes, no wrap -- and a
  `BArena` that uses it. Deferred because it is speculative and not free:
  `arael-macros` keys on the literal type name `"Ref"` in about ten places
  (`extract_wrapper_inner(ty, "Ref")` in lib.rs and constraint.rs), each of
  which would have to accept `BRef` too, plus a parallel `Index`, iterator
  and serde surface. Note the ceiling lands on `Vec` as well, and `Vec` is
  what holds the large collections -- so if scale ever bites it is a `BVec`
  that is wanted, built the same way.

- **cargo-arael: LmSession over the FFI.** DONE (2026-07-31,
  docs/dev/EXPORT.md item 5): a per-root `LmSession` object in both
  skins -- constructed over optional `SparseOptions`, `solve(model,
  cfg)`, `invalidate()` -- mirroring the Rust session (warm solves
  bit-identical to cold, parameter-count change re-analyzes by
  itself). Both slam_demo_gm drivers now carry the ramp through one
  session, like the Rust example. Original note: the generated solve
  calls were stateless, so the ramp re-analyzed the sparsity every
  pass.

- **cargo-arael: no gradient/Hessian assembly surface.** The Rust
  slam_demo_gm's env-gated Hessian-sparsity bitmap uses
  `calc_grad_hessian_sparse`; the C++ twin drops that debug feature
  because the generated API exposes solves and covariance only.
  Exposing raw assembly over the FFI is a large surface for a debug
  nicety -- revisit if a real consumer needs it.

- **arael-macros: single-instance entity field (bare, non-Option).** A
  constraint-bearing entity must live in an iterable container -- the
  generated assembly enumerates instances with `.iter()` -- so the only
  way to attach exactly one is `Option<Entity>` kept always `Some`, e.g.
  the pgo/m3500 gauge prior `Option<Prior<T>>`. A bare `prior: Prior<T>`
  field would read more honestly (the anchor always exists) but does not
  compile (`no method named iter found for Prior<T>`). Deferred because it
  is a core traversal change: the model walker, serialize/deserialize,
  covariance block enumeration, and the sidecar/export each treat entity
  fields as containers and would need to accept a directly-nested entity as
  a one-element set. `Option<Entity>` is idiomatic meanwhile.
