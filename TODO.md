# TODO

- **Automatic eliminate_first detection** (follow-up to the
  `eliminate_first(...)` root keyword, 2026-07). The hint could be derived
  from the Hessian pattern alone: group columns into blocks, greedily pick a
  maximal independent set of small blocks (structural landmarks), order it
  first. Any ordering is solution-safe, but since hints are trusted
  outright (no comparison against AMD), an automatic picker would need its
  own quality guard -- e.g. compare symbolic L nnz against AMD's before
  adopting, the check the explicit keyword deliberately does not do. Not
  built because the explicit hint covers the known use cases and the model
  knows its own structure; revisit if hand-built LmProblem users without
  macro models need it.

- **Explicit blocked Schur-complement backend**. The eliminate_first
  ordering closes most of the gap to g2o on landmark SLAM, but an explicit
  Schur backend (form S = Hpp - Hpl Hll^-1 Hlp with blocked 3x3/6x6
  kernels, factorize the reduced pose system, back-substitute) measured
  ~1.7x faster still on the linear-solve phase in g2o's stats (42 vs 71 ms
  at 300 poses). A new LmSolver backend + block-partition machinery; only
  worth it if solve-dominated problems become the primary target.
  Ordering refinement is NOT the path to that headroom: a prototype that
  ordered the trailing pose blocks by AMD on the reduced pose graph
  (via an eliminate_first range sequence) measured a wash to -2% vs
  trajectory order at 300/600 poses -- after landmark elimination the
  reduced system is ~70% dense, so ordering barely moves the factor, and
  trajectory order feeds the supernodal kernels contiguous blocks.
  Reduced-graph AMD would only matter on problems whose reduced system
  is genuinely sparse (BA-style covisibility), where trailing natural
  order is the failure case.

- **SparseFaer::with_threads(n): opt-in multithreaded faer solve**
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

- **lm_resolve(): warm re-solve for arael-sketch dragging** (design agreed
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
- **arael-macros**: support general func() syntax -- right now we have to describe all of them which is annoying..
- **arael**: support single struct model+root. right now it does not function. -- DONE (SelfBlock<Self> on root + direct-composed sub-model fields now route through EntityLocation::RootSelf / EntityLocation::DirectField in arael-macros)
- **arael**: support global optimization parameters with a triplet block, so a global param can be mixed with hessianblock, or efficiently omitted
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
- **arael-macros**: tighten the constraint attribute parser to reject the N-way positional multi-block form `constraint(hb_a, hb_b, hb_c, {body})`. The parser currently accepts it as a side effect of supporting `constraint(hb_pose, root.hbt, {body})` (self-primary + root-owned TripletBlock). N ≥ 2 regular multi-block should always be written with brackets -- `constraint([a, b, c], {body})` -- and the only positional form we genuinely need is the specific 2-item `(<local_self_block>, root.<triplet>)` case. Two parsers for the same thing is a surprise waiting to bite users.
- **arael-sketch**: investigate chain misbehavior
- **arael-sketch**: when selecting a line/point/arc/etc hilight/hover all the constraints associated with it

- **arael**: geometry helpers deferred from the F6 math-coverage review (new feature surface rather than sym/runtime parity; nothing in the repo blocks on them): matrix2/matrix3 `inverse`, matrix -> quaternion conversion, SE(2)/SE(3) compose/between helpers, skew/hat operator (`[v]_x` from a vect3).
- **arael-sym**: quaternsym operations deferred from F6 stage E: `pow`/`log`/`exp`/`slerp`/`from_two_vectors`/`get_axis_angle`. All are branchy (sign flips, acos edge cases, zero-angle guards) and not residual-friendly; the runtime `quatern` has them for data preparation.
- **arael**: SparseSchur solver deleted (it lost to faer on every benchmark and its hard-coded 6-DOF-pose/3-DOF-landmark contract failed silently outside BA-shaped problems -- REVIEW B26). The explicit Schur-elimination math is the natural seed for F2 sliding-window marginalization; recover it from git history (src/simple_lm.rs prior to the deletion commit) if F2 lands.

- **arael-sym**: investigate a first-class `branch()`/piecewise construct instead of heaviside-multiplication. Multiplying by `heaviside(c)` compiles to a select (csel/cmov), never a real branch: LLVM cannot skip evaluating the "off" arm (both operands of the multiply are computed), and it cannot fold `0 * x` to 0 because x may be inf/NaN -- so `H(c) * f + (1 - H(c)) * g` always pays for BOTH arms and propagates NaN/inf out of the disabled one. A `branch(cond, then_expr, else_expr)` expression node could codegen an inline `if`, evaluate only one arm, stay NaN-safe, and differentiate piecewise (derivative of the selected arm, like clamp's pass-through philosophy). Would also give safe formulations of log-map-style residuals (Taylor fallback arm near the singularity) without epsilon biasing.

- **arael**: LambdaDriver follow-ups. The driver abstraction and NielsenLambdaDriver are DONE and are the benchmarks/bal default (zero rejections on Ladybug-49, 2.5x on Ladybug-138 where no fixed-schedule knob pair worked, floor rendered moot). Remaining: consider making NielsenLambdaDriver the LIBRARY default after validating it across the pgo and sketch workloads (it must not regress the pose graphs' 5-7-step solves or the sketch's tiny systems).

- **benchmarks/bal**: Ladybug-1723 needs a tighter shared termination criterion before it can be a fair race: with the common 1e-5 tolerances neither arael nor Ceres converges -- they stop at plateaus 1.4% apart (arael's the lower cost) and the mutual validation gate cannot pass. Also: arael f32 fails there before the first step -- the f32 assembly produces a NaN Hessian diagonal (range overflow accumulating J^T J at 485k parameters); worth investigating whether assembly can saturate or rescale instead.

- **arael**: explicit Schur/marginalization support for camera-landmark problems. Measured motivation (benchmarks/bal, Ladybug-49): Ceres dense_schur steps at 20.6 ms vs arael's full-system 23.5 ms, while Ceres itself doing the full-system strategy costs 38.8 ms/step -- arael's ordering-driven elimination already beats non-Schur Ceres 1.65x, and the remaining ~3 ms/step is exactly the explicit camera-Schur advantage (grows with problem size). Ties into the F2 marginalization seed (recover deleted SparseSchur math from git history).

- **arael**: look into whether faer can be made faster on the pose-graph benchmark problems. On the 3D datasets faer's factorize+solve dominates the LM step (~8.5 of arael's 10.6 ms/step on parking-garage; assembly is only ~1.9 ms) and trails GPL supernodal CHOLMOD (10.6 vs 8.9 ms/step garage, 19.4 vs 17.7 sphere2500) while clearly winning the 2D datasets. Things to investigate: whether we drive faer's supernodal path and threshold knobs optimally for 6-DOF block structure, fill-reducing ordering choices, reusing more of the symbolic work across damped retries, and whether upstream faer has (or would take) improvements for block-sparse SPD problems of this shape.

- **arael**: consider an opt-in supernodal CHOLMOD backend. Measured on the 3D pose-graph benchmark: Eigen::CholmodSupernodalLLT steps at 8.9 ms on parking-garage and 17.7 on sphere2500 vs faer's 10.6 / 19.4 (faer clearly wins the 2D datasets: 3.5 / 12.7 vs 5.0 / 20.0). Much stronger on the BA-fill SLAM benchmark (SLAM_POSES=300, 5400 params, 770k Hessian nnz): flipping the cholmod shim to CholmodSupernodalLLT drops arael f64 from ~112 to ~66 ms/iter (interleaved, same optimum) -- ahead of g2o's 86.7, and arael factorizes the FULL 5400 system where g2o Schur-reduces to 1800 first. The linear solve is ~96 of faer's ~109 ms/iter there, so the factorization is the whole gap: faer loses ~2x to supernodal on dense-fill BA structure. g2o's internal split (G2O_STATS=1 in the slam runner) per 86 ms iteration: residuals 2.4, assembly ~6, Schur formation ~29, CHOLMOD numeric factorization only ~10-12 on the reduced system, total linear solution ~42. Combining supernodal with the F2 Schur/marginalization item could plausibly land near ~40 ms/iter. Not adopted because CHOLMOD's Supernodal module is GPL (the simplicial module our `cholmod` feature binds is LGPL); shipping it even behind a feature flag makes the resulting binary GPL. If ever added it must be a loudly-documented separate opt-in (e.g. `cholmod-gpl`) that users enable knowingly.

- **arael**: rotation-param uniform read-back (the "correct fix"). The semi-correct fix landed 2026-07-10: all three SO(3) primitives share one contract -- `value` is the initial guess in and the optimized orientation out, synced only by `deserialize(&result.x)` (mandatory after a solve); the working state is solver-internal (`work` for Simple, `ref_rotation` for Euler, a private `ref_value` unit quaternion for Quaternion), fixed params seed their reference from `value` at serialize (they used to evaluate as identity in constraints -- tests fixed_{euler_angle,quaternion}_param_drives_constraints), `update_self` re-derives the working state from `value`, and QuaternionParam's deserialize folds the handed-back delta with the solver's own rotation-vector retraction instead of euler angles, without mutating the reference (idempotent). REMAINING: make the external 3-number form for the delta primitives a ROTATION VECTOR (log map of the reference) instead of a delta that is always zero -- gimbal-free (singular only at angle = pi, not pi/2) and its exp/log IS the solver's own retraction (`from_rotation_vector_small`), so serialize and deserialize would carry the actual orientation and `result.x` would become self-contained for rotation params (today the result is only reachable through the deserialized model, not from `result.x` alone). Also remaining, EulerAngleParam-specific: its deserialize folds the delta into `ref_rotation`, so repeated deserialize calls would double-fold a nonzero delta (dormant, the solver always hands back delta = 0), and its euler read-back keeps the inherent pitch = +-90 recomposition loss (`get_euler_angles`/`from_euler_angles` degenerate there with ~sqrt(eps) error).

- **benchmarks/loc**: f32-vs-f64 parity is explained on the dev machine; the Cortex-A76 flip still needs an on-device run. Instrumented finding (LOC_TIMING=1 mode, aarch64 VM, steady-state per-phase means at 60 and 300 poses): the band linear solve is only 3-4% of an iteration -- assembly is ~65% and trial-cost eval ~30%, i.e. per-observation constraint evaluation dominates, and that code is straight-line SCALAR floating point. On a big out-of-order core scalar f32 is not faster than scalar f64 (microbenchmark on the VM: sqrt 0.54 vs 0.52 ns, div 0.50 vs 0.53, dependent mul-add chain 1.26 vs 1.31 -- parity everywhere), so the only f32 gain left is libm: atan2f 4.71 vs atan2 5.65 ns (17%). The bearing residual calls atan2 twice per observation (~11.3 of cost eval's 23.6 ns/obs, ~48%), which QUANTITATIVELY reproduces the measured ratios (predicted f32/f64 0.92 for cost eval vs 0.939 measured; 0.96 for assembly vs 0.966). f32 wins big only where SIMD or a weak FPU is in play: slam's faer solve (SIMD kernels, f32 doubles lane width) and the Pi Zero's VFPv2 (slow f64). The fast-atan route is implemented as `#[arael(root, fast_atan)]`, measured on the loc models (interleaved, LOC_TIMING, aarch64 VM, f64: cost eval -50% and assembly -12% at 60 poses, full iteration -24%; -22%/-6.6% at 300, -11%), and ADOPTED in the benchmark (the shared initial-cost cross-check runs at 1e-5 to accommodate it). CIRCUMSTANTIAL CONFIRMATION of the A76 mechanism: the Pi 5 rerun with fast_atan (output.pi5.updated.txt, 2026-07-10) shows f32 faster than f64 again (1.03 vs 1.09 ms/iter, previously 1.35 vs 1.28 with exact atan) -- removing libm atan2 from the pipeline removed the flip, consistent with the glibc-atan2f suspicion. Direct proof would be `cargo run -r --bin fpbench` on the Pi (atan2f vs atan2 timings); optional now.

- **security**: remove the RUSTSEC-2026-0194/0195 (quick-xml < 0.41 DoS) ignores from `.cargo/audit.toml` once `cargo update` can resolve quick-xml >= 0.41 across the tree. Blocked upstream as of 2026-07-04: wayland-scanner (latest 0.31.10) and zbus_xml (latest 5.1.1) both pin `^0.39`, and the zbus_xml 4.0 in our tree (via atspi/zbus-lockstep) pins 0.30. Exposure is build-time codegen and local-session-bus XML only, hence the ignore rather than a workaround.

- **arael-macros**: nested struct support in the model tree -- DONE (06e8f15 + 8ffd8ed). Block-bearing entities and constraints now codegen at any depth below the root, through block-less grouping sub-models and collections-of-sub-models (`Map { paths: Vec<Path> }` with `Path { poses, pose_pairs, frines }`). `resolve_nested_path` walks the registry to a multi-segment `AccessSegment` path (`EntityLocation::Nested`); self-block / cross / set_block_indices emission wrap the per-entity loop in the nested prefix (`nested_container` / `wrap_in_prefix`); `parent.<coll>` refs resolve against the containing sub-model, `root.<coll>` against the root; passive nested entities get their SelfBlock wired. One-hop emission byte-identical (verified via cargo expand on all model examples + the full arael-sketch-solver lib). Tests: nested_self_block, nested_cross_block; demo: slam2d_multi_demo. Unblocks R6.

- **arael-macros**: R6 block-precision check -- block precision must equal the root's precision, failing with the field name instead of the cryptic `expected f32, found f64` the mismatch produces deep in generated solve code. A registry-based macro-time check was built and reverted 2026-07-08 as half-baked: it only covered the root's DIRECT fields (single sub-models and collections), not entities nested behind a block-less grouping struct, because it did not recurse (and nested models did not compile at the time). Nested-struct support has since landed (06e8f15 + 8ffd8ed), so this is now actionable: redo with transitive coverage. Design that worked for the direct case (recoverable from the reverted diff): store `precision: u8` per struct in the registry `SymLayout` (0 = agnostic/no block, 1 = f32, 2 = f64), set from the struct's block; in `generate_root_methods` unwrap each field's element type (Vec/Deque/Arena/Option/Box), `registry_lookup` it, and return a spanned `syn::Error` when a block-bearing type's precision differs from the root -- returning the error before codegen suppresses the E0308. To extend to nesting, walk the registered `SymLayout.fields` (SymFieldType::Struct/OptionalStruct + collection inner types) recursively with a visited-set cycle guard. `Param` fields stay agnostic (they may differ from the root and are cast at the boundary).

- **arael**: port the COO-free first iteration to the other scalar-CSC backends -- DONE 2026-07-11. The first-call assembly is now one shared function (`assemble_first_csc` in simple_lm), used by SparseFaer, SparseEigen, SparseCholmod and SparseCholmodSupernodal alike: models with a statically-knowable pattern build their CSC and position map from the block structure, everything else falls back to COO. Interleaved at slam-300, first iteration: eigen 431.6 -> 416.2 ms, cholmod 433.4 -> 420.9, cholmod-gpl 111.3 -> 105.2; steady-state ms/iter unchanged on all three, so the tile-expanded pattern's ~1.2% explicit zeros cost the CHOLMOD and Eigen factorizations nothing measurable (the open question when this was deferred). First-assembly means from SLAM_TIMING confirm the mechanism: 12.92 -> 7.69 ms (cholmod-gpl), 13.16 -> 9.72 (eigen). Tests: eigen/cholmod/cholmod-gpl backends each land on the dense optimum (tests/block_assembly.rs, feature-gated -- they had no tests at all before).

- **arael**: threading for the Schur route's S factorization. INVESTIGATED 2026-07-11 (benchmarks/slam sfactor_bench): the earlier premise that faer trails CHOLMOD here was WRONG -- a misread G2O_STATS dump. Re-measured, g2o's CHOLMOD supernodal numeric on the same 1800-parameter reduced system takes 22.5 ms/iter against faer's 21.9: parity. faer's auto threshold already selects supernodal (identical to FORCE_SUPERNODAL; forcing simplicial is 7x slower), and at ~50 GFLOP/s it runs at the machine's dense-kernel rate, so there is no kernel headroom to reclaim single-threaded. A dense LLT on S is slower (37.3 vs 21.9 ms at 300 poses) despite S being 69% dense, because the sparse factor is half the size. What is left: more cores (faer's Par::rayon on the numeric factorization, tied to the with_threads item above) and fewer flops (a structurally different reduction). Not band+border: the wide landmarks make S dense, so there is no band to exploit.

- **arael-faer**: dedicated README.md before the next publish (the crate ships to crates.io as an arael dependency now). Cover: what the crate is (faer extensions staged for upstreaming, usable standalone), the block-CSC storage layout (referent-first: the two partition arrays, blk_col_ptr indexing blk_row_idx/val_ptr, dense column-major tiles), the to_csc/csc_pattern/csc_vals_into trio, and the Schur module's symbolic/numeric/backsub flow with the storage conventions (upper block triangle, diagonal tiles upper-only-within-tile, both coupling orientations). Mirror the crate docs in src/lib.rs per house rule.

- **benchmarks/bal**: decide whether to ship a permanent `arael LM f64/f32 schur` row. Measured 2026-07-11 (`BAL_ARAEL_SOLVER=schur`, hint `eliminate_first(points)` now on both roots): the Schur backend wins 2.1x/2.6x on Ladybug-49 (11.9/8.4 vs 25.3/22.0 ms/iter, memory 70.8 -> 37.6 MB) and 1.8x/2.2x on Ladybug-138, is a wash on Ladybug-372 (330 vs 327 f64; f32 still -14%), and LOSES badly on Ladybug-1723 -- 5306 ms/iter and 1912 MB peak against the full-system route's 3246 ms (1.63x slower, same cost and step count; factorizing S alone costs 4755 ms/iter there, with a 665 MB factor). The cost of the reduced system grows ~cubically with camera count (S numeric: 0.9 / 14.3 / 228.4 / 4754.9 ms across the four datasets), so the crossover sits near n_kept ~ 3000. A permanent row would show the win honestly at 49/138 and the wash at 372, but the default must stay the full-system solve, and the 1723 exploratory row must not use it. `benchmarks/bal/src/bin/schur_stats.rs` reports S's size, density, pair count and factorization split per dataset.

- **arael/arael-faer**: fill-reducing ordering for a LARGE SPARSE Schur complement. Measured 2026-07-11 on BAL Ladybug-1723 (1723 cameras, S is 15507x15507 and only 8.1% dense): faer's AMD leaves a factor of ~83M values -- 69% of a dense triangle -- and takes 4755 ms/iter to factorize it, while Ceres factorizes the same S in ~2.3-2.6 s (its whole linear solve is 2586 ms, from its own stats). The kernels are at parity (slam: faer 21.9 ms vs CHOLMOD 22.5), so the entire gap is fill from a worse ordering. CHOLMOD ships METIS (614 metis symbols in libcholmod.so.5) and can use nested dissection; faer offers AMD, Identity, or a Custom permutation only. Options: compute a nested-dissection ordering ourselves and hand it to faer as `SymmetricOrdering::Custom` (a METIS/ND crate, or a hand-rolled separator scheme over the camera graph), or route a large sparse S through the CHOLMOD backend. Reference points at 1723: dense S costs 28.3 s/solve (Ceres dense_schur -- nobody escapes a dense S), our AMD route 4.8 s, Ceres 2.6 s.

- **arael**: keep the Schur structural analysis across solves. The declined route now costs only 2.5% against SparseFaer at BAL-1723 (83.7 s vs 81.7 s) and the viability gate itself 339 ms on an 82 s solve, so this is no longer urgent -- but a system that re-solves repeatedly on the same structure still redoes the whole analysis every time, because reset() drops it. An opt-in "warm" solver would amortize it to zero.
- **arael-faer**: band solver for the reduced system. The Schur complement of a trajectory is BANDED, and heavily so: at 6000 slam poses the reduced pose system is 36,000 parameters with a half-bandwidth of ~900 (a landmark is seen from a bounded stretch of the trajectory -- the scene caps it at 150 poses), which is only 5% dense. `SchurSymbolic::kept_bandwidth()` already reports it for free. faer factorizes it as a general sparse matrix with AMD, leaving 35.4M values in L; a band factorization confines the fill to the band by construction (L <= n*b = 32.6M here, and O(n*b^2) flops with dense kernels throughout, no supernode discovery and no ordering pass at all). Worth an experiment: add a band Cholesky to arael-faer, and route the reduced system through it when `kept_bandwidth()` says the matrix is band-like. Two payoffs beyond speed: the symbolic factorization disappears (nothing to order or analyse -- which is most of the 1.4 s one-time cost at 6000 poses), and the factor's storage is a dense band, so no index arrays. arael already has a scalar band solver (`Band`, used by benchmarks/loc on block-tridiagonal Hessians) -- the block/variable-width version is the missing piece. Note the reduced system is only banded when the KEPT blocks are ordered along the trajectory, which is their natural order; a general BAL-style camera set has no such structure (Ladybug-1723's S is not banded, which is why it loses).

- **Schur: S is stored twice** -- once as blocks (what `schur_reduce` writes) and once as the scalar CSC values faer factorizes, ~260 MB each at 6000 slam poses. Tried removing the block form: give every S tile a strided view into the CSC values (a tile's rows are contiguous, its columns one block-column of nonzeros apart) so the reduction accumulates straight into the array the factorization reads. It works and is exact, but it is SLOWER, consistently, and was rejected: the GEMM's destination tile stops being contiguous, and the 6 columns of a 6x6 tile land ~14 KB apart. Interleaved A/B at 6000 poses -- f64 2310 -> 2344 ms/iter, f32 1612 -> 1721 (+6.8%); at 300 poses f64 49.2 -> 50.2. Memory did fall (f64 peak 2476 -> 2247 MB) but only for f64; the f32 peak did not move at all, because it is set during the symbolic phase, before S exists. The double storage is the price of handing S to a SCALAR sparse factorization, so the way out is not a cleverer copy -- it is the band/block solver above, which factorizes S in the block form the reduction already produces and needs no scalar CSC at all.
