# TODO

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
