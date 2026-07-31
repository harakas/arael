# CXX / Python export -- gap review and plan

A diff of the generated C++ and Python interfaces (cargo-arael) against
the public Rust API, 2026-07-31. What the skins already cover is in
docs/CXX.md and docs/PYTHON.md; this file lists what they lack, in
recommended execution order. Effort: S = hours, M = about a day,
L = multi-day. Design notes are sketches to refine per item, not
commitments.

## Where the skins stand

The model surface is complete: every field kind, container, ref,
option, component and rotation param crosses the FFI; multi-root
works; both skins are pinned by exact-parity tests. The solver
surface is the thin part. Today it is: `solve_dense` / `solve_sparse`
/ `solve_band(kd)`, `LmConfig` (every field except the lambda
driver), a plain-old-data `LmResult`, model-held `last_report()` /
`last_pretty_report()`, the observer callback, timing totals, and the
five covariance queries.

## 1. Documentation drift -- statements that are false today [S] [DONE 2026-07-31]

Since 62bf03d the Python element wrappers re-resolve their pointer by
key on every access and cannot dangle; three places still document the
old raw-pointer contract:

- `docs/PYTHON.md:80` -- "wrappers are raw pointers, validity is yours"
- `docs/dev/PYTHON.md:21` -- same sentence
- `emit_py.rs:129` -- generated view docstring says "wrappers stay
  valid per the C++ contract"

Also wrong or missing in `docs/PYTHON.md`:

- `ref_at` / `get` / `try_get` / `in` are listed as general collection
  features; they exist only on refs-flavoured collections
  (`refs::Vec` / `Deque` / `Arena`), not `std::vec::Vec`.
- Undocumented: `free()`, `last_error()`, `param_count`,
  `LmIter.param()/param_list()` and the `lambda_` spelling,
  `reserve()/truncate()/clear()`, the presets
  (`defaults()/conservative()/well_conditioned()`),
  `assemble_covariance` defaulting to `ALL_MARGINALS`, `arena.refs()`.

Two generator touch-ups ride along: the Python emitter drops `opaque`
fields silently where C++ leaves a marker comment (match C++), and the
generated module does not re-export `LmTiming` (only reachable as
`{ns}.arael.LmTiming`).

## 2. Map the ill_conditioned preset [S] [DONE 2026-07-31]

`preset_config` maps 0/1/2 and everything else falls to defaults, so
`LmConfig::ill_conditioned()` -- the only way to get the Nielsen
lambda driver, since the driver field is deliberately not exposed --
is unreachable from both skins. Add preset 3 to the shim,
`solver.hpp`, and `solver.py`. Note in passing: preset 0 (Defaults)
and 1 (Conservative) produce identical configs because Rust's
`Default` is `conservative()`; document, do not remove.

## 3. Sparse backend options -- the solver is locked to defaults [M] [DONE 2026-07-31]

`{root}_solve_sparse` runs `model.solve_sparse(&cfg)`, a
default-constructed `SparseFaer`. Neither skin can set:

- ordering (`FaerOrdering`: Auto / Amd / MarginalizeFirst / Natural /
  NestedDissection)
- Schur policy (`SchurPolicy`: Auto / Force / Never)
- envelope mode (`EnvelopeMode`: Auto / Always / Never), panel width
- narrow band (whole-system envelope route)
- supernodal on/off

Prerequisite, a Rust API gap in its own right: `SparseFaerOptions`
(the plain-data form `SolverKind::Sparse` carries) has no
`envelope_mode` / `envelope_panel_width` -- those exist only as
`SparseFaer` builder methods, and `from_options` leaves them at
default. Extend the options struct first, then mirror it.

Sketch: a repr(C) `CSparseOptions` (enum tags as u32, bools) with a
filler `{root}_sparse_options(out)` writing the defaults, and
`{root}_solve_sparse(h, cfg, opts, out)` taking a nullable options
pointer (null = defaults; header and shim regenerate as a unit, so
the signature change is safe). C++: `solve_sparse(cfg)` and
`solve_sparse(cfg, opts)`; Python: `solve_sparse(cfg=None,
opts=None)`. Excluded from v1: the marginalize range list (the model
author's `marginalize` attribute already covers it),
iterative/implicit Schur (`CgOptions` -- experimental), threads
(already `LmConfig.num_threads`). Tests: parity fixtures forcing each
knob and comparing plan and results against the same options driven
from Rust.

As built: as sketched, plus the Auto policy's tuning (`flop_margin`,
`obvious_flop_ratio`) and the envelope panel width in the struct.
Rust prerequisite landed first: `SparseFaerOptions` gained
`envelope` / `envelope_panel_width` (+ builders), `from_options`
applies them (tests/narrow_band_cholesky.rs pins it). FFI:
`CSparseOptions` with u32 enum tags, `{root}_sparse_options` fills
the Rust defaults, `{root}_solve_sparse` takes a nullable options
pointer; an out-of-range tag panics and surfaces as PanicError /
AraelError with the tag in the message (final policy 2026-07-31: the
shim never aborts; the typed wrappers and validating Python setters
make the case unreachable anyway). Parity pins the defaults field-for-field and two forced
routes (Force+Natural+Always takes the envelope, Force+Amd+Never
declines it) against Rust driving `SparseFaer::from_options` with
the same options.

## 4. LmResult is a scalar snapshot -- no report, plan, or partial [M] [DONE 2026-07-31]

`LmResult` in both skins carries costs, iterations, status, lambda
and timing totals. Everything else about a solve is only reachable as
text through the model-held `last_report()`, which the next solve
overwrites. Missing:

- `report()` / `pretty_report()` on the result object itself.
- The solver plan (`SchurPlan`: reduced or whole, eliminated/kept
  params, ordering, bandwidth, envelope taken or not, fill and flop
  evidence) as data. "Did it use the envelope?" is currently answered
  by reading prose.
- The partial result on failure: the shim formats
  `SolveFailureKind` with `{:?}` and discards `SolveFailure::partial`
  entirely. C++ `SolveError` is status + text; Python `AraelError`
  likewise.

Sketch: `CLmResult` keeps its POD fields and gains an owned opaque
`detail` pointer (a boxed Rust `LmResult`). Per-root FFI:
`{root}_result_report(detail, pretty)`, `{root}_result_plan(detail,
out CSchurPlan) -> bool`, `{root}_result_free(detail)`. C++
`LmResult` becomes a move-only owning wrapper with `report()`,
`pretty_report()`, `plan()`; Python adds the methods plus `__del__`.
The failure path boxes the `SolveFailure` instead: `Display` text
(not `{:?}` of the kind), and the partial result reachable from
`SolveError` / `AraelError`. `last_report()` stays as-is.
Items 3 and 4 are the two the interface review was opened for.

As built: as sketched, except model-level `last_report()` /
`last_pretty_report()` were REMOVED (user decision -- reports live on
the result only). `CLmResult.detail` boxes the Rust LmResult; the C++
LmResult shares it via shared_ptr so it stays copyable inside
`result<>`/`option<>`; SolveError moved from solver.hpp into the
generated header (it now holds `option<LmResult> partial`). Parity
pins: plan fields exact vs Rust on the sparse solve, dense carries no
plan, report non-empty and unchanged by a following solve, default
result renders "", failure partial-presence flag. Not exercised
end-to-end: a failure that carries a partial result (the degenerate
fixture fails on its first assembly, so `partial` is None on every
skin; the shim branch that boxes a partial has no test driving it).

## 5. LmSession warm reuse [M] [DONE 2026-07-31]

Already in TODO.md: the generated solves are stateless, so a
graduated ramp (cxx-examples/slam_demo_gm) re-analyzes sparsity every
pass where the Rust twin keeps one `LmSession` warm. Sketch: a
handle-owned session -- `{root}_session_begin(h, opts)` routes
subsequent `solve_sparse` calls through it,
`{root}_session_invalidate(h)`, `{root}_session_end(h)`. Parity: warm
solves bit-identical to cold, as tests/lm_session.rs already pins in
Rust. The one real performance gap in the export.

As built: as an explicit session OBJECT, not handle-owned state --
mirroring Rust's `LmSession` beats rerouting `solve_sparse` behind
the model's back. Per-root `{root}Session` behind
`{root}_session_new(opts)` (null = defaults, null return on invalid
options) / `_solve(s, h, cfg, out)` / `_invalidate` / `_free`; C++
`LmSession` (move-only, befriended by the root class) and Python
`LmSession` with GC-driven free. Sparse backend only. Parity pins:
warm == cold end cost exactly, invalidate agrees, a pushed entity
(parameter-count change) re-analyzes by itself, and an options-built
session takes the envelope. Both slam_demo_gm drivers run their ramp
through one session; the pretty report shows analysis paid on pass 1
only.

## 6. Per-constraint cost breakdown [M, needs a decision] [DONE 2026-07-31]

`calc_cost_table` (per-label sum of squared residuals) is a debugging
aid worth having in every skin, but it lives on `JacobianModel`,
which only exists for `#[arael(root, jacobian)]` models -- exporting
it as-is would make the cost table conditional on an opt-in most
models do not set. Options: emit it only when the sidecar says the
root has `jacobian`, or add a lighter per-label cost pass to the
macro that does not need the jacobian machinery. Decide when picked
up.

As built (user decision: gate on `jacobian`): the sidecar gained a
`jacobian` flag; the emitters gate `{root}_cost_table` /
`_cost_table_name` / `_cost_table_value` and the C++/Python
`cost_table()` on it. The review also surfaced and fixed a REAL BUG:
the row-derived `calc_cost_table` ignored robust losses (the gm
frine label read 1.57M raw against a robustified cost of 9.4k), so
the macro now generates a direct table pass -- each constraint's
cost blob shadowed into its label's slot, `rho(s)` applied per block
-- and the table sums to `calc_cost` (tests/jacobian.rs pins it).
`calc_jacobian` rows/entries are likewise weighted by
`sqrt(rho'(s))` so `J^T J` and `2 J^T r` reproduce the assembled
Gauss-Newton system (pinned on a lossy model). slam_demo_gm names its
constraints (gps, tilt, drift, frine, odometry) and the Python
driver prints the table after the ramp.

## 7. Covariance for param-bearing components [DROPPED 2026-07-31]

The covariance query emitters filter on `role == "entity" &&
param_count > 0`, so a user component with parameters (a
`UnitVecParam`-style plane lives happily inside an entity) cannot be
queried, although Rust's `marginal_cov<M: Model>` accepts any Model.

Dropped as fringe: the owning entity's marginal covers the practical
cases, and the rare sub-block need has the Rust API. Revisit only if
a real consumer asks.

## 8. Smaller items, in no particular order

- `continue_from`: document the recipe (`cfg.initial_lambda =
  r.final_lambda`) in both skin docs; not worth FFI surface. [S]
  [DONE 2026-07-31]
- Param vector snapshot/restore (`serialize`/`deserialize`) for
  retry-with-perturbation loops; add when a consumer asks. [S]
  [deferred 2026-07-31, user decision]
- Log control (`arael::log::set_level` / `silence`) over the FFI, to
  quiet backend warnings from a C++/Python host. [S]
  [DONE 2026-07-31: `{root}_set_log_level` + `set_log_level(LogLevel)`
  in both skins; level only, the sink callback stays Rust-side]
- g2o 3D (`Pose3` / `DeltaPose3` / `Dataset3`) in the vendored C++
  and Python support libraries; SE2 only today. Needed the day a 3D
  pose-graph demo shows up. [M]
  [DONE 2026-07-31: both libraries mirror src/g2o.rs's SE3:QUAT
  subset incl. quaternion normalization and the sqrt-information
  Cholesky blocks; pinned by the cxx_math / py_math golden parity]

## Second review, 2026-07-31

A fresh diff after items 1-8 landed. Two defects first, then gaps by
priority.

### 9. Skin defects [S]

- C++ `LmSession` stores `session_new`'s return unchecked; invalid
  options return null and the next `solve()` dereferences it. Python
  raises. Validate in the ctor (or give the session a bool).
  [DONE 2026-07-31, user decision: an out-of-range options tag now
  ABORTS loudly (panic in the shim) everywhere -- session_new never
  fails, the null path is gone -- and the Python enum setters
  validate, raising ValueError on a value outside the enum (parity
  pins it). Reaching the abort now requires poking raw struct bytes.]
- `cost_table()` on a panic: C++ returns an empty vector (ambiguous
  -- empty is a legitimate table), Python raises. Align on the
  C++ side distinguishing the failure.
  [DONE 2026-07-31, after discussion: a Rust panic is a throw, so
  both skins raise -- Python AraelError (status -2, as before), C++
  a new `arael::PanicError` thrown from the solve wrappers,
  cost_table and assemble_covariance. The catch_unwind contract was
  audited and documented: stored parameters are unchanged (solves
  are a serialize/optimize/deserialize round trip), evaluation
  scratch is rewritten per pass, and the one risky spot -- a
  session's half-built cache -- is now invalidated on a caught
  panic. Since extended: the covariance query wrappers throw too
  (a ck_ helper), option-tag panics are caught like every other
  panic (session_new defers a construction panic to the first solve,
  message intact), and cost() -- the last uncaught entry point --
  catches, returning NaN with the text in last_error. The shim never
  aborts.]

### 10. Status helpers [S]

`LmStatus::is_success` is not exported and the obvious substitute
(code >= 0) is WRONG: `MaxIterations`, `LambdaCeiling`, `TimeLimit`,
`RetryBudgetExhausted` are all Ok-side non-successes. Every consumer
re-derives this and most will get it wrong. Ship `is_success()` +
`as_str()` as inline helpers in both skins.

### 11. Jacobian / conditioning surface [M]

Only `calc_cost_table` crosses; `calc_jacobian` and its diagnostics
(`singular_values`, `column_l2_norms`, `svd`, `num_residuals`) do
not -- the DOF/rank analysis the `jacobian` opt-in exists for.
Minimum viable: `{root}_singular_values(out, cap)` +
`_column_l2_norms` + `_num_residuals`.

### 12. Gradient check with a chosen tolerance [S-M]

`check_gradients_tol` has no FFI entry; `validate()` runs the check
only at the default tolerance and always pays the full assembly. Add
`{root}_check_gradients(h, tol)` (tol <= 0 = default) -- the knob an
f32-storage model needs.

### 13. Covariance lifetime and joint queries [S-M]

The assembly is a handle slot with no release: an AllMarginals
selected inverse stays resident forever, and a second
`assemble_covariance` silently re-points every outstanding
`Covariance` view. Add `{root}_release_covariance` and document the
aliasing [S]. Joint blocks over a collection/root (any `Model` in
Rust) stay per-entity over the FFI [M, on demand].

### 14. Session for band solves [M]

The exported session is sparse-only; a banded chain solved with
`solve_band` re-analyzes every pass. A kd-carrying session variant
is the cheap version.

### 15. g2o write-back [S]

The vendored readers stop at `parse`/`load`; `save`/`to_g2o` have no
counterpart, so a host can load a pose graph but not write the
optimized one back out.

### 16. Diagnostics detail [S-M each, on demand]

- `LmTiming::steps` / `LmStep` per-iteration timeline (observer
  covers half the fields, no per-phase durations, and render never
  prints the steps table).
- `Style` granularity: unicode-without-colour is unreachable (one
  extra bool on `result_report`).
- `SolveFailureKind`'s param index reaches the skins only as prose;
  the number is what a caller acts on. Decide explicitly.
- `cost()`/`cost_table()` at an explicit params vector -- blocked on
  the deferred serialize/deserialize item; pick up together.

### 17. Math/vocabulary parity [S]

- Python lags C++: `is_finite` (only matrix3), `similar` (absent),
  `quatern.cast` (absent) -- and the golden parity never calls them
  (a test blind spot).
- `matrix2/3::similar` and `matrix3::null_space` missing from BOTH
  skins.
- `se3` twist type absent from both (nothing blocked; conversion
  convenience only).
- C++ views have `front()/back()/empty()`; Python has none.
- Result accessor idioms differ (Python property + None, C++ method
  + option) -- pick one idiom per concept and document it.

### Aging exclusion [DONE 2026-07-31]

`SchurSolve::Iterative`/`IterativeImplicit` + `CgOptions` is now the
only `SparseFaerOptions` field with no crossing and has picked up
benchmark use; the "experimental" label is aging.

Shipped: `SparseOptions` gained `schur_solve` (Factorize / Iterative
/ IterativeImplicit) plus `cg_tol` / `cg_max_iters` /
`cg_restart_every` (the `CgOptions` fields, defaults from Rust), in
both skins with a `SchurSolve` enum. Parity pins a Force+Iterative
solve exactly, including the plan's CG iteration total. Every
`SparseFaerOptions` capability except the marginalize range list now
crosses.

## Not planned (unchanged decisions)

- Raw gradient/Hessian assembly surface -- recorded in TODO.md;
  revisit only if a real consumer needs it.
- Custom lambda drivers beyond the presets; observer-shaped callback
  if ever needed.
- Eigen/CHOLMOD backend dispatch -- needs a features pass-through in
  `[package.metadata.arael]`; niche, faer + band cover the demos.
- `ExtendedModel` -- never exposed.
- Bulk numpy bridges -- out of scope until profiled (docs/dev/PYTHON.md).
- Structured `CovError` variants / `CovAssembly::dim()` /
  structured `validate()` issues -- the Display text is the contract.
