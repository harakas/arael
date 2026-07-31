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

## 3. Sparse backend options -- the solver is locked to defaults [M]

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

## 5. LmSession warm reuse [M]

Already in TODO.md: the generated solves are stateless, so a
graduated ramp (cxx-examples/slam_demo_gm) re-analyzes sparsity every
pass where the Rust twin keeps one `LmSession` warm. Sketch: a
handle-owned session -- `{root}_session_begin(h, opts)` routes
subsequent `solve_sparse` calls through it,
`{root}_session_invalidate(h)`, `{root}_session_end(h)`. Parity: warm
solves bit-identical to cold, as tests/lm_session.rs already pins in
Rust. The one real performance gap in the export.

## 6. Per-constraint cost breakdown [M, needs a decision]

`calc_cost_table` (per-label sum of squared residuals) is a debugging
aid worth having in every skin, but it lives on `JacobianModel`,
which only exists for `#[arael(root, jacobian)]` models -- exporting
it as-is would make the cost table conditional on an opt-in most
models do not set. Options: emit it only when the sidecar says the
root has `jacobian`, or add a lighter per-label cost pass to the
macro that does not need the jacobian machinery. Decide when picked
up.

## 7. Covariance for param-bearing components [S-M]

The covariance query emitters filter on `role == "entity" &&
param_count > 0`, so a user component with parameters (a
`UnitVecParam`-style plane lives happily inside an entity) cannot be
queried, although Rust's `marginal_cov<M: Model>` accepts any Model.
Extend the filter to components with params; same generated shape.

## 8. Smaller items, in no particular order

- `continue_from`: document the recipe (`cfg.initial_lambda =
  r.final_lambda`) in both skin docs; not worth FFI surface. [S]
- Param vector snapshot/restore (`serialize`/`deserialize`) for
  retry-with-perturbation loops; add when a consumer asks. [S]
- Log control (`arael::log::set_level` / `silence`) over the FFI, to
  quiet backend warnings from a C++/Python host. [S]
- g2o 3D (`Pose3` / `DeltaPose3` / `Dataset3`) in the vendored C++
  and Python support libraries; SE2 only today. Needed the day a 3D
  pose-graph demo shows up. [M]

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
