# C++ interface for arael models

`cargo arael export` turns a crate holding an `#[arael(root)]` model
into a C++ interface: build the problem from C++, solve, read the
results back -- with the same math vocabulary as the Rust side.

## Model author

```
cargo install cargo-arael
cd mymodel/          # the crate with the model
cargo arael export
git add capi cxx
```

Requirements on the model crate: structs and fields `pub`, entities
constructible with `Default` (a generated `push()` hands out a zeroed
entity to fill), containers spelled `refs::Vec<..>` /
`std::vec::Vec<..>` / `refs::Deque<..>` / `refs::Arena<..>`.

`export` generates:

| Path | Content |
|---|---|
| `capi/` | a Rust crate with the C ABI over the model (`cdylib` + `staticlib`) |
| `cxx/include/<root>.hpp` | the C++ interface |
| `cxx/include/arael/*.hpp` | vendored math + support headers |
| `cxx/CMakeLists.txt` | build glue |
| `python/<ns>/` | the Python interface (docs/PYTHON.md) |

Commit the generated files; `cargo arael check` fails when they are
stale (run it in CI). Rerun `export` after model changes.

Several roots in one crate work: the one capi crate carries every
root's shim (symbols are root-prefixed), each root gets its own
header, and the namespaces nest -- `mycrate::root_a` /
`mycrate::root_b` -- so one translation unit can use both, even at
different precisions. `--root <Name>` exports a single root with the
flat single-root layout instead.

## C++ consumer

```cmake
add_subdirectory(path/to/mymodel/cxx)
target_link_libraries(app PRIVATE arael::mymodel)
```

The glue runs cargo for the Rust parts, so a Rust toolchain is needed
alongside the C++17 compiler. Without CMake: compile against
`-I cxx/include`, link the staticlib cargo builds from `capi/`, plus
`-lpthread -ldl -lm`.

```cpp
#include <mymodel.hpp>
using namespace mymodel;   // the model AND the re-exported arael math

Fit fit;
auto p = fit.poses().push_back();       // stable pointer wrapper
p.set_pos({0, 0, 0});
p.set_ea({0.0, 0.1, 0.2});              // roll, pitch, yaw

auto tie = fit.ties().push();
tie.set_a(fit.poses().ref_at(0));       // typed refs
tie.set_b(fit.poses().ref_at(1));

LmConfig cfg;                            // the Rust defaults; edit and pass
cfg.max_iters = 50;
SolveResult r = fit.solve_dense(cfg);    // result<LmResult, SolveError>
if (r.is_err())
    fprintf(stderr, "%s\n", r.error().message);
else
    use(r->end_cost, fit.poses()[0].pos());
```

## The API surface

The generated interface lives in a namespace named after the MODEL
CRATE (hyphens as underscores; override with `namespace = "..."`
under `[package.metadata.arael]`) -- the model is the project's, not
arael's. It re-exports the arael value types it uses, so one
`using namespace mymodel;` brings both; the vendored headers
themselves stay in `namespace arael`. Different generated models
coexist in one translation unit and one binary.

Naming: an entity class carries the model type's own name (`Pose` is
a thin stable pointer into its collection); `PoseRef` is the typed
u32 handle -- the C++ spelling of Rust's `Ref<Pose>`. Collection
views are named by their container's nature: `PathPosesDeque`,
`PathLandmarksArena`, `LandmarkFrinesVec`.


- **Math types** (`vect2/3`, `matrix2/3`, `quatern`, f/d suffixes)
  mirror the Rust types: `*` is dot, `%` is cross, euler is x=roll,
  y=pitch, z=yaw with `R = R(z)R(y)R(x)`, quaternions store the
  scalar part first. `matrix2/3::symmetric_eigen()` matches the Rust
  one (ascending eigenvalues, eigenvector columns) to precision, not
  bits. `arael/geometry.hpp` carries the pinhole camera
  (`cameraf` / `camerad`; `Camera` is a legacy alias of `cameraf`);
  `arael/g2o.hpp` the pose-graph file I/O (`Dataset2` for
  VERTEX_SE2/EDGE_SE2, `Dataset3` for VERTEX_SE3:QUAT/EDGE_SE3:QUAT
  with sqrt-information Cholesky blocks; `to_g2o()`/`save(path)`
  write the graph back out, byte-identical to the Rust writer).
- **Params** read/write their value; `set_<p>_optimize(false)` fixes
  one. Rotation params take euler `vect3` (or a quaternion for
  `QuaternionParam`); `TransformParam` exposes translation, rotation,
  and per-half optimize flags; `UnitVecParam` exposes `unit` and the
  read-only chart basis `unit_d0`/`unit_d1` (for covariance
  Jacobians); `AngleParam` exposes `<f>_angle` (read/write + optimize)
  and the read-only `<f>_rotation_matrix`. User `#[arael(component)]` structs surface like nested
  sub-models: their fields (set-before / read-after values included)
  behind an accessor. Entity wrappers carry
  `static constexpr param_count`.
- **Collections**: `push` (deque: `push_back`/`push_front`) returns an
  element wrapper; `refs::`-backed containers also give `ref_at(i)` /
  `get(ref)` / `contains(ref)` / `try_get(ref)` (an `option`, empty
  for a stale ref where `get` would abort), and an `Arena` `push`
  returns the ref itself (`remove` takes it back). A
  default-constructed ref is the null sentinel, same as Rust
  `Ref::default()`; `ref.valid()` tests it. Vec gives
  `first_ref()`/`last_ref()` and deque `front_ref()`/`back_ref()`
  (null when empty). Every view has `size()`, `empty()`, and
  `reserve(n)`; vec and deque add `front()`/`back()` (empty container
  is UB, like the STL). Removal mirrors Rust: vec
  `pop`/`truncate`/`clear`, deque
  `pop_back`/`pop_front`/`truncate`/`clear`, arena
  `remove`/`clear` (pops report whether anything was dropped; the
  value itself is dropped in place). All views carry canonical
  BIDIRECTIONAL iterators (`for (auto e : view)`, `++`/`--` pre and
  post, `==`/`!=`, `->`, `rbegin()`/`rend()`, the std iterator
  typedefs) -- the arena walks its live slots in both directions and
  its iterator also exposes `.ref()`. Dereference yields a value wrapper (like
  `vector<bool>`). Standard C++ contract: modifying a container while
  iterating it is undefined behavior. `refs::` element pointers are stable across
  pushes; `std::vec::Vec` ones are not -- re-fetch after a push.
- **Option entities**: `has_x()` / `make_x()` / `clear_x()` and `x()`
  returning `option<X>`.
- **LmConfig** is a plain struct holding the chosen preset's
  Rust values (fetched through the FFI at construction) -- inspect the
  real defaults, edit fields, pass it back whole. Presets:
  `defaults()`, `conservative()`, `well_conditioned()`,
  `ill_conditioned()` (the Nielsen lambda driver). Every Rust field is
  there except the lambda driver (the preset supplies it); Rust
  `Option` fields are `arael::option<F>`
  (`cfg.gradient_tolerance = 1e-8;` or `= {};`), `time_limit` in
  seconds. The shared solver surface lives in `arael/solver.hpp`.
  Warm restart: `cfg.initial_lambda = r.final_lambda` re-enters at
  the previous damping (Rust's `continue_from`); the optimized
  parameters already live in the model.
  `set_log_level(LogLevel::Warn)` quiets arael's diagnostics
  process-wide (Info, everything, is the default).
- **Observer**: `cfg.observer` (+ `cfg.observer_user`, passed back as
  its first argument) is called once per damped attempt with an
  `LmIter` -- iteration counts, accepted flag, costs, lambda, and the
  current parameter vector. Return false to stop the solve (status
  `ObserverTerminated`, current best state kept). A capture-less
  lambda converts directly.
- **Timing**: set `cfg.gather_timing`; the result then carries
  `timing` (per-phase wall-clock seconds plus call counts,
  `has_timing` flags validity) and `r.steps()` -- the per-attempt
  timeline as a vector of `LmStep` (attempt and retry indices,
  accepted, lambda, costs, step and gradient norms, per-phase
  seconds; a damping retry is its own record).
- **Sparse backend options**: `solve_sparse(cfg, opts)` takes a
  `SparseOptions` -- Schur policy (Auto / Force / Never, with the
  Auto pricing's tuning), elimination ordering (Auto / Amd /
  MarginalizeFirst / Natural / NestedDissection), the envelope route
  for the reduced system (Auto / Always / Never, plus panel width),
  supernodal on/off, narrow band, and the block supernodal Cholesky
  (`BlockSupernodalMode` Auto / Always / Never, its batching ratio and
  its memory-lean amalgamation). Constructed with the actual Rust
  defaults; the one-argument `solve_sparse(cfg)` uses them.
- **LmSession**: warm reuse over repeated sparse solves --
  `LmSession sess;` (optionally over a `SparseOptions`), then
  `sess.solve(model, cfg)` keeps the sparsity analysis (pattern,
  ordering, symbolic factorization, Schur plan) across solves, so
  only the first pays for it. Warm solves are bit-identical to cold
  ones. A parameter-count change re-analyzes by itself; call
  `invalidate()` after a structural change at the same count.
- **cost_table()** -- only when the root is `#[arael(root, jacobian)]`:
  per-constraint cost breakdown, label -> that group's robustified
  cost, sorted by label and summing to `cost()`. Labels come from
  `name = "..."` on the constraint attribute, else the struct name.
- **calc_jacobian()** -- same gate: an owned snapshot of the sparse
  Jacobian for DOF/rank analysis. `num_residuals()` / `num_params()`,
  `singular_values()` (descending, always f64; pass `true` to
  column-normalise first, so near-zero values count the free DOF
  without per-parameter scale leaking in), `column_l2_norms()`.
  Copies share it, the last copy releases it; later solves or edits
  do not touch it.
- **Reports and the plan**: the returned `LmResult` owns the full
  Rust-side result. `r.report()` / `r.pretty_report()` render it --
  status, costs, iterations, the timing breakdown, the backend's
  plan -- and stay valid however many solves follow. `r.plan()`
  returns the sparse backend's `SchurPlan` as data (reduction taken
  or not, eliminated/kept parameters, ordering, bandwidth, whether the
  envelope or the block supernodal route factorized, the Auto policy's
  evidence); empty for dense and band solves. Copies of a result share ownership.
- **result / option** mirror Rust's shapes; reading the wrong side
  prints the failed check and aborts (`arael_assert_true` -- always
  on).
- **Status helpers**: `is_success(r->status)` / `as_str(r->status)`
  mirror the Rust helpers -- note success is NOT `code >= 0` (hitting
  max_iters or the time limit is Ok-side but not a success).
- **Failures**: a solve failure comes back as
  `Err(SolveError{status, message, partial, failure})` -- `partial`
  holds the best accepted state when the solve got past its first
  assembly, usable for diagnosis (its report renders like any
  result); `failure` is the structured cause (`SolveFailureKind` plus
  the indices a caller can act on: the offending parameter for
  `UnconstrainedParameter` / `DegenerateDiagonal` with its
  `DiagonalFault`, row/col/kd for `BandOverflow`, and so on; -1
  where not applicable). A caught
  Rust panic (a stale ref, an unguarded Option read, a bad options
  tag) THROWS `arael::PanicError` with the panic text; the model's
  parameters are unchanged and a session in use was invalidated.
  `validate()` returns the diagnostic text ("" when clean).
- **Covariance**: `assemble_covariance(CovMode)` at the solution
  returns a `Covariance` view; `cov->marginal(entity)` answers the
  entity's marginal block, typed by size (1 param -> `double`, 2 ->
  `matrix2d`, 3 -> `matrix3d`, larger via a caller buffer);
  `cov->cross(a, b, out, cap)` the row-major pa x pb cross-covariance
  between two entities (returns the row count or a negative code);
  `cov->std_dev(entity, out, cap)` the per-parameter standard
  deviations (works on every CovMode, including TriDiagonal);
  `cov->conditional(entity, out, cap)` the conditional covariance
  (all other parameters held fixed).
  The view OWNS its assembly: copies share it, the last copy releases
  it, and every `assemble_covariance` call is independent -- older
  views keep answering from their own assembly. Entity arguments must
  come from the live model; `last_error()` carries the failure text.
- **Solves**: `solve_dense` / `solve_sparse` / `solve_band(kd)` --
  band Cholesky for banded Hessians, `kd` the half-bandwidth in
  scalar parameters. `cost()` on the root evaluates the total cost at
  the current parameter values (no solve).

## Contract

One model, one thread. Wrappers are plain pointers: they die with the
model (or an `Arena::remove`). Whatever validity discipline the Rust
API enforces at compile time is the C++ caller's responsibility here.

Worked examples live in `cxx-examples/`: `slam2d_simple_demo` (2D
SLAM, plotting, covariance ellipses), `slam_demo_gm` (visual-inertial
SLAM with a robust loss, a graduated ramp, and relative covariance
from `cross()`), `loc_demo` (localization against a known map on the
band solver, TriDiagonal `std_dev`), and `m3500_demo` (pose graph
from a g2o file, digit-for-digit with the Rust example).

The Python interface over the same C ABI is docs/PYTHON.md -- the
same export writes it, the same parity suite holds it.
