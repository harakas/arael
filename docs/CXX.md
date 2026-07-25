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

Commit the generated files; `cargo arael check` fails when they are
stale (run it in CI). Rerun `export` after model changes.

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
  bits. `arael/geometry.hpp` carries the pinhole `Camera`.
- **Params** read/write their value; `set_<p>_optimize(false)` fixes
  one. Rotation params take euler `vect3` (or a quaternion for
  `QuaternionParam`); `TransformParam` exposes translation, rotation,
  and per-half optimize flags; `UnitVecParam` exposes `unit` and the
  read-only chart basis `unit_d0`/`unit_d1` (for covariance
  Jacobians). Entity wrappers carry `static constexpr param_count`.
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
  real defaults, edit fields, pass it back whole. Every Rust field is
  there except the lambda driver, observer, and gather_timing (the
  preset supplies those); Rust `Option` fields are `arael::option<F>`
  (`cfg.gradient_tolerance = 1e-8;` or `= {};`), `time_limit` in
  seconds. The shared solver surface lives in `arael/solver.hpp`.
- **result / option** mirror Rust's shapes; reading the wrong side
  prints the failed check and aborts (`arael_assert_true` -- always
  on).
- **Failures**: a solve failure or a caught Rust panic comes back as
  `Err(SolveError{status, message})`; `validate()` returns the
  diagnostic text ("" when clean).
- **Covariance**: `assemble_covariance(CovMode)` at the solution
  returns a `Covariance` view; `cov->marginal(entity)` answers the
  entity's marginal block, typed by size (1 param -> `double`, 2 ->
  `matrix2d`, 3 -> `matrix3d`, larger via a caller buffer);
  `cov->cross(a, b, out, cap)` the row-major pa x pb cross-covariance
  between two entities (returns the row count or a negative code).
  Valid until the model is dropped or reassembled.
- **cost()** on the root evaluates the total cost at the current
  parameter values (no solve).

## Contract

One model, one thread. Wrappers are plain pointers: they die with the
model (or an `Arena::remove`). Whatever validity discipline the Rust
API enforces at compile time is the C++ caller's responsibility here.

Worked examples live in `cxx-examples/`: `slam2d_simple_demo` (2D
SLAM, plotting, covariance ellipses) and `slam_demo_gm` (visual-
inertial SLAM with a robust loss, a graduated ramp, and relative
covariance from `cross()`).

A Python interface over the same C ABI is planned; the design notes
live in docs/dev/CXX.md.
