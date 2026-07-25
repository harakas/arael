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
using namespace arael;

Fit fit;
auto p = fit.poses().push_back();       // stable pointer wrapper
p.set_pos({0, 0, 0});
p.set_ea({0.0, 0.1, 0.2});              // roll, pitch, yaw

auto tie = fit.ties().push();
tie.set_a(fit.poses().ref_at(0));       // typed refs
tie.set_b(fit.poses().ref_at(1));

LmConfig cfg;                            // defaults; fields override
cfg.max_iters = 50;
SolveResult r = fit.solve_dense(cfg);    // result<LmResult, SolveError>
if (r.is_err())
    fprintf(stderr, "%s\n", r.error().message);
else
    use(r->end_cost, fit.poses()[0].pos());
```

## The API surface

- **Math types** (`vect2/3`, `matrix2/3`, `quatern`, f/d suffixes)
  mirror the Rust types: `*` is dot, `%` is cross, euler is x=roll,
  y=pitch, z=yaw with `R = R(z)R(y)R(x)`, quaternions store the
  scalar part first.
- **Params** read/write their value; `set_<p>_optimize(false)` fixes
  one. Rotation params take euler `vect3` (or a quaternion for
  `QuaternionParam`); `TransformParam` exposes translation, rotation,
  and per-half optimize flags.
- **Collections**: `push` (deque: `push_back`/`push_front`) returns an
  element wrapper; `refs::`-backed containers also give `ref_at(i)` /
  `get(ref)`, and an `Arena` `push` returns the ref itself (`remove`
  takes it back). `refs::` element pointers are stable across pushes;
  `std::vec::Vec` ones are not -- re-fetch after a push.
- **Option entities**: `has_x()` / `make_x()` / `clear_x()` and `x()`
  returning `option<XRef>`.
- **result / option** mirror Rust's shapes; reading the wrong side
  prints the failed check and aborts (`arael_assert_true` -- always
  on).
- **Failures**: a solve failure or a caught Rust panic comes back as
  `Err(SolveError{status, message})`; `validate()` returns the
  diagnostic text ("" when clean).
- **Covariance**: `assemble_covariance(CovMode)` at the solution
  returns a `Covariance` view; `cov->marginal(entity)` answers the
  entity's marginal block, typed by size (1 param -> `double`, 2 ->
  `matrix2d`, 3 -> `matrix3d`, larger via a caller buffer). Valid
  until the model is dropped or reassembled.

## Contract

One model, one thread. Wrappers are plain pointers: they die with the
model (or an `Arena::remove`). Whatever validity discipline the Rust
API enforces at compile time is the C++ caller's responsibility here.

A Python interface over the same C ABI is planned; the design notes
live in docs/dev/CXX.md.
