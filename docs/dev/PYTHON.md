# Python interface for arael models -- the plan

Stage 6 of docs/dev/CXX.md, expanded. `cargo arael export` gains a
Python backend: pure-`ctypes` bindings over the same C ABI the C++
skin uses, loading the cdylib the capi crate already builds. No
compiled extension, no dependency beyond CPython -- the settled
decision from CXX.md 7b (stdlib-only core; numpy accepted wherever a
sequence is, never required).

## Principles

- **Same C ABI, third consumer.** The shim crate is untouched; the
  Python emitter reads the same sidecar IR the C++ emitter reads.
  Every symbol, layout, and status code is already pinned by the C++
  parity suite -- Python only has to mirror them.
- **Mirror the C++ classes one-to-one, idiomatic where Python is.**
  Same names (`Fit`, `Pose`, `PoseRef`, `Covariance`), same methods --
  but fields are properties, collections speak `len`/`[]`/iteration,
  absent options are `None`, and solve failures raise. No invented
  vocabulary.
- **Relaxed contract, same as C++.** Wrappers hold raw pointers;
  validity discipline is the caller's. One model, one thread.
- **Self-contained output.** The generated package vendors its own
  copy of the support library (math, camera, g2o, solver mirrors),
  like the C++ tree vendors `arael/*.hpp`. `import` works with one
  `sys.path` entry and a built cdylib; nothing to install.

## Generated layout

```
model/
  capi/                  # unchanged -- cdylib + staticlib
  cxx/                   # unchanged
  python/
    {ns}/                # package named like the C++ namespace
      __init__.py        # re-exports the root module(s), load()
      arael/             # vendored support (source of truth:
        __init__.py      #   cargo-arael/python/arael/)
        math.py          # vect2/3, matrix2/3, quatern
        geometry.py      # Camera
        g2o.py           # Dataset2
        solver.py        # LmConfig/LmResult/LmIter/LmTiming layouts,
                         # enums, AraelError
      _{root_sn}_ffi.py  # ctypes signatures for one root (generated)
      {root_sn}.py       # the API classes for one root (generated)
```

Multi-root crates get one `_ffi`/api module pair per root in the same
package -- the Python twin of the nested C++ namespaces:
`from cxx_mr import line, decay`.

## Loading the cdylib

The capi crate already builds `lib{crate}_capi.so` (`.dylib`/`.dll`).
Resolution order in the generated `load()`:

1. an explicit path argument: `slam2d_simple.load("path/to/lib...so")`;
2. the `ARAEL_CAPI` environment variable;
3. the conventional build locations relative to the package
   (`../capi/target/release/`, `../../target/release/` for workspace
   members), release then debug.

First model construction calls `load()` implicitly; a clear
`AraelError` names the searched paths when nothing is found. No
building from Python -- the README says `cargo build --release -p
{crate}-capi` (or the cmake build both skins share).

## The vendored support library

`ctypes.Structure` does double duty: the math classes ARE the FFI
structs (field order = repr(C) layout), so values cross the boundary
without conversion.

```python
class vect3d(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]
    # +, -, unary -, * (dot/scale), % (cross), norm(), unit(),
    # sequence protocol (len/iter/index), constructor from any
    # 3-sequence
```

Ported surface = what the C++ headers carry (the Python demo
drivers need the same pieces the C++ ones did): vect2/3, matrix2/3 (rows, from_*
constructors, rotation_from_euler_angles, get_euler_angles, transpose,
operators, symmetric_eigen, cast), quatern (from/get euler,
rotation_matrix, from_two_vectors, slerp...), `Camera`, `g2o.Dataset2`.
f32 variants are distinct classes (`vect3f`) -- the ABI needs the
distinction; constructors accept plain sequences everywhere.

Parity: a `tests/py_math.rs` twin of `tests/cxx_math.rs` -- the same
golden values, the Python script prints `name %.17e` lines, compared
at the same tolerances. Skipped when `python3` is absent.

## The generated API, worked example

The C++ worked example from CXX.md 6, in Python:

```python
import sys; sys.path.insert(0, "model/python")
from cxx_fit import fit
from cxx_fit.arael import vect3d, CovMode

f = fit.Fit()
for i in range(6):
    o = f.obs.push()
    o.x = float(i)
    o.y = 2.0 * i + 1.0 + (0.05 if i % 2 == 0 else -0.05)

cfg = fit.LmConfig.well_conditioned()
cfg.max_iters = 50
cfg.verbose = True
r = f.solve_sparse(cfg)          # LmResult; raises AraelError on
print(r.status, r.end_cost)      # solve failure or panic
print(f.m, f.c)                  # root params read back

pose = f.poses.push_back()
pose.pos = (0.1, -0.1, 0.05)     # any 3-sequence in
print(pose.pos.x)                # vect3d out

tie = f.ties.push()
tie.a = f.poses.ref_at(0)        # typed refs
tie.b = f.poses.ref_at(1)

for n in f.items:                # every view iterates
    print(n.v)
mark = f.marks.push()            # arena: ref back
f.marks[mark].t = 0.4            # __getitem__ by ref (arena) /
                                 # index (vec, deque)
cov = f.assemble_covariance(CovMode.ALL_MARGINALS)
print(cov.marginal(f.items[0]))  # 1x1 -> float
print(f.last_report())
```

## Field-kind mapping (the emitter's table)

| Sidecar kind | Python surface |
|---|---|
| data / param scalar | property, `float`/`int`/`bool` |
| data / param math | property, vect/matrix/quatern class; any sequence in |
| param optimize flag | `p.<f>_optimize` property |
| euler_param (simple/universal) | vect3 property + optimize |
| euler_param (rotvec) | quatern property + optimize |
| TransformParam | `<f>_translation` / `<f>_rotation` props + per-half optimize |
| UnitVecParam | `<f>_unit` property + read-only `<f>_unit_d0/d1` |
| AngleParam | `<f>_angle` prop + optimize; read-only `<f>_rotation_matrix` |
| user component / struct | sub-object property (fresh thin wrapper) |
| optional | `make_<f>()` / `del`-style `clear_<f>()`; `<f>` returns the wrapper or `None` |
| ref | typed `Ref` dataclass (raw u32, `valid`, `==`, hashable) |
| collection vec/deque/arena | view object: `len`, `[]`, iteration, push/pop family, `ref_at`/end refs, `contains`, `try_get` (returns `None`), `reserve`, arena `remove` |
| skip / opaque | absent |

Collection iteration wraps the same index/cursor C calls the C++
iterators use; mutating while iterating is undefined, same contract.
`__getitem__` on vec/deque takes an index or a typed ref; on arena a
ref only.

## Solver surface

- **LmConfig**: `ctypes.Structure` mirror of `CLmConfig` (layout
  pinned by the same defaults-parity test the C++ side has), filled by
  the preset FFI: `LmConfig.defaults() / .conservative() /
  .well_conditioned()`. The `COpt` fields hide behind properties:
  `cfg.gradient_tolerance = 1e-8` / `= None` / reads back `None`.
- **Observer**: `cfg.observer = fn` wraps `fn(it)` in a
  `CFUNCTYPE(c_bool, c_void_p, POINTER(LmIter))`; the live parameter
  vector reads via `it.param(i)` / `it.param_list()`. Return `False`
  (or nothing -> `True`) to stop/continue. The wrapper object owns the CFUNCTYPE
  reference for the config's lifetime (the classic ctypes GC trap,
  handled once in the generator). GIL: released around the solve,
  re-acquired for each callback -- ctypes does both automatically.
- **Result**: `solve_dense/solve_sparse/solve_band(kd)` return
  `LmResult` (status enum, costs, iterations, `timing` or `None`);
  status codes < 0 (solver failure, panic) raise
  `AraelError(status, last_error_text)` -- the Err side of the C++
  `result`, as an exception. `cfg.gather_timing = True` fills
  `r.timing`.
- **Reports**: `m.last_report()` / `m.last_pretty_report()` return
  `str`. `m.cost()`, `m.validate()` (returns "" when clean) as in C++.

## Covariance

`m.assemble_covariance(mode)` returns a `Covariance` view or raises.
Queries mirror C++ but return Python-shaped values: `marginal(e)` a
float / matrix2d / matrix3d / row-major tuple-of-tuples by param
count, `cross(a, b)`, `conditional(e)` likewise, `std_dev(e)` a list.
Failed queries raise `AraelError` with the last_error text.

## Testing

Same battery shape as the C++ skin, sharing the fixtures:

1. **py_math parity** (`tests/py_math.rs`): vendored support library
   vs Rust, exact-tolerance framework of cxx_math.
2. **Python parity** (`cxx-tests/runner/tests/python.rs`): a
   `parity_main.py` building the SAME fixture problem as
   `parity_main.cpp` -- same `name %.17e` protocol, same exact
   comparison against the Rust mirror (floats formatted with
   `%.17e` round-trip exactly). Covers the same feature list:
   containers, refs, options, compound params, solves, observer,
   timing, report, covariance queries, band, degenerate failure ->
   exception. Skips when `python3` is missing.
3. **Multi-root**: `multiroot_main.py` twin importing both modules
   from one interpreter.
4. **Golden files**: the emitted `fit.py`/`_fit_ffi.py` pinned
   byte-exact in cargo-arael/tests/golden, same regeneration recipe.
5. **CI**: the existing cxx-tests step picks all of this up
   (python3 is present on the runner).

## The Python demo drivers

Four examples, same functionality and reports as their Rust and C++
twins. Each demo directory owns ONE model and per-language drivers
(layout reshaped 2026-07-25 from a separate python-examples/ tree):

```
cxx-examples/<demo>/
  model/       # shared crate + generated capi/ cxx/ python/
  cxx/         # main.cpp + CMakeLists.txt
  python/      # main.py
  README.md    # one demo, both drivers
```

Each `main.py` inserts the sibling `../model/python` on `sys.path`
and states the one build command it needs. Content:

- **slam2d_simple_demo**: world gen with `random.Random`, compose,
  verbose + gather_timing solve, `last_pretty_report()`, pose and
  landmark errors, covariance ellipses (matrix2 symmetric_eigen), the
  same hand-rolled EPS plot (stdlib file I/O -- no matplotlib).
- **slam_demo_gm**: the graduated ramp, re-anchoring, relative
  covariance from `cross()`, LM error/sigma report.
- **loc_demo**: band solve (`solve_band(11)`), TriDiagonal +
  `std_dev` last-pose sigmas.
- **m3500_demo**: `arael.g2o.Dataset2` load, solve_sparse, metrics,
  EPS -- deterministic, so it must match the Rust and C++ output
  digit for digit (the strongest cross-skin check of the three).

RNG note: `random.Random(seed)` differs from StdRng and mt19937 --
same shape, same behavior, numbers differ (as C++ already does).

## Staged implementation

All stages landed 2026-07-25; the notes below are the plan they
followed. Not carried over: the C++ demos' EPS outputs keep their
names, the Python ones write `*_py.eps`. The f32 tolerance in
py_math is 1e-5 (Python computes in doubles and rounds through the
storage type; the C++ twin computes in f32 proper).

- **P0 -- vendored support library + math parity.** `cargo-arael/
  python/arael/` (math, geometry, g2o, solver), embedded via
  `include_str!` like the headers; `tests/py_math.rs`. No emitter yet.
- **P1 -- emitter core.** `emit_py.rs`: ffi signature module + API
  module for the fit fixture -- root lifecycle, scalar/math
  properties, vec push/at, solve_dense/sparse, exceptions. Python
  parity test harness (subset of the C++ one) + goldens.
- **P2 -- full container and field surface.** deque/arena, refs and
  end-refs, contains/try_get, pops, iteration, options, components,
  rotation params, TransformParam/UnitVecParam. Parity catches up to
  parity_main.cpp feature for feature.
- **P3 -- solver extras.** Band, observer, timing, report, cost,
  validate, config option-properties. Parity: observer count/stop,
  timing counts, report transitions.
- **P4 -- covariance.** All five queries + shapes. Parity vs Rust
  exact.
- **P5 -- multi-root + the demo drivers.** mr twin test; the four
  examples; m3500 digit-parity against the C++ output; README and
  docs/PYTHON.md (user-facing twin of docs/CXX.md); regenerate all
  committed trees (they gain `python/`).

Each stage regenerates cxx-tests + the demo trees and keeps every
suite green, as usual.

## Open questions

1. **Python floor**: 3.9 (matches current-LTS distros) unless told
   otherwise -- nothing in the design needs newer.
2. **Emit always vs opt-in**: emit `python/` unconditionally
   (symmetric with cxx/, it is only text) -- or a
   `[package.metadata.arael] skins = [...]` toggle if tree size
   bothers. Default: always.
3. **Bulk numpy bridges** (zero-copy param/collection views): out of
   scope until a real consumer profiles a need, per CXX.md 7b.
