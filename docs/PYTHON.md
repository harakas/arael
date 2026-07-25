# Python interface for arael models

`cargo arael export` generates a Python package next to the C++ tree:
pure-`ctypes` bindings over the same C ABI, loading the cdylib the
capi crate builds. No compiled extension, no dependency beyond
CPython (3.9+); numpy is accepted wherever a sequence is, never
required.

## Model author

Nothing extra: the same `cargo arael export` that writes `capi/` and
`cxx/` also writes `python/{ns}/` -- the package (named like the C++
namespace), the generated ffi/api module pair per root, and a
vendored `arael/` support subpackage (math value types, `Camera`, the
g2o reader, the solver surface). Commit it; `cargo arael check`
covers it.

## Python consumer

Build the cdylib once (`cargo build --release -p <crate>-capi`), put
the package on the path, import:

```python
import sys; sys.path.insert(0, "path/to/model/python")
from cxx_fit import fit
from cxx_fit.arael import CovMode

f = fit.Fit()
for i in range(6):
    o = f.obs.push()
    o.x = float(i)
    o.y = 2.0 * i + 1.0

cfg = fit.LmConfig.well_conditioned()
cfg.max_iters = 50
r = f.solve_sparse(cfg)       # LmResult; raises AraelError on failure
print(r.status, r.end_cost)
print(f.m, f.c)               # root params read back
```

The cdylib is found via an explicit `load(path)`, `$ARAEL_CAPI`, or
the conventional cargo build locations next to the package. Multiple
roots in one crate are separate modules in one package
(`from cxx_mr import line, decay`).

## The API, briefly

The classes mirror the C++ ones one-to-one (docs/CXX.md is the full
surface reference); the differences are Python idiom:

- **Fields are properties**: `pose.pos = (0.1, -0.1, 0.05)` (any
  sequence in), `pose.pos.x` (an `arael.math` value out). Optimize
  flags: `pose.ea_optimize = False`.
- **Math values** (`vect2/3`, `matrix2/3`, `quatern`, f/d variants)
  live in the vendored `arael.math`: the classes ARE the FFI structs,
  with the same operators as the C++ headers (`*` dot, `%` cross,
  `symmetric_eigen`, euler/quaternion conversions...). `arael.geometry.Camera`
  and `arael.g2o.Dataset2` complete the support library.
- **Collections** speak Python: `len(f.obs)`, `f.obs[3]` (negative
  indices too), `for n in f.items:`, `push`/`pop` families,
  `ref_at`/end refs, `r in view` for containment, `try_get` returns
  `None` for a stale ref, arena `view[ref]` and `view.refs()`.
- **Options**: `info.gps` is the entity or `None`; `make_gps()` /
  `clear_gps()`.
- **Refs** are small typed handles (`raw`, `.valid`, equality,
  hashable); default-constructed = null.
- **Solves**: `solve_dense/solve_sparse/solve_band(kd)` return an
  `LmResult` for every healthy termination and raise
  `AraelError(status, message)` for a solver failure or a caught Rust
  panic. `cfg.observer = fn` gets an `LmIter` per damped attempt
  (return `False` to stop); `cfg.gather_timing = True` fills
  `r.timing`; `cfg.gradient_tolerance = 1e-8` / `= None` for the
  optional fields. `last_report()` / `last_pretty_report()`, `cost()`,
  `validate()` as in C++.
- **Covariance**: `assemble_covariance(mode)` returns a view or
  raises; `marginal`/`conditional` shaped by param count (float,
  matrix2d/3d, row-major tuples), `cross(a, b)` tuples, `std_dev(e)`
  a list.

Contract as in C++: wrappers are raw pointers, validity is yours, one
model one thread. The GIL is released around foreign calls, so solves
do not block other Python threads.

Worked examples live in `cxx-examples/<demo>/python/` -- each demo
directory holds the shared model crate plus its C++ (`cxx/`) and
Python (`python/`) drivers; the m3500 twin matches the Rust and C++
output digit for digit.

The parity suite (`cxx-tests/runner/tests/python.rs`) holds the
Python skin to the same standard as the C++ one: the identical
fixture problem, every value compared exactly against the Rust
mirror. Design notes: docs/dev/PYTHON.md.
