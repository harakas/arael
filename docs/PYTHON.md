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
  `symmetric_eigen`, euler/quaternion conversions...).
  The `arael.geometry` pinhole camera (`cameraf` / `camerad`;
  `Camera` is a legacy alias of `cameraf`) and the `arael.g2o`
  pose-graph readers
  (`Dataset2` for SE2, `Dataset3` for SE3:QUAT with sqrt-information
  Cholesky blocks) complete the support library.
- **Collections** speak Python: `len(f.obs)`, `f.obs[3]` (negative
  indices too), `for n in f.items:`, the `push`/`pop` families,
  `reserve`/`clear`/`truncate`. Ref lookups -- `ref_at`/end refs,
  `r in view`, `try_get` returning `None` for a stale ref, arena
  `view[ref]` and `view.refs()` -- exist on the refs-flavoured
  containers (`refs::Vec`/`Deque`/`Arena`); a plain `std::vec::Vec`
  field has index access and iteration only.
- **Options**: `info.gps` is the entity or `None`; `make_gps()` /
  `clear_gps()`.
- **Refs** are small typed handles (`raw`, `.valid`, equality,
  hashable); default-constructed = null.
- **Solves**: `solve_dense/solve_sparse/solve_band(kd)` return an
  `LmResult` for every healthy termination and raise
  `AraelError(status, message)` for a solver failure or a caught Rust
  panic. `solve_sparse(cfg, opts)` takes a `SparseOptions` (filled
  with the actual Rust defaults at construction): Schur policy,
  elimination ordering, the envelope route for the reduced system,
  supernodal, narrow band -- the `SchurPolicy` / `FaerOrdering` /
  `EnvelopeMode` enums live beside it. `LmSession()` (optionally over
  a `SparseOptions`) keeps the sparsity analysis warm across repeated
  `sess.solve(model, cfg)` calls -- bit-identical to cold solves;
  `invalidate()` after a structural change at the same parameter
  count. `LmConfig` starts from a preset -- `defaults()`,
  `conservative()`, `well_conditioned()`, `ill_conditioned()` -- with
  the actual Rust values filled in. `cfg.observer = fn` gets an
  `LmIter` per damped attempt (`it.lambda_`, `it.param(i)`,
  `it.param_list()`; return `False` to stop); `cfg.gather_timing =
  True` fills `r.timing` and `r.steps` (the per-attempt timeline, a
  list of `LmStep` records); `cfg.gradient_tolerance = 1e-8` / `=
  None` for the optional fields. The result owns the full Rust-side solve:
  `r.report()` / `r.pretty_report()` render it (status, costs, the
  timing breakdown, the backend's plan) and stay valid however many
  solves follow; `r.plan` is the sparse backend's `SchurPlan` as data
  (`None` for dense and band solves). A failed solve raises
  `AraelError(status, message)` whose `.partial` holds the best
  accepted state when the solve got that far and whose `.failure`
  carries the structured cause (a `SolveFailure`: `SolveFailureKind`
  plus the parameter / row / block indices, `DiagonalFault` for a
  degenerate diagonal; -1 where not applicable). `cost()`, `validate()`,
  `last_error()` as in C++; a model frees on garbage collection
  (`free()` to force it). Warm restart: `cfg.initial_lambda =
  r.final_lambda` re-enters at the previous damping; the optimized
  parameters already live in the model. `set_log_level(LogLevel.WARN)`
  quiets arael's diagnostics process-wide (INFO, everything, is the
  default). When the root is `#[arael(root, jacobian)]`,
  `cost_table()` returns the per-constraint cost breakdown as a dict
  (label -> that group's robustified cost, summing to `cost()`), and
  `calc_jacobian()` an owned snapshot for DOF/rank analysis:
  `num_residuals` / `num_params` properties,
  `singular_values(column_normalised=False)` and `column_l2_norms()`
  as lists (freed on garbage collection, `free()` to force it).
- **Covariance**: `assemble_covariance(mode)` returns a view or
  raises; `marginal`/`conditional` shaped by param count (float,
  matrix2d/3d, row-major tuples), `cross(a, b)` tuples, `std_dev(e)`
  a list. The view owns its assembly (freed on garbage collection,
  `free()` to force it); reassembling never disturbs older views.
  Entity arguments must come from the live model.

Idiom notes: where C++ has `front()`/`back()`/`empty()` on views,
Python spells them `view[0]` / `view[-1]` / `len(view)`; where C++
returns `option<T>` from a method (`r.plan()`), Python uses a
property returning the value or `None` (`r.plan`).
`r.status.is_success()` and `r.status.as_str()` mirror the Rust
helpers -- note success is NOT `status >= 0` (hitting max_iters or
the time limit is Ok-side but not a success).

One model, one thread. Unlike C++, element wrappers are not raw
pointers: they re-resolve by index or ref on every access, so growing
a collection cannot leave a held wrapper dangling (a removed arena
slot still fails loudly through the generation check). The GIL is
released around foreign calls, so solves do not block other Python
threads.

Worked examples live in `cxx-examples/<demo>/python/` -- each demo
directory holds the shared model crate plus its C++ (`cxx/`) and
Python (`python/`) drivers; the m3500 twin matches the Rust and C++
output digit for digit.

The parity suite (`cxx-tests/runner/tests/python.rs`) holds the
Python skin to the same standard as the C++ one: the identical
fixture problem, every value compared exactly against the Rust
mirror. Design notes: docs/dev/PYTHON.md.
