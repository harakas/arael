# Changelog

## 0.8.0 - 2026-07-27

### Added

- **C++ and Python interfaces.** `cargo-arael` generates a C ABI, C++ headers and
  Python bindings from a model, with CMake glue. Demos in `cxx-examples/`.
- **Parameter covariance.** Marginal, conditional and cross covariance, and
  standard deviations, at the solution. A single query factors once and solves
  per block; `AllMarginals` runs one bulk selected inverse for every block at
  once, at a cost independent of how many are read. Banded systems take a
  block-tridiagonal path.
- **User-defined components.** A component struct with symbolic fields and
  declared Jacobian caches packages reusable geometry for any model.
- **Cross-crate models.** `export_models!` / `arael_import!` share a model
  definition between crates.
- **Generic models.** One entity, constraint or component definition serves both
  f64 and f32 models instead of duplicated twins.
- **Block-level robust loss.** Geman-McClure and Cauchy applied to a whole
  residual block rather than per element.
- **`LmSession`** keeps the learned structure across solves, for warm re-solves.
- **Narrow-band Cholesky** (`SparseFaer::with_narrow_band`) for systems banded in
  natural order: no fill-reducing ordering and no symbolic phase.
- **`SolverKind`** selects the linear backend at runtime.
- **New parameters:** `TransformParam` (se3 twist), `UnitVecParam`, `AngleParam`.
- **`LmConfig`:** presets and per-field builders, a wall-clock budget, gradient
  and parameter tolerances, `min_diagonal`, `num_threads` (rayon feature), an
  iteration observer, and a solve report with setup timing.
- **`model.validate()`** linter and a finite-difference gradient checker.
- **g2o pose-graph file I/O.**
- **Generational refs.** A `Ref` carries a generation, so a stale handle, or one
  from another collection, fails instead of silently naming the wrong element.
- New examples: `slam_demo_gm`, `plane_slam_demo`, `robust_curve_fitting`,
  `root_fit_demo`; a plane-SLAM benchmark.

### Breaking

- Solve entry points return `Result<LmResult, SolveFailure>`. A system that
  cannot be built or factored, and a bad Hessian diagonal, are errors rather than
  panics or a normal-looking result; the best accepted state rides in
  `SolveFailure::partial`.
- One meaning per sparse-solve name: the free `solve_sparse` is now the faer
  backend rather than the COO baseline, `solve` picks by problem size, and the
  old `Sparse` / `SparseDirect` names are gone. The COO and direct-CSC baselines
  are renamed and deprecated.
- A `Ref` comes from its collection -- `push`/`alloc`, `ref_at`, or iteration --
  never from a bare index. `Arena`'s slot-keyed lookup is gone.
- `simple_lm::BandError` is now `BandOverflow`.

### Performance

- Per-iteration cost is down roughly 5-15% on the sparse and Schur paths.
- Unrolled kernels for the Schur one-column updates, and a cheaper symbolic pass.
- Logging macros no longer de-optimize the hot callers they inline into.
