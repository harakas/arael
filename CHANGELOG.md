# Changelog

Released versions only; entries are written from the commit log when a release
is cut.

## 0.8.1 - 2026-08-01

### Added

- **Iterative Schur.** `SchurSolve::Iterative` solves the reduced system by
  preconditioned conjugate gradients instead of factorizing it;
  `SchurSolve::IterativeImplicit` never forms it, applying the operator block by
  block. On Ladybug-1723-clean the iteration is 5.7x and 6.6x cheaper than the
  factorized route, the implicit one on under half the memory.
- **`EnvelopeMode`** prices the envelope factorization of the reduced system
  against the ordered sparse factor it would replace, per solve. `SchurPlan`
  reports which route ran.
- **`camera<T>`**, with `cameraf` and `camerad`. `Camera` remains as an alias.
- **C++ and Python interfaces** gained the per-constraint cost table, Jacobian
  diagnostics (singular values, column norms), covariance views that own their
  assembly, `LmSession` warm reuse, the sparse backend options including
  iterative Schur, the per-attempt timeline, structured `SolveFailure` as data,
  log-level control, SE3 g2o reading, and g2o write-back. A caught Rust panic
  raises in both skins instead of aborting the process.

### Breaking

- **`Model` is generic over the solve width.** Each 32/64 method pair collapses
  into one method: `serialize_params`, `deserialize_params`, `update_params`,
  `advance_params` and the `accumulate_hessian*` family take `<F: Float>`;
  `collect_hessian_cells` and `bind_hessian_positions` lose their suffixes.
  `ParamType` follows, with `write_to<F>` / `read_from<F>`.
- **`ExtendedModel` is `ExtendedModel<F: Float>`**, implemented at the root's
  precision, and its methods drop the 32/64 suffixes.
- **The generated `serialize64` / `serialize32` / `deserialize64` /
  `deserialize32` are gone.** Call `RootProblem::serialize` / `deserialize`,
  which is in the prelude; off-width serialization is
  `Model::serialize_params::<F>`.
- `SchurPlan::narrow_band` is renamed `SchurPlan::envelope`: both band routes set
  it, and the reduced system's envelope is often not narrow.
  `SparseFaer::with_narrow_band` keeps its name.

### Fixed

- Robust losses reached neither `calc_cost_table` nor `calc_jacobian`: the table
  summed raw squared residuals, and the Jacobian rows broke the `J^T J` /
  `2 J^T r` match with the assembled system on lossy models.
- `SchurPolicy::Auto` could take a reduction that keeps the system nearly whole.
  Its cheap flop ratio prices the reduced route against a floor no whole-system
  factorization can beat, which is a bound rather than a cost.
- A Python element wrapper cached a pointer into its collection, so growing that
  collection left it reading the old buffer. Wrappers now re-resolve on access.
- A constraint sweep over an `Option` frine field emitted `for x in &field`,
  which raises `for_loops_over_fallibles` in the model crate's own build.

### Performance

- Peak memory is down 6-35% across the pose-graph and bundle-adjustment suites:
  the assembly scatter map rides in the blocks, and value-buffer offsets and the
  sparse structures index 32 bits wide.

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
