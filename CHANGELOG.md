# Changelog

## Unreleased (0.8.1)

### Added

- **Iterative Schur.** `SparseFaer::with_iterative_schur` solves the reduced
  system by preconditioned conjugate gradients instead of factorizing it, and
  `with_implicit_schur` does the same without ever forming the reduced matrix.
  Both take a `CgOptions` (tolerance, iteration cap, restart period); the
  factorizing route stays the default. `SchurPlan::cg_iterations` reports the
  work done.

### Changed

- **The reduced Schur system is factorized under its own envelope, in block
  form, by default.** Where the reduction leaves a naturally-ordered system --
  banded or dense -- `S` is factored in place in panels sized to its envelope,
  instead of being flattened to a scalar CSC and handed to faer's sparse
  Cholesky. That drops the scalar copy of `S`, its pattern, the symbolic
  analysis and the supernodal scratch. Across a landmark-span sweep (a reduced
  system from 2% to 69% dense) it is faster AND smaller at every point: 1-11%
  less time, 8-39% less peak memory, same optimum.
  `SparseFaer::with_envelope_schur(false)` goes back to faer. A reduction
  ordered by AMD or nested dissection has no envelope to exploit and is
  unaffected, as are the iterative routes. `with_narrow_band` is unchanged and
  remains the separate, opt-in whole-system route.
- **Assembly keeps one scatter target per tile, not per value.** Blocks with a
  static tile shape derive every position from a tile origin and stride stored
  beside their parameter indices, so the per-scalar position map is gone for
  them. It survives for `TripletBlock` and COO-built patterns, which have no
  static shape. Every indexed backend benefits. At Ladybug-372 this is 47 MB
  off the peak of each route with no change in assembly time.
- **The sparse structures are indexed 32 bits wide.** Block rows, column
  pointers, permutations and the offsets the Schur analysis keeps were `usize`
  and are now `arael_faer::SparseIndex`, an alias for `u32`. faer's sparse
  Cholesky is generic over its index type, so the reduced system is analysed
  and factorized at that width too. Worth 14-16% of peak memory on a pose
  graph and 9-10% on bundle adjustment: at 1000 SLAM poses 394 -> 332 MB, at
  Ladybug-1723-clean 557 -> 501 MB iterative and 1240 -> 1110 MB factorizing.
- **Value-buffer offsets are `ValueIndex`, 32 bits wide.** Every map that
  addresses a matrix's values -- the assembly scatter map, `CscMatrix::diag_pos`,
  the block Hessian's damping map, the CG preconditioner's factor offsets, the
  band factor's source map -- was `usize` and is now `arael::ValueIndex`, an
  alias for `u32`. 32 bits addresses 4e9 values (34 GB of `f64`); a problem
  past that needs the alias widened in `arael-faer` and a rebuild, with no
  other edit. `CooMatrix::to_csc_with_map`, `build_scatter_map` and
  `scatter_into`, and `LmProblem::calc_grad_hessian_sparse_indexed` carry the
  new type.
- **`Model::accumulate_hessian_positions64` / `_32` are now
  `bind_hessian_positions64` / `_32`**, taking `&mut self` and a
  `HessianBinder` in place of a position callback. Same for
  `LmProblem::accumulate_hessian_positions` ->
  `LmProblem::bind_hessian_positions`. Generated code moves with the macro;
  hand-written implementations of these methods need updating.

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
