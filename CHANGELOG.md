# Changelog

Released versions only; entries are written from the commit log when a release
is cut.

## 0.8.2 - 2026-08-06

### Added

- **Block supernodal Cholesky**, and it is the default.
  `BlockSupernodalMode { Auto, Always, Never }` with `with_block_supernodal` on
  the solver and the options. `Auto` factorizes in block form wherever the
  scalar route would -- the whole Hessian, and a reduced Schur system the
  envelope declined. Models without block structure keep the scalar route, and
  the envelope and iterative routes keep precedence in every mode.
- **The supernodal factorization takes threads.** Dense kernels run behind a
  work threshold, and independent subtrees are chunked across workers -- each
  owns whole subtrees and runs their panels sequentially, then the top is
  factored with the threaded kernels.
- **`CovOptions`** carries the two decisions a covariance assembly makes:
  `ordering` (`CovOrdering`) and `block_supernodal`. `assemble_covariance_with`
  takes it; `assemble_covariance` keeps its signature and the defaults. Neither
  changes the covariance, only what producing it costs.
- **`CovAssembly::plan`** reports what an assembly picked -- the ordering it
  kept, the flops of each candidate `CovOrdering::Auto` priced, and how many
  symbolic analyses it built -- the way `SchurPlan` reports a solve's route.
  `took_block_route` says whether the block route ran.
- **The whole-Hessian supernodal route picks its elimination ordering.** A
  marginalize set named by `with_marginalize` or the model's `marginalize(..)`
  attribute is eliminated first; otherwise, under `FaerOrdering::Auto`, a
  detected set is priced against block-AMD by building both symbolics and
  taking the fewer flops.
- **C++ and Python** gained the block supernodal mode with its batching and
  amalgamation knobs, `SchurPlan`'s block flag, and the covariance ordering and
  factorization options.

### Breaking

- **`arael_faer::supernodal_factorize` takes a `faer::Par`.** It hands it to
  the update GEMMs, the panel triangular solve and the diagonal Cholesky, each
  behind its own work threshold.
- **The default factorization route changed**, so results can differ from 0.8.1
  in the last ulp. Bit-identity holds per route: pinned to `Never` or `Always`,
  one and four threads match exactly.

### Fixed

- `SchurPolicy::Auto` priced the reduction against the whole system under AMD
  alone, so a trajectory that revisits -- a loop closure, a figure-8 crossing --
  was compared against the worse of the two orderings and the reduction won a
  comparison it should have lost. It now analyses both, prices against the
  cheaper, and hands the winning ordering to the route that runs.
- A declined reduction (`SchurPolicy::Auto` weighing the reduction and saying
  no) fell through to the scalar route, bypassing `BlockSupernodalMode::Auto`.
- `EnvelopeMode::Auto` priced the envelope against faer's scalar symbolic,
  which stopped being the competitor once a declined envelope went to the block
  supernodal. It builds a `SupernodalSymbolic` in the same natural order and
  compares against that, handing the symbolic on rather than letting the route
  rebuild it.
- `nd::order_graph` recursed once per cut with no balance guarantee and
  overflowed the stack on a 47.8k-block bundle-adjustment problem.
- A covariance assembly built symbolic analyses it discarded: `block_perm`
  returns a permutation for the scalar factorization, which runs its own
  symbolic analysis over it, so a named ordering built a full
  `SupernodalSymbolic` and dropped it unread and `CovOrdering::Auto` built two.
  Named orderings now take the order alone.

### Performance

- The supernodal default measured at or ahead of the scalar route on every
  benchmark, with a 2-10x cheaper symbolic phase and 17-35% less peak memory.
  Pose-graph iterations are 5.5-19% cheaper on it, their peak memory 8-27%.
- Threading reaches the supernodal factorization: at four threads, figure-8
  landmark SLAM is 2.2x at 1200 poses and 2.5x at 4800, bundle adjustment 1.5x.
  A pose graph is unchanged -- its panels sit below the work threshold.
- Pricing the whole route under nested dissection turns a revisiting trajectory
  around: the 1200-pose figure-8 iterates 4.0x cheaper on half the peak memory.
- A covariance assembly is about twice as cheap to build on a bundle problem,
  and a revisiting trajectory recovers a pose marginal 2.2x cheaper and every
  marginal 3.2x cheaper by ordering the factor by dissection.

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
