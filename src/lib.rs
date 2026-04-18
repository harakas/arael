//! **ARAEL** -- Algorithms for Robust Autonomy, Estimation, and Localization.
//!
//! Nonlinear optimization framework with compile-time symbolic differentiation.
//!
//! Define model structs with optimizable parameters, write constraints as
//! symbolic expressions, and the framework symbolically differentiates at
//! compile time, applies common subexpression elimination, and generates
//! compiled cost, gradient, and Gauss-Newton hessian (J^T J approximation)
//! code.
//!
//! # Features
//!
//! - **Symbolic math** -- expression trees with automatic differentiation,
//!   simplification, LaTeX/Rust code generation (via `arael-sym`)
//! - **Compile-time constraint code generation** -- write constraints
//!   symbolically, get compiled derivative code with CSE
//! - **Levenberg-Marquardt solver** -- with robust error suppression
//!   via the [Starship method (US12346118)](https://patents.google.com/patent/US12346118)
//!   `gamma * atan(r / gamma)` and switchable
//!   constraints (`guard = expr`)
//! - **Multiple solver backends** via `LmSolver` trait:
//!   - Dense Cholesky (nalgebra) -- fixed-size dispatch up to 9x9
//!   - Band Cholesky -- pure Rust O(n*kd^2) for block-tridiagonal systems
//!   - Sparse Cholesky (faer, pure Rust) -- for general sparse hessians
//!   - Eigen SimplicialLLT and CHOLMOD (SuiteSparse) -- optional C++ backends via FFI
//!   - LAPACK band -- optional dpbsv/spbsv backend
//! - **Indexed sparse assembly** -- precomputed position lists for
//!   zero-overhead hessian assembly after first iteration
//! - **f32 and f64 precision** -- `#[arael(root)]` for f64,
//!   `#[arael(root, f32)]` for f32 throughout
//! - **Model trait** -- hierarchical serialize/deserialize/update protocol
//! - **Type-safe references** -- `Ref<T>`, `Vec<T>`, `Deque<T>`, `Arena<T>`
//! - **Runtime differentiation** -- parse equations from strings at runtime,
//!   auto-differentiate symbolically, and optimize via `ExtendedModel` +
//!   `TripletBlock` (see `examples/runtime_fit_demo.rs`)
//! - **Hessian blocks** -- `SelfBlock<A>` and `CrossBlock<A, B>` for 1- and
//!   2-entity constraints (packed dense); `TripletBlock` for 3+ entities (COO sparse)
//! - **Jacobian computation** -- `#[arael(root, jacobian)]` generates
//!   `calc_jacobian()` returning a sparse [`Jacobian<T>`](model::Jacobian)
//!   matrix for DOF analysis via SVD.
//!   [`#[arael(constraint_index)]`](model::JacobianRow) tracks constraint
//!   provenance per row. See `examples/jacobian_demo.rs`.
//! - **Gimbal-lock-free rotations** -- `EulerAngleParam` optimizes a small
//!   delta around a reference rotation matrix
//! - **WASM/browser support** -- compiles to WebAssembly; the `arael-sketch`
//!   constraint editor runs in the browser via eframe/egui
//!
//! # Scope
//!
//! Arael is a **nonlinear optimization framework**, not a complete SLAM or
//! state estimation system. The SLAM and localization demos show how to use
//! arael as the optimizer backend, but a production SLAM pipeline would
//! additionally need:
//!
//! - **Front-end perception**: feature detection, descriptor extraction
//! - **Data association**: matching observed features to existing landmarks,
//!   handling ambiguous or incorrect matches
//! - **Landmark management**: initializing new landmarks from observations,
//!   merging duplicates, pruning unreliable ones
//! - **Keyframe selection**: deciding when to add new poses vs. discard
//!   redundant frames
//! - **Loop closure**: detecting revisited places, verifying loop closure
//!   candidates, and injecting constraints
//! - **Outlier rejection logic**: deciding which observations to reject
//! - **Marginalization / sliding window**: limiting optimization scope for
//!   real-time operation, marginalizing old poses while preserving their
//!   information
//! - **Map management**: spatial indexing, map saving/loading, multi-session
//!   map merging
//!
//! Arael provides the compile-time-differentiated solver that sits at the
//! core of such a system. Everything above is application-level logic that
//! builds on top of it.
//!
//! # Example: Symbolic Math
//!
//! ```ignore
//! use arael::sym::*;
//!
//! arael::sym! {
//!     let x = symbol("x");
//!     let f = sin(x) * x + 1.0;
//!
//!     println!("f(x)   = {}", f);           // sin(x) * x + 1
//!     println!("f'(x)  = {}", f.diff("x")); // x * cos(x) + sin(x)
//!
//!     let vars = std::collections::HashMap::from([("x", 2.0)]);
//!     println!("f(2.0) = {}", f.eval(&vars).unwrap()); // 2.8185...
//! }
//! ```
//!
//! The `sym!` macro auto-inserts `.clone()` on variable reuse, so you write
//! natural math without ownership boilerplate.
//! See [docs/SYM.md](https://github.com/harakas/arael/blob/master/docs/SYM.md)
//! for the full symbolic math reference.
//!
//! # Example: Robust Linear Regression
//!
//! Define a model with optimizable parameters and a residual expression.
//! The Starship method `gamma * atan(plain_r / gamma)` suppresses outlier
//! influence while preserving smooth differentiability:
//!
//! ```ignore
//! #[arael::model]
//! struct DataEntry { x: f32, y: f32 }
//!
//! #[arael::model]
//! #[arael(fit(data, |e| {
//!     let plain_r = (a * e.x + b - e.y) / sigma;
//!     gamma * atan(plain_r / gamma)
//! }))]
//! struct LinearModel {
//!     a: Param<f32>,
//!     b: Param<f32>,
//!     data: Vec<DataEntry>,
//!     sigma: f32,
//!     gamma: f32,
//! }
//! ```
//!
//! The macro generates `calc_cost()`, `calc_grad_hessian()`, and `fit()`
//! with symbolically differentiated, CSE-optimized compiled code.
//! The robust fit ignores outliers while tracking the inlier data:
//!
//! ![Linear Regression](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/linear_regression.png)
//!
//! See [examples/linear_demo.rs](https://github.com/harakas/arael/blob/master/examples/linear_demo.rs) for the full source.
//!
//! # Runtime Differentiation
//!
//! Compile-time differentiation generates optimized Rust code with CSE at
//! build time -- ideal when the model structure is fixed. But many
//! applications need equations that are only known at runtime: user-typed
//! formulas in a CAD parametric dimension, configuration-driven curve
//! fitting, or symbolic constraints loaded from a file.
//!
//! Arael supports this through **runtime differentiation**: parse an
//! equation string with `arael_sym::parse`, symbolically differentiate
//! once at setup with `E::diff`, then evaluate the expression tree
//! numerically each solver iteration. The
//! [`ExtendedModel`](model::ExtendedModel) trait and
//! [`TripletBlock`](model::TripletBlock) provide the integration point
//! with the LM solver.
//!
//! The sketch editor (`arael-sketch`) uses this extensively for parametric
//! expression dimensions -- a user can type `d0 * 2 + 3` as a dimension
//! value, and the solver constrains the geometry to satisfy the equation
//! in real time, with full symbolic derivatives.
//!
//! The model uses `#[arael(root, extended)]` and implements
//! [`ExtendedModel`](model::ExtendedModel) to push residuals and
//! derivatives into a [`TripletBlock`](model::TripletBlock) at each
//! solver iteration:
//!
//! ```ignore
//! #[arael::model]
//! #[arael(root, extended)]
//! struct RegressionModel {
//!     coeffs: refs::Vec<Coefficient>,         // optimizable parameters
//!     hb: TripletBlock<f64>,                  // Gauss-Newton accumulator
//!     residual_expr: Option<arael_sym::E>,    // parsed equation
//!     derivs: Vec<(String, u32, arael_sym::E)>, // pre-computed derivatives
//!     // ...
//! }
//!
//! impl ExtendedModel for RegressionModel {
//!     fn extended_compute64(&mut self, params: &[f64], grad: &mut [f64]) {
//!         for &(x, y) in &self.data {
//!             vars.insert("x", x);
//!             vars.insert("y", y);
//!             let r = residual.eval(&vars).unwrap();
//!             let dr: Vec<f64> = self.derivs.iter()
//!                 .map(|(_, _, d)| d.eval(&vars).unwrap()).collect();
//!             // writes 2*r*dr into grad AND pushes full upper-triangle
//!             // Hessian into the TripletBlock (one call, both done)
//!             self.hb.add_residual(r, &indices, &dr, grad);
//!         }
//!     }
//! }
//! ```
//!
//! See
//! [examples/runtime_fit_demo.rs](https://github.com/harakas/arael/blob/master/examples/runtime_fit_demo.rs)
//! for a complete example that accepts an arbitrary equation from the
//! command line and fits it to data with robust error suppression.
//!
//! # Instrumentation & Debugging
//!
//! When a model fails to converge or the solution is wrong, the usual
//! chain of inspection is: look at the *cost distribution*, check the
//! *gradient and Hessian* for bad values, then look at the *rank of the
//! Jacobian*. Each step corresponds to a specific arael API.
//!
//! Enable instrumentation by adding the `jacobian` flag on the root:
//!
//! ```ignore
//! #[arael::model]
//! #[arael(root, jacobian)]
//! struct MyModel { /* ... */ }
//! ```
//!
//! This generates an impl of [`model::JacobianModel`] with two methods:
//! `calc_jacobian` and `calc_cost_table`.
//!
//! ## My solve doesn't converge. What do I check?
//!
//! 0. **Turn on solver verbose mode first.** Set `verbose: true` on
//!    `LmConfig` and every LM step prints cost, lambda, and the step
//!    outcome. On a Cholesky rejection the line also reports
//!    non-finite counts for grad / diagonal / cur_x / matrix and a
//!    count of non-positive diagonal entries -- four quick signals
//!    that narrow the problem before any deeper digging:
//!
//!    ```ignore
//!    let cfg = arael::simple_lm::LmConfig::<f32> { verbose: true, ..Default::default() };
//!    let result = arael::simple_lm::solve_sparse_faer_f32(&x0, &mut model, &cfg);
//!    ```
//!
//!    A healthy pass looks like steady cost drops with rising /
//!    stabilising step sizes and no Cholesky rejections -- see
//!    `examples/slam_demo.rs` run for a reference trace. If verbose
//!    already reports NaN / Inf or diag ≤ 0, skip to steps 2 / 3
//!    below; otherwise continue to the cost-by-label breakdown.
//!
//! 1. **Cost breakdown by label.** Name your constraint attributes with
//!    `#[arael(constraint(hb, name = "drift", { ... }))]` so each group
//!    shows up under its own label in the sum-of-squares. Call
//!    `model.calc_cost_table(&params)` to get `HashMap<&'static str, T>`:
//!
//!    ```ignore
//!    use arael::model::JacobianModel;
//!    let table = model.calc_cost_table(&params);
//!    for (label, cost) in &table { println!("{:<20} {:.6}", label, cost); }
//!    ```
//!
//!    A single label dominating the total is usually the culprit --
//!    either an overly tight sigma, bad initial values for its inputs,
//!    or a constraint that's mathematically unsatisfiable.
//!
//! 2. **NaN or Inf residuals / derivatives.** The verbose-mode output
//!    from step 0 already tells you whether grad / matrix / params
//!    contain non-finite values at the failing step. If they do, walk
//!    the Jacobian to find the specific row:
//!
//!    ```ignore
//!    let j = model.calc_jacobian(&params);
//!    for row in &j.rows {
//!        if !row.residual.is_finite()
//!            || !row.entries.iter().all(|(_, v)| v.is_finite())
//!        {
//!            eprintln!("bad row cid={} label={}", row.constraint, row.label);
//!        }
//!    }
//!    ```
//!
//!    A NaN residual or partial derivative usually means a `sqrt`,
//!    `acos`, `asin`, or `atan2` saw a degenerate input (zero-length
//!    vector, both-zero arguments, `|x| > 1` for asin/acos).
//!    `arael_sym` ships `safe_sqrt`, `safe_asin`, `safe_acos`,
//!    `safe_atan2` that clamp / regularise at the singular point and
//!    produce non-diverging derivatives. Before reaching for them,
//!    though, **prefer to redesign the constraint so the singularity
//!    can't be hit**. A `safe_*` wrapper hides the degeneracy from
//!    the solver and may leave the residual insensitive to the
//!    parameters that should drive it out of the singular region;
//!    an equivalent constraint formulated on the right geometric
//!    quantity avoids the singularity entirely. E.g. match 3D
//!    landmarks to features in 3D space (compare world-frame
//!    directions or positions) instead of projecting through a
//!    camera model and computing 2D image-plane residuals -- the
//!    3D formulation is simpler, better conditioned, and has no
//!    pixel-wraparound / behind-camera pathology.
//!
//! 3. **Non-positive diagonal.** The verbose-mode `diag<=0: N`
//!    counter at a Cholesky rejection is the loudest possible signal
//!    that some parameter is untouched by every constraint (indices
//!    left at `u32::MAX`) or is receiving a negative contribution.
//!    Either outcome is a bug distinct from f32 accumulation noise.
//!
//! 4. **Gradient magnitude.** After
//!    [`simple_lm::LmProblem::calc_grad_hessian_dense`], the maximum
//!    absolute gradient component should be small relative
//!    to the cost scale at a local minimum. A huge gradient with tiny
//!    cost means the parameter scaling is off -- one parameter moves
//!    cost several orders of magnitude more than another, which
//!    destabilises Levenberg-Marquardt.
//!
//! 5. **Hessian health.** The same `hessian` array should be finite and
//!    positive-semi-definite at a minimum (smallest eigenvalue ≥ 0
//!    modulo roundoff). A significantly-negative smallest eigenvalue
//!    means the Gauss-Newton approximation J^T J is a poor local fit
//!    -- often because constraints are ill-conditioned or cancel.
//!
//! 6. **Stiffness.** Ratios between the smallest and largest sigmas
//!    (or equivalently, smallest and largest eigenvalues of J^T J)
//!    that span many orders of magnitude make the problem numerically
//!    stiff. LM damping has to pick a lambda that suits both ends,
//!    which is hard at f32 precision. Keep isigmas comparable where
//!    you can; if a tight constraint dominates one direction, a gauge
//!    direction orthogonal to it will starve for signal. Starting
//!    with a loose scale and ramping up (graduated optimisation --
//!    see `examples/loc_demo.rs` and `slam_demo.rs` for the
//!    `frine_isigma_scale` pattern) helps LM climb a stiff problem
//!    without rejecting early steps.
//!
//! 7. **Simpler math beats clever math.** Reformulate residuals on the
//!    most natural geometric quantity. 3D direction / position errors
//!    are cheaper and better-conditioned than 2D reprojection errors;
//!    relative rotations compared as matrices or unit quaternions
//!    avoid Euler-angle gimbal lock; distances compared in squared
//!    form avoid `sqrt` derivatives near zero. Every nonlinear
//!    operation you remove is one less place for the residual /
//!    derivative to misbehave and one less source of stiffness.
//!
//! 8. **Inspect the generated code.** Use [`cargo expand`](https://github.com/dtolnay/cargo-expand)
//!    to see what the macro emitted for your constraint. See the
//!    next section for a walkthrough.
//!
//! 9. **Rank / DOF.** Call [`model::Jacobian::singular_values`] (or the
//!    full [`Jacobian::svd`](model::Jacobian::svd) for directions):
//!
//!    ```ignore
//!    let j = model.calc_jacobian(&params);
//!    let svs = j.singular_values();
//!    println!("{:?}", svs); // descending; near-zero entries count free DOF
//!    ```
//!
//!    Near-zero singular values count the degrees of freedom. If this
//!    is higher than you expect, the model is under-constrained. The
//!    right singular vectors (columns of [`SvdResult::v`](model::SvdResult))
//!    corresponding to σ ≈ 0 name the unconstrained parameter
//!    directions -- useful for identifying *which* parameters are free.
//!    SVD is always performed in f64 regardless of the model's element
//!    type, so rank detection stays reliable even for f32 models.
//!
//! ## How do I know my new constraint is actually doing anything?
//!
//! Name the attribute, run the solve before and after adding it, and
//! compare `calc_cost_table` entries and row counts. If the new label
//! appears with a non-trivial cost contribution, it's participating.
//! If its row count is zero, a `guard` is excluding it. See
//! [examples/jacobian_demo.rs](https://github.com/harakas/arael/blob/master/examples/jacobian_demo.rs)
//! for an end-to-end walkthrough.
//!
//! ## Looking under the hood with `cargo expand`
//!
//! Mastering arael means being able to read what the macros actually
//! generated for your equations. `#[arael::model]` does a lot: it
//! interprets the constraint body symbolically, differentiates it
//! against every reachable parameter, runs common-subexpression
//! elimination, and emits Rust code for three call paths
//! (`__compute_blocks`, `__set_block_indices`, `calc_jacobian`).
//! [`cargo expand`](https://github.com/dtolnay/cargo-expand)
//! (`cargo install cargo-expand`) prints the expansion exactly as
//! the compiler sees it.
//!
//! ```bash
//! cargo expand --example single_root_demo
//! # or, for your own crate:
//! cargo expand --lib              # library
//! cargo expand --bin my_bin       # binary
//! cargo expand my_mod::MyModel    # a specific path
//! ```
//!
//! ### Example: a one-line fix constraint
//!
//! The single-root demo declares
//!
//! ```ignore
//! #[arael(constraint(hb, name = "fix_x", {
//!     [(singleroot.x - 3.0) * singleroot.isigma]
//! }))]
//! struct SingleRoot {
//!     x: Param<f64>,
//!     y: Param<f64>,
//!     isigma: f64,
//!     /* ... */
//! }
//! ```
//!
//! `cargo expand --example single_root_demo` shows the macro emits a
//! `__compute_blocks` method with a block like:
//!
//! ```ignore
//! /// arael: SingleRoot[fix_x] @ examples/single_root_demo.rs:28
//! let __r_0       = singleroot.isigma * (singleroot.x.work() - 3.0);
//! let __dr_0_0    = singleroot.isigma;      // d/d x
//! let __dr_0_1    = 0.0;                    // d/d y
//! __item.hb.add_residual(
//!     __r_0 as f64,
//!     &[__dr_0_0 as f64, __dr_0_1 as f64],
//!     grad,
//! );
//! ```
//!
//! Things to notice:
//!
//! - `singleroot.x.work()` -- each param access is rewritten to
//!   `work()` so the LM trial step is used in place of the stored
//!   value without mutating it.
//! - Derivatives for every param the constraint touches appear
//!   individually (`__dr_0_0`, `__dr_0_1`). The `0.0` entry for `y`
//!   is not elided because the index into `hb` is positional;
//!   dead rows fold out at optimisation time.
//! - The residual and the partials flow into the entity's Hessian
//!   block via `hb.add_residual(r, dr, grad)` -- one call per
//!   residual, accumulating `2*r*dr` into `grad` and `2*dr_i*dr_j`
//!   into the block's packed upper triangle.
//! - The `/// arael: ...` doc comment is a source marker pointing at
//!   the constraint attribute the block came from -- invaluable when
//!   the expansion runs to thousands of lines.
//!
//! ### Example: shared subexpressions
//!
//! In a larger body -- say a landmark observation that builds a
//! rotation matrix and reuses it across x/y/z residuals -- the
//! macro runs CSE before emitting code, so you see lines like
//!
//! ```ignore
//! let __cse_0 = cos(pose.ea.z.work());
//! let __cse_1 = sin(pose.ea.z.work());
//! let __cse_2 = __cse_0 * (lm.pos.x - pose.pos.x.work())
//!             + __cse_1 * (lm.pos.y - pose.pos.y.work());
//! // __cse_2 reused in __r_0, __r_1, and every __dr_* that needs it
//! ```
//!
//! Reading these tells you what the compiler *actually* has to
//! evaluate -- useful for understanding the cost of a constraint,
//! spotting accidental non-shared work, and sanity-checking that
//! symbolic simplification collapsed things you expected it to.
//!
//! ### What to look for
//!
//! - **`__set_block_indices`** -- where each `SelfBlock` /
//!   `CrossBlock` / `TripletBlock` gets its global parameter indices
//!   written into place. A block that isn't touched here is invisible
//!   to the solver (its `u32::MAX` sentinel causes every `add_residual`
//!   to silently skip) -- a common failure mode.
//! - **`__compute_blocks`** -- the grad + block-Hessian accumulation
//!   path. Each constraint is a nested block with its own CSE'd body.
//! - **`calc_jacobian`** -- same body structure but builds a
//!   `JacobianRow` per residual instead of accumulating into the
//!   blocks. Generated only when you declare `#[arael(root, jacobian)]`.
//! - **source markers** -- doc comments like
//!   `/// arael: PointFrine[<name>] @ path/to/file.rs:NNN`
//!   pinpoint the constraint attribute each block came from.
//!
//! Expansion grows quickly (the single_root demo is ~800 lines; a
//! full SLAM model is several thousand). Use `sed -n` or a pager
//! scoped to the method you care about:
//!
//! ```bash
//! cargo expand --example slam_demo | sed -n '/fn __compute_blocks/,/^    fn /p'
//! ```
//!
//! # 2D Sketch Editor
//!
//! The `arael-sketch` crate provides an interactive constraint-based 2D sketch
//! editor built on the optimization framework. It combines both differentiation
//! modes: geometric constraints (horizontal, coincident, parallel, tangent, etc.)
//! use compile-time differentiation, while parametric dimensions use runtime
//! differentiation -- the user types an expression like `d0 * 2 + 3` and the
//! solver constrains the geometry to satisfy it in real time, with full symbolic
//! derivatives. Dimensions can reference each other, entity properties
//! (`L0.length`, `A0.radius`), and arithmetic expressions, making it a fully
//! parametric constraint solver. Runs natively and in the browser via
//! WebAssembly.
//!
//! [![Sketch Editor](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/sketch.png)](https://sketch.mare.ee/)
//!
//! [Try it in the browser](https://sketch.mare.ee/)
//!
//! The editor includes a command panel (`/` to toggle) with 40+ scripting
//! commands, and an embedded MCP server (`--mcp`) that enables AI agents
//! like Claude Code to create and modify sketches programmatically.
//!
//! ![Dark mode](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/arael-sketch/docs/dark.png)
//!
//! # Example: SLAM Constraints
//!
//! For multi-body optimization (SLAM, bundle adjustment), define model
//! hierarchies with constraints. The macro handles symbolic differentiation,
//! reference resolution, and code generation.
//!
//! ![Hessian Sparsity](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/sparsity.png)
//!
//! The sparsity pattern shows pose-pose blocks (upper-left), pose-landmark
//! coupling (off-diagonal), and landmark self-blocks (lower-right). Sparse
//! Cholesky exploits this for large speedups over dense.
//!
//! ```ignore
//! #[arael::model]
//! #[arael(constraint(hb_pose, guard = self.info.gps.is_some(), {
//!     // GPS constraint (guarded -- only when data present)
//!     let raw = pose.pos - pose.info.gps.pos;
//!     let whitened = pose.info.gps.cov_r.transpose() * raw;
//!     [gamma * atan(whitened.x * pose.info.gps.cov_isigma.x / gamma), ...]
//! }))]
//! #[arael(constraint(hb_pose, {
//!     // Tilt sensor -- accelerometer constrains roll and pitch
//!     [(pose.ea.x - pose.info.tilt_roll) * path.tilt_isigma,
//!      (pose.ea.y - pose.info.tilt_pitch) * path.tilt_isigma]
//! }))]
//! struct Pose {
//!     pos: Param<vect3f>,
//!     ea: SimpleEulerAngleParam<f32>,
//!     info: PoseInfo,
//!     hb_pose: SelfBlock<Pose>,
//! }
//!
//! // Observation linking a landmark to a pose
//! #[arael::model]
//! #[arael(constraint(hb, parent=lm, {
//!     let mr2w = pose.ea.rotation_matrix();
//!     let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
//!     let r_f = feature.mf2r.transpose() * (lm_r - feature.camera_pos);
//!     [gamma * atan(atan2(r_f.y, r_f.x) * feature.isigma.x / gamma),
//!      gamma * atan(atan2(r_f.z, r_f.x) * feature.isigma.y / gamma)]
//! }))]
//! struct PointFrine {
//!     #[arael(ref = root.poses)]
//!     pose: Ref<Pose>,
//!     #[arael(ref = pose.info.features)]
//!     feature: Ref<PointFeature>,
//!     hb: CrossBlock<PointLandmark, Pose>,
//! }
//! ```
//!
//! See [examples/slam_demo.rs](https://github.com/harakas/arael/blob/master/examples/slam_demo.rs) for the full 60-pose, 240-landmark SLAM demo
//! with GPS, odometry, tilt sensor, graduated optimization, and covariance
//! estimation.
//! See [docs/SLAM.md](https://github.com/harakas/arael/blob/master/docs/SLAM.md) for the full walkthrough.
//!
//! # Example: Localization
//!
//! Same model as SLAM but landmarks are fixed (known map). With only pose
//! parameters the hessian is block-tridiagonal, so the band solver gives
//! O(n) scaling -- 9.4x faster than dense at 500 poses.
//! See [examples/loc_demo.rs](https://github.com/harakas/arael/blob/master/examples/loc_demo.rs).
//!
//! # Examples
//!
//! The `examples/` directory is the primary place to see the API in use.
//! Each file is a runnable `cargo run --release --example <name>`.
//!
//! - **[`bench_band`](https://github.com/harakas/arael/blob/master/examples/bench_band.rs)**
//!   -- benchmarks the band Cholesky backend against dense on the
//!   localisation model at increasing pose counts. Prints timing +
//!   speedup.
//! - **[`bench_investigate`](https://github.com/harakas/arael/blob/master/examples/bench_investigate.rs)**
//!   -- deeper comparison of sparse backends (faer, schur) on the
//!   SLAM model, with assembly vs solve breakdown and numeric
//!   cross-check of the solutions.
//! - **[`bench_sparse`](https://github.com/harakas/arael/blob/master/examples/bench_sparse.rs)**
//!   -- sparse Cholesky backends (faer / schur) vs dense on SLAM.
//! - **[`calc_demo`](https://github.com/harakas/arael/blob/master/examples/calc_demo.rs)**
//!   -- `bc`-style REPL calculator built on `arael-sym`. Shows
//!   `parse_with_functions` + `FunctionBag` for user-defined
//!   functions, persistent history via rustyline.
//! - **[`jacobian_demo`](https://github.com/harakas/arael/blob/master/examples/jacobian_demo.rs)**
//!   -- `#[arael(root, jacobian)]`, `#[arael(constraint_index)]`, and
//!   `calc_jacobian` / `calc_cost_table` walk-through. End-to-end
//!   reference for the instrumentation features referenced from
//!   "My solve doesn't converge".
//! - **[`linear_demo`](https://github.com/harakas/arael/blob/master/examples/linear_demo.rs)**
//!   -- robust linear regression on noisy 2D data. Residual wrapped
//!   in `gamma * atan(r / gamma)` -- the Starship method
//!   (US12346118), same robustifier used by the feature constraints
//!   in loc/SLAM. Minimal single-struct model + LM fit, compared
//!   against plain closed-form least squares.
//! - **[`loc_demo`](https://github.com/harakas/arael/blob/master/examples/loc_demo.rs)**
//!   -- localisation with fixed known landmarks (no gauge freedom).
//!   Block-tridiagonal Hessian + band solver. Graduated-isigma
//!   optimisation via a root `frine_isigma_scale` field.
//! - **[`loc_global_demo`](https://github.com/harakas/arael/blob/master/examples/loc_global_demo.rs)**
//!   -- how to put `Param` fields on the root struct and have
//!   constraints consume them. Uses a system-global rigid transform
//!   (translation + 3-axis rotation applied to every pose) as the
//!   running example; every residual that reads the robot's world
//!   pose composes the globals before evaluating. Shows the two
//!   wiring shapes for pose<->root cross-Hessian pairs
//!   (`CrossBlock<Pose, Path>` on the constraint struct, and a
//!   root-owned `TripletBlock` named via the `root.<field>` block
//!   spec) and a `Path::optimise_center` pass that freezes pose
//!   params and optimises only the globals before the main sweep.
//! - **[`model_demo`](https://github.com/harakas/arael/blob/master/examples/model_demo.rs)**
//!   -- minimal `#[arael::model]` walk-through showing how
//!   `Param`, `SimpleEulerAngleParam`, and the update cycle fit
//!   together.
//! - **[`refs_demo`](https://github.com/harakas/arael/blob/master/examples/refs_demo.rs)**
//!   -- `Ref<T>`, `refs::Vec`, `refs::Deque`, and `refs::Arena`
//!   behaviour: insertion, iteration, stable handles.
//! - **[`runtime_fit_demo`](https://github.com/harakas/arael/blob/master/examples/runtime_fit_demo.rs)**
//!   -- curve fitting where the residual equation is a string parsed
//!   at runtime. Demonstrates `ExtendedModel` + robust loss on top
//!   of the symbolic front end.
//! - **[`single_root_demo`](https://github.com/harakas/arael/blob/master/examples/single_root_demo.rs)**
//!   -- single-struct model-and-root + a direct-composed sub-model,
//!   each carrying its own `SelfBlock<Self>`. The smallest example
//!   that exercises the "root has its own params" path.
//! - **[`slam_demo`](https://github.com/harakas/arael/blob/master/examples/slam_demo.rs)**
//!   -- synthetic visual-inertial SLAM: S-curve trajectory, 20 poses,
//!   40 landmarks, odometry + tilt + GPS + feature observations.
//!   Full verbose-LM trace across graduated isigma passes -- the
//!   reference for what a healthy solver run looks like.
//! - **[`sym_demo`](https://github.com/harakas/arael/blob/master/examples/sym_demo.rs)**
//!   -- symbolic-math tour: expression building, automatic
//!   differentiation, CSE, pretty printing, parsing. No solver
//!   involvement; pure `arael-sym`.
//!
//! # Crate structure
//!
//! - `arael-sym` -- symbolic math engine (expression trees, differentiation, CSE)
//! - `arael-macros` -- proc macros (`#[arael::model]`, `#[derive(Model)]`)
//! - `arael` (this crate) -- runtime: model traits, solvers, geometry, vectors

#[macro_use]
mod log;
/// Numeric traits (`Float`).
pub mod utils;
/// 2D and 3D vector types.
pub mod vect;
/// 3x3 and 2x2 matrix types with rotation and linear algebra.
pub mod matrix;
/// Quaternion type for 3D rotations.
pub mod quatern;
/// Type-safe indexed collections: `Ref`, `Vec`, `Deque`, `Arena`.
pub mod refs;
/// Re-export of the `arael-sym` symbolic math crate.
pub use arael_sym as sym;
/// Model trait, parameter types, and Hessian blocks.
pub mod model;
/// Levenberg-Marquardt solver with dense, band, and sparse backends.
pub mod simple_lm;
/// Camera model and geometric utilities.
pub mod geometry;

/// Re-export Jacobian types for convenient access.
pub use model::{Jacobian, JacobianRow, jacobian_entries};
/// The `sym!` auto-clone macro for symbolic expressions (from `arael-sym`).
pub use arael_sym::sym;
/// Derive macro for the `Model` trait.
pub use arael_macros::Model;
/// Attribute macro: `#[arael::model]`.
pub use arael_macros::model;
