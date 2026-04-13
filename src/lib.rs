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
//!     fn extended_compute64(&mut self, params: &[f64]) {
//!         for &(x, y) in &self.data {
//!             vars.insert("x", x);
//!             vars.insert("y", y);
//!             let r = residual.eval(&vars).unwrap();
//!             let dr: Vec<f64> = self.derivs.iter()
//!                 .map(|(_, _, d)| d.eval(&vars).unwrap()).collect();
//!             self.hb.add_residual(r, &indices, &dr);
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
//! 2. **NaN or Inf residuals / derivatives.** Walk the Jacobian rows:
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
//!    A NaN residual or partial derivative usually means a `sqrt` or
//!    `atan2` saw a degenerate input (zero-length vector, both-zero
//!    arguments). Use `arael_sym::safe_sqrt` / `safe_atan2` in the
//!    constraint body -- they return finite values and non-diverging
//!    derivatives at the singular point.
//!
//! 3. **Gradient magnitude.** After
//!    [`simple_lm::LmProblem::calc_grad_hessian_dense`], the maximum
//!    absolute gradient component should be small relative
//!    to the cost scale at a local minimum. A huge gradient with tiny
//!    cost means the parameter scaling is off -- one parameter moves
//!    cost several orders of magnitude more than another, which
//!    destabilises Levenberg-Marquardt.
//!
//! 4. **Hessian health.** The same `hessian` array should be finite and
//!    positive-semi-definite at a minimum (smallest eigenvalue ≥ 0
//!    modulo roundoff). A significantly-negative smallest eigenvalue
//!    means the Gauss-Newton approximation J^T J is a poor local fit
//!    -- often because constraints are ill-conditioned or cancel.
//!
//! 5. **Rank / DOF.** Call [`model::Jacobian::singular_values`] (or the
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
