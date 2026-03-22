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
//! - **Hessian blocks** -- `SelfBlock<A>` and `CrossBlock<A, B>` generic
//!   over float type
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
//!     println!("f(2.0) = {}", f.eval(&vars)); // 2.8185...
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
//! # 2D Sketch Editor
//!
//! The `arael-sketch` crate provides an interactive constraint-based 2D sketch
//! editor built on the optimization framework. Runs natively and in the browser
//! via WebAssembly.
//!
//! [![Sketch Editor](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/sketch.png)](https://sketch.mare.ee/)
//!
//! [Try it in the browser](https://sketch.mare.ee/)
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

/// The `sym!` auto-clone macro for symbolic expressions (from `arael-sym`).
pub use arael_sym::sym;
/// Derive macro for the `Model` trait.
pub use arael_macros::Model;
/// Attribute macro: `#[arael::model]`.
pub use arael_macros::model;
