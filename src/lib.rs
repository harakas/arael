//! ARAEL -- Algorithms for Robust Autonomy, Estimation, and Localization.
//!
//! Nonlinear least-squares optimization framework.
//!
//! You write constraints as symbolic expressions inside the model
//! definition; at compile time Rust's procedural macros differentiate,
//! simplify, eliminate common subexpressions -- and emit plain Rust for
//! calculating the cost, gradient, and Gauss-Newton hessian (J^T J).
//!
//! Solve regression and curve fitting, sensor fusion, SLAM, bundle
//! adjustment, pose-graph and geometric constraint optimization.
//!
//! Instead of constructing a graph, you build a hierarchical data structure
//! from plain Rust structs and containers. This yields high performance and
//! low memory usage. The optimizer can also be automatically exported to
//! C++, Python and JavaScript (WebAssembly), where the generated classes
//! mirror your Rust model structure.
//!
//! # Contents
//!
//! - [Features](#features)
//! - [Benchmarks](#benchmarks)
//! - [Scope](#scope)
//! - [Example: Symbolic Math](#example-symbolic-math)
//! - [Example: Robust Linear Regression](#example-robust-linear-regression)
//! - [Example: SLAM Constraints](#example-slam-constraints)
//! - [Runtime Differentiation](#runtime-differentiation)
//! - [Model Structure](#model-structure)
//! - [Cross-Crate Models](#cross-crate-models)
//! - [C++, Python and JavaScript Interfaces](#c-python-and-javascript-interfaces)
//! - [Solvers](#solvers)
//!   - [Solving on a GPU](#solving-on-a-gpu)
//! - [Parameter Covariance](#parameter-covariance)
//! - [Instrumentation & Debugging](#instrumentation--debugging)
//! - [Robustness and loss functions](#robustness-and-loss-functions)
//! - [Examples](#examples)
//! - [2D Sketch Editor](#2d-sketch-editor)
//! - [Crate structure](#crate-structure)
//!
//! # Features
//!
//! - Symbolic math: expression trees with automatic differentiation,
//!   simplification and LaTeX and Rust code generation
//! - Constraints written as math and compiled at build time into
//!   derivative code, with common subexpressions computed once
//! - Levenberg-Marquardt solver with robust losses and constraints that
//!   switch themselves off
//! - C++, Python and JavaScript (WebAssembly) interfaces generated from a
//!   model, so a model built in Rust is solved from those languages and in
//!   the browser
//!   ([docs/CXX.md](https://github.com/harakas/arael/blob/master/docs/CXX.md),
//!   [docs/PYTHON.md](https://github.com/harakas/arael/blob/master/docs/PYTHON.md),
//!   [docs/WASM.md](https://github.com/harakas/arael/blob/master/docs/WASM.md))
//! - Sparse Cholesky in block form, landmark-style blocks marginalized
//!   automatically, and conjugate gradients in place of a factor
//!   ([docs/SOLVERS.md](https://github.com/harakas/arael/blob/master/docs/SOLVERS.md))
//! - Dense, band and sparse backends in pure Rust; optional Eigen, CHOLMOD
//!   and LAPACK backends (CHOLMOD's supernodal module is GPL)
//! - Multithreaded assembly and factorization (see [Threads](#threads))
//! - Solving on NVIDIA, AMD and Apple silicon GPUs (see
//!   [Solving on a GPU](#solving-on-a-gpu))
//! - f32 and f64 models, and mixed precision: an f32 factor refined to f64,
//!   wider arithmetic inside an f32 model where it is needed
//! - Fast repeated solves of one problem: the analysis of its structure is
//!   kept
//! - Rotation, rigid and similarity transform, unit direction and 2D
//!   heading parameters, without gimbal lock
//! - Typed references between entities, held in collections that keep
//!   them valid
//! - Parameter covariance at the solution, without a dense inverse
//! - Model validation, gradient checks, and Jacobians for constraint
//!   diagnostics
//! - Equations parsed and differentiated at run time, and user-defined
//!   functions in constraint bodies
//! - Models shared across crates
//!
//! # Benchmarks
//!
//! <picture>
//!   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/harakas/arael/master/benchmarks/charts/v0.9.0/pgo-dark.svg">
//!   <img alt="2x2 bar charts of per-iteration solve time on the M3500, city10000, sphere2500 and parking-garage pose graphs, comparing arael in f32 and f64 against g2o, Ceres, SymForce, factrs and GTSAM" src="https://raw.githubusercontent.com/harakas/arael/master/benchmarks/charts/v0.9.0/pgo-light.svg">
//! </picture>
//!
//! Time per iteration on the four canonical pose-graph datasets, 2D and
//! 3D. One iteration is one linearize + assemble + factorize + solve over
//! the identical validated cost function -- the same work in every system,
//! independent of each solver's damping schedule -- measured as the
//! difference between a two-iteration and a one-iteration solve, so the
//! one-time setup cancels out.
//!
//! Full methodology, the initial-damping policy, and the cross-system
//! validation harness:
//! [benchmarks/pgo](https://github.com/harakas/arael/tree/master/benchmarks/pgo).
//! A bundle-adjustment benchmark on the BAL Ladybug problems (arael vs
//! Ceres and g2o):
//! [benchmarks/bal](https://github.com/harakas/arael/tree/master/benchmarks/bal).
//! A heterogeneous monocular SLAM benchmark (six factor types, seven
//! systems), over open, loop-closing and figure-8 trajectories and
//! including a Raspberry Pi edge run:
//! [benchmarks/slam](https://github.com/harakas/arael/tree/master/benchmarks/slam).
//! A plane-landmark SLAM benchmark, where six systems each parameterize the
//! plane normal differently:
//! [benchmarks/plane](https://github.com/harakas/arael/tree/master/benchmarks/plane).
//!
//! <picture>
//!   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/harakas/arael/master/benchmarks/charts/v0.9.0/slam-loc-dark.svg">
//!   <img alt="2x2 bar charts of per-iteration solve time (top) and peak process memory (bottom): landmark SLAM at 1200 poses on an Apple M4 Pro, and localization at 60 poses on a Raspberry Pi 5" src="https://raw.githubusercontent.com/harakas/arael/master/benchmarks/charts/v0.9.0/slam-loc-light.svg">
//! </picture>
//!
//! The same per-iteration metric, and the peak memory of those runs, on
//! the landmark-SLAM benchmark and on fixed-map localization running on a
//! Raspberry Pi 5
//! ([benchmarks/loc](https://github.com/harakas/arael/tree/master/benchmarks/loc)):
//! with the landmarks fixed the localization Hessian is
//! block-tridiagonal, and arael solves it with its band Cholesky.
//!
//! <picture>
//!   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/harakas/arael/master/benchmarks/charts/v0.9.0/threads-dark.svg">
//!   <img alt="Two grouped bar charts, landmark SLAM at 1200 poses and bundle adjustment on Ladybug-372 on an AMD Ryzen 9 7950X3D: one complete iteration at 1, 2, 4 and 8 threads for arael f64, arael f32 and Ceres" src="https://raw.githubusercontent.com/harakas/arael/master/benchmarks/charts/v0.9.0/threads-light.svg">
//! </picture>
//!
//! The same iteration at one to eight threads, on the landmark-SLAM
//! benchmark and on bundle adjustment, against Ceres on the same machine
//! (see [Threads](#threads)).
//!
//! # Scope
//!
//! Arael is a nonlinear optimization framework, not a complete SLAM or
//! state estimation system. The SLAM and localization demos show how to use
//! arael as the optimizer backend, but a production SLAM pipeline would
//! additionally need:
//!
//! - Front-end perception: feature detection, descriptor extraction
//! - Data association: matching observed features to existing landmarks,
//!   handling ambiguous or incorrect matches
//! - Landmark management: initializing new landmarks from observations,
//!   merging duplicates, pruning unreliable ones
//! - Keyframe selection: deciding when to add new poses vs. discard
//!   redundant frames
//! - Loop closure: detecting revisited places, verifying loop closure
//!   candidates, and injecting constraints
//! - Outlier rejection logic: deciding which observations to reject
//! - Marginalization / sliding window: limiting optimization scope for
//!   real-time operation, marginalizing old poses while preserving their
//!   information
//! - Map management: spatial indexing, map saving/loading, multi-session
//!   map merging
//!
//! Arael provides the compile-time-differentiated solver that sits at the
//! core of such a system. Everything above is application-level logic that
//! builds on top of it.
//!
//! # Example: Symbolic Math
//!
//! ```
//! use arael::sym::*;
//!
//! arael::sym! {
//!     let x = symbol("x");
//!     let f = sin(x) * x + 1.0;
//!
//!     println!("f(x)    = {}", f);           // sin(x) * x + 1
//!     println!("f'(x)   = {}", f.diff("x")); // x * cos(x) + sin(x)
//!     println!("as Rust = {}", f.diff("x").to_rust("f64")); // x * x.cos() + x.sin()
//!
//!     let vars = std::collections::HashMap::from([("x", 2.0)]);
//!     println!("f(2.0)  = {}", f.eval(&vars).unwrap()); // 2.8185...
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
//! ```
//! # use arael::model::Param;
//! # use arael::simple_lm::FitProblem;
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
//! # let mut m = LinearModel {
//! #     a: Param::new(0.0), b: Param::new(0.0),
//! #     data: vec![DataEntry { x: 0.0, y: 1.0 }, DataEntry { x: 1.0, y: 3.0 },
//! #                DataEntry { x: 2.0, y: 5.0 }],
//! #     sigma: 1.0, gamma: 10.0,
//! # };
//! # let r = m.fit().unwrap();
//! # assert!(r.end_cost < r.start_cost);
//! # assert!((m.a.value - 2.0).abs() < 1e-2 && (m.b.value - 1.0).abs() < 1e-2);
//! ```
//!
//! The macro generates `calc_cost()`, the `calc_grad_hessian_*` backend
//! family, and `fit()`/`fit_with()`
//! with symbolically differentiated, CSE-optimized compiled code.
//! The robust fit ignores outliers while tracking the inlier data:
//!
//! ![Linear Regression](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/linear_regression.png)
//!
//! See [examples/linear_demo.rs](https://github.com/harakas/arael/blob/master/examples/linear_demo.rs) for the full source.
//!
//! # Example: SLAM Constraints
//!
//! A 2D SLAM problem: find the most likely robot path and landmark positions
//! from the sensor readings. Each pose is a position `(x, y)` and a heading
//! `gamma`; each landmark is a point `(x, y)`. The sensors give the angle
//! (bearing) to a landmark from the poses that saw it, and how far the robot
//! moved between consecutive poses -- never the distance to a landmark, only
//! its direction. You define the model as plain Rust structs, each
//! `#[arael(constraint(...))]` computing one or more residuals -- the
//! differences between what the model predicts and what the sensor measured.
//! Driving them toward zero pulls the estimate into agreement with the data,
//! a soft constraint on the poses and landmarks involved; the same
//! per-measurement term is what factor-graph frameworks call a factor (GTSAM)
//! or an edge (g2o). There is no factor graph to build here, though -- each
//! residual lives as code right next to the data it reads.
//!
//! ```ignore
//! // A robot pose, plus the movement measured since the previous pose.
//! #[arael::model]
//! struct Pose {
//!     pos: Param<vect2f>,            // solved for: position (x, y)
//!     gamma: Param<f32>,             // solved for: heading (0 = east, turning left is positive)
//!     delta_pos: vect2f,             // measured: movement since the previous pose
//!     delta_gamma: f32,              // measured: change of heading since the previous pose
//!     delta_pos_isigma: f32,         // 1 / sigma, where sigma is the sensor's uncertainty (standard deviation)
//!     delta_gamma_isigma: f32,
//!     hb_pose: SelfBlock<Pose, f32>, // solver storage for this pose's parameters
//! }
//!
//! // Two consecutive poses: their actual relative motion must match the
//! // measured movement (delta_pos, delta_gamma).
//! #[arael::model]
//! #[arael(constraint(hb, {
//!     let local = matrix2sym::rotation(prev.gamma).transpose() * (cur.pos - prev.pos);
//!     [(local.x - cur.delta_pos.x) * cur.delta_pos_isigma,
//!      (local.y - cur.delta_pos.y) * cur.delta_pos_isigma,
//!      rad_diff(cur.gamma - prev.gamma, cur.delta_gamma) * cur.delta_gamma_isigma]
//! }))]
//! struct PosePair {
//!     #[arael(ref = root.poses)] prev: Ref<Pose>,  // Ref<T> = typed index into a root collection
//!     #[arael(ref = root.poses)] cur: Ref<Pose>,
//!     hb: CrossBlock<Pose, Pose, f32>,             // solver storage for the two poses this couples
//! }
//!
//! // A landmark and the bearing sightings that observed it.
//! #[arael::model]
//! struct Landmark {
//!     pos: Param<vect2f>,
//!     frines: std::vec::Vec<Frine>,
//!     hb: SelfBlock<Landmark, f32>,
//! }
//!
//! // One bearing sighting of `lm` from `pose`: the residual is the angle difference
//! // between the landmark's actual direction and the measured bearing (zero when they agree).
//! #[arael::model]
//! #[arael(constraint(hb, parent = lm, {
//!     let world_angle = pose.gamma + frine.bearing;
//!     let aligned = matrix2sym::rotation(world_angle).transpose() * (lm.pos - pose.pos);
//!     [atan2(aligned.y, aligned.x) * frine.isigma]
//! }))]
//! struct Frine {
//!     #[arael(ref = root.poses)] pose: Ref<Pose>,
//!     bearing: f32,
//!     isigma: f32,
//!     hb: CrossBlock<Landmark, Pose, f32>,
//! }
//!
//! // The root. #[arael(root, f32)] triggers codegen over everything reachable.
//! #[arael::model]
//! #[arael(root, f32)]
//! struct Path {
//!     poses: refs::Deque<Pose>,
//!     pose_pairs: std::vec::Vec<PosePair>,
//!     landmarks: refs::Arena<Landmark>,
//! }
//! ```
//!
//! Build the problem and solve it -- `solve_sparse` runs Levenberg-Marquardt
//! on the faer sparse backend and writes the optimized values back into the
//! structs:
//!
//! ```ignore
//! let (mut path, ..) = build_path(&Cfg::default());  // synthetic arc + noisy bearings
//! let result = path.solve_sparse(&LmConfig::well_conditioned().with_verbose(true)).unwrap();
//! ```
//!
//! Full runnable demo:
//! [examples/slam2d_simple_demo.rs](https://github.com/harakas/arael/blob/master/examples/slam2d_simple_demo.rs).
//! For the 3D version -- full position and orientation per pose, camera
//! bearings, and rejection of wrong measurements -- see
//! [examples/slam_demo.rs](https://github.com/harakas/arael/blob/master/examples/slam_demo.rs)
//! and the full walkthrough in
//! [docs/SLAM.md](https://github.com/harakas/arael/blob/master/docs/SLAM.md).
//! For localization against a known map -- the same model with the
//! landmarks fixed, whose block-tridiagonal hessian lets the O(n) band
//! solver run -- see
//! [examples/loc_demo.rs](https://github.com/harakas/arael/blob/master/examples/loc_demo.rs).
//!
//! # Runtime Differentiation
//!
//! Compile-time differentiation generates optimized Rust code with CSE at
//! build time -- ideal when the model structure is fixed. But many
//! applications need equations that are only known at runtime: user-typed
//! formulas in a CAD parametric dimension, configuration-driven curve
//! fitting, or symbolic constraints loaded from a file.
//!
//! Arael supports this through runtime differentiation: parse an
//! equation string with `arael_sym::parse`, symbolically differentiate
//! once at setup with `E::diff`, then evaluate the expression tree
//! numerically each solver iteration. The
//! [`ExtendedModel`](model::ExtendedModel) trait is the integration
//! point with the LM solver: the solve hands its hook a
//! [`Coo`](model::Coo) list to push Hessian entries into.
//!
//! The sketch editor (`arael-sketch`) uses this extensively for parametric
//! expression dimensions -- a user can type `d0 * 2 + 3` as a dimension
//! value, and the solver constrains the geometry to satisfy the equation
//! in real time, with full symbolic derivatives.
//!
//! The model uses `#[arael(root, extended)]` and implements
//! [`ExtendedModel`](model::ExtendedModel) to push residuals and
//! derivatives into the list at each solver iteration:
//!
//! ```ignore
//! #[arael::model]
//! #[arael(root, extended, seq)]
//! struct RegressionModel {
//!     coeffs: refs::Vec<Coefficient>,         // optimizable parameters
//!     residual_expr: Option<arael_sym::E>,    // parsed equation
//!     derivs: Vec<(String, u32, arael_sym::E)>, // pre-computed derivatives
//!     // ...
//! }
//!
//! impl ExtendedModel<f64> for RegressionModel {
//!     fn extended_compute(&mut self, params: &[f64], grad: &mut [f64], coo: &mut Coo<f64>) {
//!         for &(x, y) in &self.data {
//!             vars.insert("x", x);
//!             vars.insert("y", y);
//!             let r = residual.eval(&vars).unwrap();
//!             let dr: Vec<f64> = self.derivs.iter()
//!                 .map(|(_, _, d)| d.eval(&vars).unwrap()).collect();
//!             // writes 2*r*dr into grad AND pushes full upper-triangle
//!             // Hessian into coo (one call, both done)
//!             coo.add_residual(r, &indices, &dr, grad);
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
//! # Model Structure
//!
//! This is the reference for every piece that appears in an
//! `#[arael::model]` declaration: parameter types, Hessian blocks,
//! collection types, macro attributes, and the patterns for placing
//! constraints. For end-to-end code, see
//! [examples/single_root_demo.rs](https://github.com/harakas/arael/blob/master/examples/single_root_demo.rs)
//! (smallest complete model) and
//! [examples/slam_demo.rs](https://github.com/harakas/arael/blob/master/examples/slam_demo.rs)
//! (full SLAM setup). The full reference is
//! [docs/MODEL.md](https://github.com/harakas/arael/blob/master/docs/MODEL.md).
//!
//! ## Parameter types
//!
//! | Type | Size | Use it when |
//! |---|---|---|
//! | [`Param<f32>`](model::Param) / [`Param<f64>`](model::Param) | 1 | scalar parameter |
//! | [`Param<vect2<T>>`](vect::vect2) | 2 | 2D point / direction |
//! | [`Param<vect3<T>>`](vect::vect3) | 3 | 3D position, velocity, linear vector |
//! | [`AngleParam<T>`](angle::AngleParam) | 1 | a 2D rotation (heading): the angle is optimized directly and its rotation matrix is cached, so constraints that rotate through it read a cached constant instead of recomputing sin/cos per observation |
//! | [`SimpleEulerAngleParam<T>`](model::SimpleEulerAngleParam) | 3 | three direct Euler angles (roll, pitch, yaw) |
//! | [`EulerAngleParam<T>`](model::EulerAngleParam) | 3 | "universal" delta composed with a fixed reference rotation; avoids parameterisation singularities for large-angle motion |
//! | [`MixedEulerAngleParam<T, H = f64>`](model::MixedEulerAngleParam) | 3 | `EulerAngleParam` with its state in the wide scalar `H` whatever `T` and the caches a body reads in `T`; for an f32 model whose body reads the rotation inside a `with_wide` scope |
//! | [`QuaternionParam<T>`](model::QuaternionParam) | 3 | a rotation-vector delta (not euler angles) composed with a unit-quaternion reference, renormalized each re-center so it never drifts off SO(3) |
//! | [`TransformParam<T>`](transform::TransformParam) | 6 | a rigid transform, such as a robot pose: a translation and a rotation moved together; the optimized delta is represented as a twist (se(3)), so a rotation correction carries the translation with it |
//! | [`ScaledTransformParam<T>`](transform::ScaledTransformParam) | 7 | a similarity transform (Sim(3)): `TransformParam` plus a uniform scale acting as `s * (R * x) + t` -- monocular loop closing's state. The scale is optimized as its logarithm; clear `optimize_scale` to fix it |
//! | [`UnitVecParam<T>`](unitvec::UnitVecParam) | 2 | a direction on the unit sphere, such as the normal of a mapped plane landmark: read and write `unit`, which stays unit length because the two parameters rotate a reference direction rather than move its components |
//!
//! ```ignore
//! #[arael::model]
//! struct Pose {
//!     pos: Param<vect3f>,              // 3 scalar params
//!     ea: SimpleEulerAngleParam<f32>,  // 3 scalar params (roll, pitch, yaw)
//!     // info: plain non-Param data (sigmas, measurements) is fine
//!     info: PoseInfo,
//!     hb_pose: SelfBlock<Pose, f32>,   // mandatory; see below
//! }
//! ```
//!
//! Every parameter stores a current `value` and a per-iteration
//! `work()` copy. The macro rewrites `pose.ea` in a constraint body
//! to `pose.ea.work()` so the LM trial step is evaluated without
//! mutating the stored value until the step is accepted.
//!
//! ### Initial values via the `_value` suffix
//!
//! Inside a constraint body, `pose.pos_value` (any `<field>_value`)
//! resolves to the original stored value of the parameter -- the
//! point the LM trial-step is measured against, not the trial step
//! itself. Use it to build drift / regularising residuals:
//!
//! ```ignore
//! let pos_drift = pose.pos - pose.pos_value;
//! [pos_drift.x * path.drift_pos_isigma,
//!  pos_drift.y * path.drift_pos_isigma,
//!  pos_drift.z * path.drift_pos_isigma]
//! ```
//!
//! ## Parameter control
//!
//! Three ways to keep a parameter out of the solve:
//!
//! 1. [`Param::fixed(v)`](model::Param::fixed) at construction --
//!    immutable; the macro never emits indices for it. Typical use:
//!    problem-wide constants that happen to be parameter-shaped
//!    (e.g. a known camera pose).
//! 2. Mutate `.optimize` at runtime -- `pose.pos.optimize = false;`
//!    freezes a live parameter for the next solve call. Flip it back
//!    on to re-include. Use for staged optimisation (freeze subset,
//!    solve, unfreeze, solve again).
//! 3. The `_value` trick above -- the parameter still moves but
//!    the residual is anchored to its initial position via a drift
//!    constraint.
//!
//! ```ignore
//! // (1) fixed at construction
//! let camera = Pose {
//!     pos: Param::fixed(known_position),     // never optimised
//!     ea:  Param::fixed(known_orientation),
//!     /* ... */
//! };
//!
//! // (2) runtime freeze for a staged solve
//! for pose in path.poses.iter_mut() {
//!     pose.pos.optimize = false;
//!     pose.ea.optimize = false;
//! }
//! // ...solve only the remaining live params...
//! for pose in path.poses.iter_mut() {
//!     pose.pos.optimize = true;
//!     pose.ea.optimize = true;
//! }
//! ```
//!
//! See `Path::optimise_center` in `loc_global_demo.rs` for a real
//! usage of runtime `.optimize = false` -- it freezes every pose,
//! solves only for the root-level global rigid transform, bakes the
//! result into the poses, then un-freezes.
//!
//! ## Hessian block types
//!
//! The full Gauss-Newton Hessian is a symmetric block matrix,
//! with one block per (entity, entity) pair in the parameter
//! vector. The block at position `(Ei, Ej)` is the `NEi x NEj`
//! matrix of second partials; by symmetry
//! `H[Ei, Ej] = H[Ej, Ei]^T`. Each unique block is kept once and the
//! accumulator fills in the transpose when assembling into a dense /
//! band / sparse matrix.
//!
//! You declare which blocks the Hessian has by declaring block fields
//! on the model. A block field is a marker, not a container: it names
//! a coupling and carries that block's place in the block store, a
//! generated struct of one array per declared block field that the
//! solve keeps in its [`threads::Context`].
//!
//! - A diagonal block (`Ei == Ej`) is declared by `Ei`'s own
//!   `SelfBlock<Ei>`. It is symmetric, so only its upper triangle is
//!   kept, and every constraint touching `Ei`'s parameters adds to it.
//! - An off-diagonal block (`Ei != Ej`) is declared by a
//!   `CrossBlock<Ei, Ej>`, or, unnamed, through the `coo` keyword.
//!   One `CrossBlock<A, B>` declares both `H[A, B]` and its
//!   transpose `H[B, A]`; the accumulator writes both halves from the
//!   one rectangle.
//!
//! Gradient contributions `2 * r * dr` reach the solver's gradient
//! vector directly. Only the Hessian is described block by block.
//!
//! Pick the block shape that matches the constraint body's parameter
//! reach:
//!
//! | Type | Declares | Pick it when |
//! |---|---|---|
//! | [`SelfBlock<T>`](model::SelfBlock) | the `(T, T)` diagonal block -- entity T coupled with itself | mandatory on every params-having struct |
//! | [`CrossBlock<A, B>`](model::CrossBlock) | the `(A, B)` off-diagonal block, and its transpose | default for cross-entity Hessian pairs. Packed in-place writes, cheap to assemble. One per unordered (A, B) entity pair; (A, A) / (B, B) diagonals stay on each entity's `SelfBlock` |
//! | [`VarCrossBlock<A, B>`](model::VarCrossBlock) | the same block, stored at the size of the two entities' live parameters | entities whose parameters are commonly held, in whole or in part: a held parameter then costs the assembly nothing. A `CrossBlock` write is at constant offsets, so keep it where every parameter is usually live |
//! | [`VarSelfBlock<T>`](model::VarSelfBlock) | the `(T, T)` diagonal block, stored at the size of the entity's live parameters | an entity whose parameters are commonly held and which many residuals write |
//! | `coo` (a keyword, not a field) | across-entity pairs a constraint does not name, kept as row, column and value in a list the solve owns | pairs between an entity's params and the root's or a containing parent's (`[hb, coo]`), N-ary constraints without a block per pair (`constraint(coo, ...)`), and the [`ExtendedModel`](model::ExtendedModel) hook, which is handed the list. Slower to assemble than a `CrossBlock`: every entry is a push |
//!
//! `SelfBlock<Self>` is required on every Model that has
//! parameters -- omitting it is a compile-time error. An entity's
//! diagonal block is always its own `SelfBlock`; `CrossBlock` and
//! `coo` carry cross pairs only.
//!
//! ```
//! # use arael::model::{Model, Param, SimpleEulerAngleParam, SelfBlock, CrossBlock};
//! # use arael::refs::{self, Ref};
//! # use arael::vect::vect3f;
//! // Entity with its mandatory SelfBlock.
//! #[arael::model]
//! struct Pose {
//!     pos: Param<vect3f>,
//!     ea:  SimpleEulerAngleParam<f32>,
//!     hb_pose: SelfBlock<Pose, f32>,   // required
//! }
//!
//! // Constraint struct linking two entities via a CrossBlock.
//! #[arael::model]
//! #[arael(constraint(hb, {
//!     let d = cur.pos - prev.pos;
//!     [d.x, d.y, d.z]
//! }))]
//! struct PosePair {
//!     #[arael(ref = root.poses)] prev: Ref<Pose>,
//!     #[arael(ref = root.poses)] cur:  Ref<Pose>,
//!     hb: CrossBlock<Pose, Pose, f32>, // only the (prev, cur) cross block
//! }
//! # // The root triggers code generation for the whole model (entities
//! # // must be defined before it -- see docs/MODEL.md).
//! # #[arael::model]
//! # #[arael(root, f32)]
//! # struct World {
//! #     poses: refs::Vec<Pose>,
//! #     pairs: std::vec::Vec<PosePair>,
//! # }
//! ```
//!
//! ### Named pairs and the `coo` list
//!
//! A `CrossBlock` names its two sides in the type, so it gets a
//! packed tile of a size known in advance and writes in place. The
//! `coo` keyword names nothing: each entry is kept on its own, as a
//! triplet of row, column and value, the coordinate form a sparse
//! matrix is built from, in a list the solve owns. Every entry is a
//! push, so it assembles slower than a tile.
//!
//! So name the pairs wherever they can be named, and use `coo` where
//! they cannot. The macro accepts three shapes:
//!
//! - `constraint([hb_ab, hb_ac, hb_bc], { ... })` -- one
//!   `CrossBlock<A, B>` field per unordered entity pair on the
//!   constraint struct. Packed rectangular storage, one write per
//!   pair.
//! - `constraint(coo, { ... })` -- every across-entity pair of
//!   the constraint's refs goes to the list. Each participant keeps
//!   its diagonal on its own `SelfBlock`.
//! - `constraint([hb, coo], { ... })` -- the entity's own params
//!   couple to the root's or to its containing parent's: the body
//!   reads one of the two, and that is the co-entity. The pairs go to
//!   the list, the diagonals to each side's own `SelfBlock`.
//!   `parent = <name>` binds the parent under that name.
//!
//! `coo` stands alone or second behind the struct's own `SelfBlock`;
//! no other block list holds it. A constraint whose body reads only
//! the root's params, or only the parent's, names that block as its
//! primary instead (`root.hb`, `parent.hb`): dense writes, no list.
//!
//! Prefer multi-`CrossBlock` whenever the set of cross-pairs is
//! fixed and dense. The push grows a list and gives up locality,
//! whereas `CrossBlock` writes in place into a pre-sized `NA * NB`
//! rectangle at a known offset; the rectangle also suits the
//! factorization that follows.
//!
//! Reach for `coo` in two situations:
//!
//! 1. The root has its own `Param` fields and constraints
//!    couple per-entity params with root params. The (entity, root)
//!    pairs need a place; `[hb, coo]` gives them one without a block
//!    field per entity type. A `CrossBlock<Entity, Root>` on the
//!    constraint struct works too; `loc_global_demo` shows both.
//! 2. Runtime-parsed residuals via [`ExtendedModel`](model::ExtendedModel).
//!    The macro cannot enumerate pairs it does not see;
//!    `extended_compute` is handed the list and pushes into it -- see
//!    [examples/runtime_fit_demo.rs](https://github.com/harakas/arael/blob/master/examples/runtime_fit_demo.rs).
//!
//! Caveat for case 1 -- root-level `Param`s destroy sparsity.
//! Every constraint that reads a root param introduces an
//! (entity, root) cross pair in the Hessian. If *many* constraints
//! read the same root param -- which is the whole point of
//! "global" root params -- the root's rows and columns in the
//! Hessian become dense (coupled to every entity that touches
//! them). Sparse Cholesky's fill-in grows accordingly and solve
//! times suffer. Use root `Param`s only when the quantity is
//! genuinely system-wide. Two canonical examples:
//!
//! - Frame corrections -- rigid translation + rotation applied
//!   to *every* pose (the `loc_global_demo` pattern).
//! - Global calibration -- one-per-sensor quantities referenced
//!   by every observation from that sensor: camera intrinsics
//!   (`fx`, `fy`, `cx`, `cy`), lens-distortion coefficients, IMU
//!   bias and scale factors, magnetometer declination, barometric
//!   altitude reference. These live on the root naturally because
//!   there's one of them for the whole problem and every
//!   measurement reads them.
//!
//! Prefer per-entity params whenever the quantity is local. A root
//! `Param` referenced by 1% of constraints is fine; one referenced
//! by 90% of them will dominate factorisation cost.
//!
//! ```ignore
//! // Multi-CrossBlock: explicit Hessian pair per unordered entity pair.
//! #[arael::model]
//! #[arael(constraint([hb_ab, hb_ac, hb_bc], { /* 3-line residual */ }))]
//! struct SymmetryLL {
//!     #[arael(ref = root.lines)] a: Ref<Line>,
//!     #[arael(ref = root.lines)] b: Ref<Line>,
//!     #[arael(ref = root.lines)] c: Ref<Line>,
//!     #[arael(cross = (a, b))] hb_ab: CrossBlock<Line, Line>,
//!     #[arael(cross = (a, c))] hb_ac: CrossBlock<Line, Line>,
//!     #[arael(cross = (b, c))] hb_bc: CrossBlock<Line, Line>,
//! }
//!
//! // The coo list: the residual reads the root's params, so the
//! // (pose, root) pairs go to the solve's list; no block field for them.
//! #[arael(constraint([hb_pose, coo], { /* residual touching pose + root */ }))]
//! struct Pose { /* ... hb_pose: SelfBlock<Pose, f32> ... */ }
//! ```
//!
//! ## Collection types
//!
//! | Type | Use it when |
//! |---|---|
//! | [`refs::Vec<T>`](refs::Vec) | dense indexed list, contiguous storage, stable `Ref<T>` handles |
//! | [`refs::Deque<T>`](refs::Deque) | like `Vec` but supports `push_front` / `push_back` (rolling pose history) |
//! | [`refs::Arena<T>`](refs::Arena) | arbitrary insertion / deletion with stable handles |
//! | [`Ref<T>`](refs::Ref) | a handle into the containing collection |
//!
//! A Model struct is "directly composed" when a child Model appears
//! as a plain field (e.g. `sub: Sub` in `single_root_demo.rs`).
//! It's "collection-composed" when wrapped in one of the containers
//! above.
//!
//! ```ignore
//! #[arael::model]
//! #[arael(root)]
//! struct Path {
//!     // collection-composed: many Pose instances, iterated by the macro
//!     poses: refs::Deque<Pose>,
//!     // direct composition: a single Sub entity as a plain field
//!     globals: Globals,
//!     hb: SelfBlock<Path, f64>,
//! }
//! ```
//!
//! ## Struct-level macro attributes
//!
//! | Attribute | Purpose |
//! |---|---|
//! | `#[arael::model]` | declare a Model; generates Model trait impl |
//! | `#[arael(root)]` | mark the top-level Model. Generates `LmProblem` impl, manages indices, owns the update cycle |
//! | `#[arael(root, f32)]` | scalar precision for the generated solver surface (default is f64) |
//! | `#[arael(root, jacobian)]` | additionally emit `calc_jacobian` and `calc_cost_table` for diagnostics |
//! | `#[arael(root, extended)]` | the root also implements [`ExtendedModel`](model::ExtendedModel) by hand, for residuals the macro cannot express (parsed at runtime, say); the generated sweeps call its `extended_update` / `extended_cost` / `extended_compute` |
//! | `#[arael(root, fast_atan)]` | generated code calls [`utils::fast_atan`] / [`utils::fast_atan2`] instead of libm atan / atan2 everywhere; derivatives are the exact forms either way |
//! | `#[arael(root, cost_plain)]` / `cost_kahan` / `cost_f64` | how the cost is summed: plain adds (the default), Kahan summation for more precision at a slight computational cost, or an f64 sum on an f32 root; the last two combine (docs/MODEL.md, "The cost sum") |
//! | `#[arael(root, grad_plain)]` / `grad_kahan` / `grad_f64` | the same three choices for the gradient's sums in the block stores (docs/MODEL.md, "The gradient sum") |
//! | `#[arael(root, wide = <type>)]` | the scalar the constraint bodies' `with_wide(x)` scopes compute in: `f64` when absent, or [`df32`], two f32s with 48 bits of significand, for a target without f64. A `MixedEulerAngleParam<T, H>` in the model must have the same `H` |
//! | `#[arael(root, seq)]` | keep the cost and assembly sweeps on one thread whatever `num_threads` says; the linear solve still threads. For a root that is not `Sync` (a `Cell`, `RefCell` or `Rc` field) or one that should stay sequential |
//! | `#[arael(root, marginalize(field, ...))]` | landmark-style fields (small blocks coupled to poses but never to each other) the sparse solver eliminates first |
//! | `#[arael(fit(coll, \|e\| body))]` | shorthand: sum-of-squares fit of a residual body over one collection (optional trailing `loss = \|s\| rho(s)` for a robust M-estimator) |
//! | `#[arael(skip_self_block)]` | opt out of the mandatory `SelfBlock<Self>` (rare) |
//!
//! Constraints can also appear on the root itself -- useful for
//! regularising root-level parameters.
//!
//! ```ignore
//! // Root-level constraint pinning global_delta near its initial value.
//! #[arael::model]
//! #[arael(root, f32, jacobian)]
//! #[arael(constraint(hb, name = "global_delta_drift", {
//!     let d = path.global_delta - path.global_delta_value;
//!     [d.x * path.drift_pos_isigma,
//!      d.y * path.drift_pos_isigma,
//!      d.z * path.drift_pos_isigma]
//! }))]
//! struct Path {
//!     // ...
//!     global_delta: Param<vect3f>,
//!     drift_pos_isigma: f32,
//!     hb: SelfBlock<Path, f32>,
//! }
//! ```
//!
//! ## Constraint attributes
//!
//! Block-spec forms:
//!
//! ```ignore
//! #[arael(constraint(hb, { body }))]                      // single local block
//! #[arael(constraint([hb_ab, hb_ac, hb_bc], { body }))]   // bracketed multi-block (N >= 2)
//! #[arael(constraint(pose.hb_pose, { body }))]            // remote SelfBlock via Ref field
//! #[arael(constraint([hb_pose, coo], { body }))]          // self-primary + coo pairs with the root or the parent
//! #[arael(constraint(coo, { body }))]                     // every across-entity pair of the refs to the coo list
//! ```
//!
//! The positional form carries a single block only; every N >= 2 block
//! list must use the bracketed form. Writing
//! `constraint(hb_a, hb_b, { body })` is rejected at macro
//! expansion -- use `constraint([hb_a, hb_b], { body })`.
//!
//! Dotted names mean two different things depending on the first
//! segment:
//!
//! - `<ref_field>.<block>` -- reach through a `Ref<T>` field on
//!   this struct to the target entity's `SelfBlock`. PointFrine uses
//!   this to write grad / diagonal into the referenced Pose.
//! - `coo` -- the solve's COO list. Alone, every across-entity
//!   pair of the constraint's refs goes there; second behind the
//!   entity's own `SelfBlock`, the pairs between the entity's params
//!   and the root's or the containing parent's, whichever the body
//!   reads.
//!
//! ```ignore
//! // Remote SelfBlock: PointFrine lives on PointLandmark but writes
//! // pose's diagonal via pose.hb_pose; the (pose, path) cross-pair
//! // goes into a local CrossBlock<Pose, Path>.
//! #[arael::model]
//! #[arael(constraint([pose.hb_pose, hb_root], parent = lm, {
//!     /* residual involving lm, pose, feature, path */
//! }))]
//! struct PointFrine {
//!     #[arael(ref = root.poses)]         pose:    Ref<Pose>,
//!     #[arael(ref = pose.info.features)] feature: Ref<PointFeature>,
//!     hb_root: CrossBlock<Pose, Path, f32>,
//! }
//!
//! // Self-primary + coo: tilt on Pose reads path.global_rot, so the
//! // (pose, path) pairs go to the solve's COO list.
//! #[arael(constraint([hb_pose, coo], {
//!     let mr_global = path.global_rot.rotation_matrix();
//!     let mr2w_eff  = mr_global * pose.ea.rotation_matrix();
//!     let ea_eff    = mr2w_eff.get_euler_angles();
//!     [(ea_eff.x - pose.info.tilt_roll)  * path.tilt_isigma,
//!      (ea_eff.y - pose.info.tilt_pitch) * path.tilt_isigma]
//! }))]
//! struct Pose { /* ... hb_pose: SelfBlock<Pose, f32> ... */ }
//! ```
//!
//! Modifiers (comma-separated before the body):
//!
//! | Modifier | Purpose |
//! |---|---|
//! | `parent = <name>` | bind the parent iteration variable to `<name>` (default is `a_type.to_lowercase()`) |
//! | `name = "label"` | label the residual group. Shows up in `calc_cost_table` and `JacobianRow::label` |
//! | `guard = <bool expr>` | evaluated once per iteration; when false the whole constraint is skipped for that iteration |
//! | `<var>: <Type>` | declare an extra typed binding for the body |
//!
//! ```ignore
//! // `parent = lm` so the body can refer to the enclosing PointLandmark
//! // as `lm`; `name = "feature_obs"` labels the residual group; `guard`
//! // skips the whole constraint when the flag is false.
//! #[arael(constraint([pose.hb_pose, hb_root],
//!     parent = lm,
//!     name = "feature_obs",
//!     guard = feature.enabled,
//!     {
//!         /* residual using lm.pos, pose.*, feature.*, path.* */
//!     }
//! ))]
//! struct PointFrine { /* ... */ }
//! ```
//!
//! Placement decides iteration:
//!
//! | Attribute on | What iterates | Typical use |
//! |---|---|---|
//! | Entity struct (`Pose`) | root iterates the containing collection | per-entity drift / tilt |
//! | Dedicated constraint struct (`PointFrine`) with `Ref<T>` fields | root iterates (often nested) collection of constraint structs | observation linking two or more entities |
//! | Root struct | fires once per solve | regularise root-level params, fix global DOF |
//!
//! ## Field-level macro attributes
//!
//! | Attribute | Applies to | Purpose |
//! |---|---|---|
//! | `#[arael(ref = <path>)]` | `Ref<T>` field | where to resolve the Ref. `root.<collection>` or `<other_ref>.<sub_collection>` |
//! | `#[arael(cross = (<refA>, <refB>))]` | `CrossBlock<T, T>` field | disambiguate which ref pair a same-typed CrossBlock serves |
//! | `#[arael(constraint_index)]` | `u32` field | receives a unique row id per constraint instance; useful for diagnostics |
//! | `#[arael(skip)]` | any field | exclude from the Model's serialize / accumulate paths (rare) |
//!
//! ```ignore
//! #[arael::model]
//! struct PointFeature {
//!     pixel: vect2f,
//!     // The camera is a Ref<cameraf>; we don't want the macro to walk
//!     // it as a nested Model, so skip it.
//!     #[arael(skip)] camera: Ref<cameraf>,
//!     // ... measurement data ...
//! }
//!
//! #[arael::model]
//! #[arael(constraint(hb, { /* ... */ }))]
//! struct PosePair {
//!     #[arael(ref = root.poses)] prev: Ref<Pose>,
//!     #[arael(ref = root.poses)] cur:  Ref<Pose>,
//!     // constraint_index: the macro writes the per-constraint row id
//!     // here so you can correlate log entries to this specific pair.
//!     #[arael(constraint_index)] ci: u32,
//!     hb: CrossBlock<Pose, Pose, f32>,
//! }
//! ```
//!
//! ## User-defined functions (`#[arael::function]`)
//!
//! Constraint bodies have a fixed set of built-in ops (arithmetic,
//! `sin` / `cos` / `exp` / `sqrt` / `clamp` / `safe_asin` / ...,
//! vector helpers). When your residual needs a custom function --
//! a factored-out symbolic helper, or an opaque numerical routine
//! with a known closed-form derivative -- declare it with
//! `#[arael::function]` and use it in constraint bodies the same
//! way you'd use `sin`.
//!
//! Three forms, distinguished by the attributed fn's signature.
//!
//! ### Form A: purely symbolic
//!
//! `fn name(x: E, ...) -> E { expr }` -- the body is an arael-sym
//! expression. The macro captures the body as an arael-sym source
//! string, re-parses it at constraint-expansion time, and inlines
//! the resulting `E` tree into the surrounding residual.
//! Derivatives come from arael-sym's own auto-diff.
//!
//! ```ignore
//! use arael::sym::E;
//!
//! #[arael::function]
//! fn sigmoid(x: E) -> E {
//!     1.0 / (1.0 + exp(-x))
//! }
//!
//! #[arael::function]
//! fn square(x: E) -> E { x * x }
//!
//! #[arael::model]
//! #[arael(root, jacobian)]
//! #[arael(constraint(hb, name = "fit", {
//!     [(sigmoid(m.x) - m.target) * m.isigma,
//!      (square(m.y) - 9.0) * m.isigma]
//! }))]
//! struct M {
//!     x: Param<f64>,
//!     y: Param<f64>,
//!     target: f64,
//!     isigma: f64,
//!     hb: SelfBlock<M>,
//! }
//! ```
//!
//! The body is stringified and handed to
//! `arael_sym::parse_with_functions`, so identifiers resolve
//! against arael-sym's parser rather than Rust's name resolution.
//! `E` is `arael::sym::E`; the crate needs no dependency on
//! arael-sym itself. A `match` in the body is rewritten to
//! `select` / `select_or` before the parse, under the constraint-body
//! rules: literal patterns `0, 1, ...` in order, an optional trailing
//! `_`, no guards, one expression per arm.
//!
//! Optional `derivs = [expr, ...]` overrides auto-diff with an
//! explicit partial per parameter. Expressions are raw tokens,
//! not strings or closures.
//!
//! ### Form B: opaque numerical eval
//!
//! `#[arael::function(sym_name[, derivs = [...]])]` on a
//! `fn name_eval(...)` over `f32` or `f64`. The eval fn is opaque
//! numerical code the macro never inspects. The positional
//! `sym_name` names the symbolic sibling the macro emits for use
//! inside constraints. The signature says where the derivative
//! comes from:
//!
//! | shape | signature | derivative |
//! |---|---|---|
//! | without one | `fn f_eval([root: &R,] x: f64, ...) -> f64`, no `derivs` | none |
//! | symbolic | the same, with `derivs = [...]` | the expressions |
//! | numeric | `fn f_eval([root: &R,] x: f64, ..., derivs: bool) -> (f64, [f64; N])`, no `derivs` | from the call |
//!
//! The root. The eval fn may take the model root as its first
//! parameter, a shared reference to the root type, or to a generic
//! parameter with the fn's own bound. Every call passes it
//! explicitly, as `root` or the root's own name: `gain(root, m.x)`
//! in a body, `slope(root, x)` in derivs. A root-taking function
//! works in constraint bodies, Form C bodies and derivs; not in a
//! `symbolic =` field, where no root exists.
//!
//! Symbolic. `derivs` has one expression per scalar parameter,
//! same token shape as Form A derivs, over the eval fn's own
//! parameter names in declaration order, the root's fields as
//! `root.<field>`, and other registered functions, root-taking ones
//! included; mutual recursion is resolved by a two-pass bag build
//! at constraint-expansion time.
//!
//! Numeric. The fn returns its value and its first partial
//! derivative by each scalar parameter from the same call: `.0` the
//! value, `.1[i]` the partial derivative by the i-th scalar
//! parameter. The trailing `derivs:
//! bool` says whether the partials will be read: the cost sweep
//! passes false, the Jacobian assembly true, and the array is
//! ignored when false. One call serves the residual and its
//! Jacobian. Only first derivatives exist, by design: a numeric
//! partial has no derivative of its own.
//!
//! Without one. A scalar-returning fn with no `derivs`. It
//! serves where no derivative of it is needed: in another
//! function's derivs, or applied to data. Applied to a parameter
//! in a residual, the Jacobian needs the derivative it does not
//! have, and that is a compile error naming the function.
//!
//! ```ignore
//! // A gain table on the root: its value, the slope it also
//! // gives, and a lookup returning both from one search.
//! #[arael::function(slope)]
//! fn slope_eval(root: &Scene, x: f64) -> f64 { root.table.slope(x) }
//!
//! #[arael::function(gain, derivs = [slope(root, x)])]
//! fn gain_eval(root: &Scene, x: f64) -> f64 { root.table.value(x) }
//!
//! #[arael::function(lookup)]
//! fn lookup_eval(root: &Scene, x: f64, derivs: bool) -> (f64, [f64; 1]) {
//!     let (v, dv) = root.table.at(x);
//!     (v, [if derivs { dv } else { 0.0 }])
//! }
//!
//! // `my_safe_asin` clamps its input before calling the libm
//! // asin and supplies a closed-form derivative that stays
//! // finite at the clamp edge. The `identity(...)` guard blocks
//! // the simplifier from reordering `1 - x*x + 1e-12` into
//! // `1 + 1e-12 - x*x` -- at |x| ~ 1 the subtraction already
//! // cancels most significant bits and the reordered form loses
//! // the 1e-12 floor. Same pattern as arael-sym's built-in
//! // `safe_asin`.
//! #[arael::function(my_safe_asin,
//!     derivs = [1.0 / sqrt(identity(1.0 - x * x) + 1e-12)])]
//! fn my_safe_asin_eval(x: f64) -> f64 {
//!     x.clamp(-1.0, 1.0).asin()
//! }
//!
//! #[arael::model]
//! #[arael(root, jacobian)]
//! #[arael(constraint(hb, name = "fit", {
//!     [(gain(root, scene.x) - scene.target) * scene.isigma,
//!      lookup(root, scene.x) - scene.target2,
//!      my_safe_asin(scene.y) - 0.5]
//! }))]
//! struct Scene {
//!     table: Table,
//!     x: Param<f64>,
//!     y: Param<f64>,
//!     target: f64,
//!     target2: f64,
//!     isigma: f64,
//!     hb: SelfBlock<Scene>,
//! }
//! ```
//!
//! The sibling takes the root as an `E` too, `gain(root, x)`, and
//! builds the expression for ordinary Rust. A root-taking function
//! cannot be evaluated there, since there is no root at runtime:
//! the scalar shapes panic, the numeric one returns an error.
//!
//! ### Form C: typed, with `let` bindings and tuples
//!
//! `fn name(p: vect3sym, k: E, ...) -> (E, E) { let ...; expr }` --
//! the parameters and the result are the values a constraint body
//! works with: `E`, `vect2sym`, `vect3sym`, `matrix2sym`,
//! `matrix3sym`, `quaternsym`, a tuple of them, or an `[E; N]`
//! array. The body is a block under the constraint-body grammar:
//! `let` bindings (a name, `_`, or a tuple pattern over a tuple
//! value), vector and matrix arithmetic, component access,
//! `match`, `branch`, calls of builtins and of other functions. A
//! call inlines the body at the call site with the arguments bound
//! to the parameters, so the generated code is what the body would
//! have produced written out, derivatives included. The body sees
//! only its parameters and its `let`s.
//!
//! A tuple is destructured, or it is the residual: `let (u, v) =
//! project(...);` in a body, or a call returning `(E, E)` or
//! `[E; 2]` as the body's final expression, whose elements become
//! the rows. A tuple is not a value to compute with.
//!
//! ```ignore
//! use arael::sym::{vect2sym, vect3sym, E};
//!
//! /// Pinhole projection of a camera-frame point, zero below
//! /// machine-epsilon depth.
//! #[arael::function]
//! fn project(p: vect3sym, f: E, c: vect2sym) -> (E, E) {
//!     let ok = p.z - epsilon_for(p.z);
//!     (branch(ok, f * (p.x / p.z) + c.x, 0.0),
//!      branch(ok, f * (p.y / p.z) + c.y, 0.0))
//! }
//!
//! #[arael::function]
//! fn pixel(p: vect3sym, f: E, c: vect2sym, obs: vect2sym) -> [E; 2] {
//!     let (u, v) = project(p, f, c);
//!     [u - obs.x, v - obs.y]
//! }
//! ```
//!
//! An all-`E` signature with a single-expression body is Form A;
//! the same signature with `let` bindings is Form C. A typed
//! function is declared before the constraint that calls it, in
//! the same crate, and it may not call itself. The fn is also
//! emitted as ordinary Rust over the arael-sym types, so user code
//! can call it to build expressions; the signature's type names
//! must be in scope there.
//!
//! ### Ergonomics
//!
//! - Parameter names in deriv expressions resolve to the
//!   attributed fn's own parameters, not to anything in the
//!   surrounding module.
//! - Numeric literals accept scientific notation (`1e-12`,
//!   `2.5E+2`).
//! - The sibling fn (Form A body, Form B positional name, Form C
//!   as written) is also callable from ordinary Rust with `E` and
//!   sym vector arguments, so user fns compose with
//!   [`model::ExtendedModel`] / runtime `parse_with_functions`
//!   workflows for residuals that aren't known at compile time.
//!   Mutually-referencing Form A and B fns (and forward references
//!   to fns declared later in the file or in a dependency) work at
//!   runtime via a registry populated through `inventory`;
//!   cross-crate composition works without re-declaration. Form C
//!   is inlined at expansion and has no runtime registry: same
//!   crate, declared first.
//! - Errors point at user source: bad signatures, mismatched
//!   deriv counts, and name collisions fire at attribute
//!   expansion; parse failures and arity mismatches fire at the
//!   call site inside the constraint body.
//!
//! See
//! [`examples/user_function_demo.rs`](https://github.com/harakas/arael/blob/master/examples/user_function_demo.rs)
//! for a runnable two-form demo.
//!
//! ## Runtime differentiation (`ExtendedModel`)
//!
//! For Models whose residuals aren't known at compile time (e.g. a
//! user-supplied expression parsed at runtime), implement
//! [`model::ExtendedModel`] alongside [`model::Model`]. The macro does
//! not generate the residual evaluation; you write:
//!
//! ```ignore
//! fn extended_update(&mut self, params: &[f64]);
//! fn extended_compute(&mut self, params: &[f64], grad: &mut [f64], coo: &mut Coo<f64>);
//! ```
//!
//! `extended_compute` evaluates residuals, writes into the solver's
//! `grad` slice and pushes Hessian entries into `coo`, the solve's own
//! list; `Coo::add_residual` does both for one residual. The entries
//! it pushes must come in the same number and order on every call
//! within one solve: the sparse solvers bind the pattern once and
//! replay it by position. Before the first assembly the solve calls
//! the hook once on its own to see whether it pushes anything. See
//! [`examples/runtime_fit_demo.rs`](https://github.com/harakas/arael/blob/master/examples/runtime_fit_demo.rs)
//! for the full pattern: runtime parse + compile-time differentiation
//! of the parsed expression.
//!
//! # Cross-Crate Models
//!
//! A crate can share its models. After all `#[arael::model]`
//! definitions (macro expansion is top-down -- the bottom of lib.rs is
//! the natural place), emit the crate's import macro:
//!
//! ```ignore
//! arael::export_models!();
//! ```
//!
//! Every `pub` model struct and enum defined above the invocation joins
//! the bundle. An importing crate registers all of them in one line,
//! before defining its own models over them:
//!
//! ```ignore
//! use model_crate::{Pose, Frine};
//! model_crate::arael_import!();
//! ```
//!
//! After that the imported types work like local ones: component
//! fields, `Ref<Pose>` on local constraint structs,
//! `CrossBlock<Pose, Local>`, local roots over imported entities, at
//! either precision. Importing the same bundle twice (diamond
//! dependencies) is harmless, and a model crate that imports another
//! and calls `export_models!()` re-exports what it imported.
//!
//! Rules:
//!
//! - Exported structs need `pub` fields -- generated code in the
//!   importing crate reads them directly. A `pub` struct with a non-pub
//!   field still compiles but is excluded from the bundle; an importer
//!   that reaches for it gets an error naming the field.
//!   `#[arael(skip)]` fields may stay private.
//! - Roots and `fit(...)` structs are not importable: their generated
//!   solvers are already ordinary pub API.
//! - An imported constraint struct keeps its `root.<field>` resolution
//!   paths: the importing root must name its collections as the model
//!   crate's constraints expect.
//! - The bundle records each struct's param count; the importer
//!   recomputes it from the same tokens and fails the build on mismatch
//!   (incompatible arael-macros versions between the two crates).
//!
//! # C++, Python and JavaScript Interfaces
//!
//! A crate holding a root model exports to other languages with one
//! command: `cargo install cargo-arael`, then `cargo arael export` in
//! the model crate. This generates a C ABI shim crate (`capi/`, cdylib +
//! staticlib), C++ wrapper classes with vendored math headers
//! (`cxx/`, with CMake glue), a pure-`ctypes` Python package
//! (`python/`, stdlib only -- one cdylib serves every CPython 3.x), and
//! a wasm-bindgen crate for the browser (`wasm/`); `interfaces =
//! ["cxx"]` under `[package.metadata.arael]` in the crate's `Cargo.toml`
//! writes only the trees listed.
//!
//! The C++ and Python interfaces cover composing the problem, the
//! solvers (dense, sparse, band, the GPU), robust losses, configs with
//! the Rust preset values, timing, solve reports and the covariance
//! queries; docs/WASM.md lists what the JavaScript one leaves out.
//!
//! ```cpp
//! Fit fit;                                  // C++
//! auto p = fit.poses().push_back();
//! p.set_pos({0, 0, 0});
//! auto r = fit.solve_sparse(cfg);           // result<LmResult, SolveError>
//! ```
//!
//! ```python
//! f = fit.Fit()                             # Python
//! p = f.poses.push_back()
//! p.pos = (0, 0, 0)
//! r = f.solve_sparse(cfg)                   # raises AraelError on failure
//! ```
//!
//! See `docs/CXX.md`, `docs/PYTHON.md` and `docs/WASM.md`;
//! [`cxx-examples/`](https://github.com/harakas/arael/tree/master/cxx-examples)
//! carries five demos with C++ and Python drivers over shared models,
//! `bal_demo` among them on the CPU or a GPU.
//!
//! A page built on the JavaScript skin runs at
//! [arael.mare.ee/pgo](https://arael.mare.ee/pgo/): a pose graph solved
//! in the browser, with poses to drag, lock and fix and the graph
//! re-solving around them. Its source is
//! [cxx-examples/m3500_demo](https://github.com/harakas/arael/tree/master/cxx-examples/m3500_demo).
//!
//! # Solvers
//!
//! arael's Levenberg-Marquardt (LM) solver has several linear-algebra
//! backends behind one trait ([`simple_lm::LmSolver`]) and one config
//! struct ([`simple_lm::LmConfig`]). The full reference is
//! [docs/SOLVERS.md](https://github.com/harakas/arael/blob/master/docs/SOLVERS.md).
//!
//! ## Entry points and backends
//!
//! The main entry point is [`solve_with`](simple_lm::LmProblem::solve_with),
//! which every `#[arael(root)]` model has: it reads the parameters from
//! the model, solves with any backend instance and writes them back.
//! `solve_sparse` and `solve_dense` are shorthands for it,
//! `solve(SolverKind, &cfg)` picks the backend at run time, and
//! `lm_solve` runs the same solve over a parameter vector you manage
//! yourself:
//!
//! ```rust,ignore
//! use arael::simple_lm::LmProblem; // or `use arael::prelude::*;`
//! let result = model.solve_with(&mut Band::new(11), &cfg)?; // any LmSolver backend
//! let result = model.solve_sparse(&cfg)?; // = solve_with(SparseFaer): the default backend
//! let result = model.solve_dense(&cfg)?;  // = solve_with(Dense)
//! let result = model.solve(SolverKind::Dense, &cfg)?; // a backend named at run time
//! let result = lm_solve(&x0, &mut SparseFaer::new(), &mut problem, &cfg)?; // the raw form
//! ```
//!
//! Every solve returns [`SolveResult`](simple_lm::SolveResult) --
//! `Ok(LmResult)` when the solve terminated by a stopping rule (including
//! `MaxIterations` and `TimeLimit`), `Err(`[`SolveFailure`](simple_lm::SolveFailure)`)`
//! when the system could not be built or factored or a Hessian diagonal
//! went bad, with the best accepted state carried in the error when one
//! exists.
//!
//! The generated methods match the root's precision: on an
//! `#[arael(root, f32)]` model they take `f32` configs and `solve_sparse`
//! uses [`SparseFaer<f32>`](simple_lm::SparseFaer).
//!
//! | Backend (`solve_with(&mut ..., &cfg)`) | What it is |
//! |---|---|
//! | [`SparseFaer`](simple_lm::SparseFaer)`::<T>::new()` (`T` = `f64`/`f32`) | the default (= `solve_sparse`): sparse Cholesky, pure Rust, in block form. Marginalizes the model's landmark-like blocks (a Schur complement) when that is faster than factorizing the whole system, and decides which by itself; [`SchurPolicy`](simple_lm::SchurPolicy), [`SolveOrdering`](simple_lm::SolveOrdering) and [`BlockSupernodalMode`](simple_lm::BlockSupernodalMode) override it. The only backend that takes mixed precision and a GPU |
//! | [`Dense`](simple_lm::Dense) | dense nalgebra Cholesky (= `solve_dense`): few parameters, or dense problems |
//! | [`Band`](simple_lm::Band)`::new(kd)` | pure-Rust band Cholesky for banded Hessians (localization); an element outside the band is an error |
//! | `BandLapack::new(kd)` | the same band solve through LAPACK `dpbsv`/`spbsv` (feature `lapack`) |
//! | `SparseEigen::<T>::new()` | Eigen `SimplicialLLT` (feature `eigen`) |
//! | `SparseCholmod::new()` | CHOLMOD simplicial Cholesky, LGPL (feature `cholmod`; f64 only) |
//! | `SparseCholmodSupernodal::new()` | CHOLMOD supernodal Cholesky, a GPL-licensed module (feature `cholmod-gpl`; f64 only) |
//!
//! ## Damping-schedule drivers
//!
//! Levenberg-Marquardt's damping parameter, lambda, moves each step between
//! Gauss-Newton (small lambda) and gradient descent (large lambda). Every
//! iteration is a full linearization, factorization and solve, so the
//! iteration count is close to the run time, and the damping schedule and
//! tolerances that suit a problem are worth finding.
//!
//! The lambda schedule is pluggable: the LM loop consults a
//! `LambdaDriver` for every damping decision, feeding it each attempted
//! step's outcome (costs, gradient, Hessian diagonal, attempted step).
//! `DefaultLambdaDriver` is the classic fixed-multiplier schedule (the
//! default on every `LmConfig`); `NielsenLambdaDriver` is the gain-ratio
//! adaptive schedule: it scales lambda by how well the
//! quadratic model predicted the actual cost change, escalating
//! geometrically on rejections, which removes the fixed schedule's
//! accept/reject sawtooth on strongly nonlinear problems such as bundle
//! adjustment. The driver lives on the config -- select one with
//! `LmConfig::with_driver(NielsenLambdaDriver::default())`; every solve
//! entry point uses `config.driver`.
//!
//! A driver can also stop the solve: its step hooks return `Option`, and `None`
//! means stop -- for a rule the config cannot express, such as a step-norm test
//! or an external deadline. From `accepted` the step is kept
//! (`LmStatus::DriverTerminated`); from `rejected` or `factorization_failed` the
//! last accepted one comes back (`LmStatus::LambdaCeiling`).
//!
//! To watch a solve without touching the schedule, attach an
//! [`LmObserver`](simple_lm::LmObserver) with `LmConfig::with_observer`
//! (closures qualify): it is called once per damped attempt with the
//! iteration state and the current best parameters, and can stop the
//! solve (`LmStatus::ObserverTerminated`), for progress reporting,
//! cancellation and good-enough checks. With the LM loop on a GPU, an
//! observer gets the parameters and a driver the gradient, the diagonal
//! and the step only when the config asks for them (see
//! [Solving on a GPU](#solving-on-a-gpu)).
//!
//! ## Basic usage
//!
//! ```
//! # use arael::model::{Param, SelfBlock};
//! # #[arael::model]
//! # #[arael(root, f32)]
//! # #[arael(constraint(hb, { [m.x - 3.0] }))]
//! # struct M { x: Param<f32>, hb: SelfBlock<M, f32> }
//! # let mut model = M { x: Param::new(0.0), hb: SelfBlock::new() };
//! use arael::simple_lm::{LmConfig, LmProblem};
//!
//! let cfg = LmConfig::conservative();
//! // solve_sparse reads the params from the model, runs LM, and writes the
//! // optimized values back into it (LmProblem provides the solve methods).
//! let result = model.solve_sparse(&cfg).unwrap();
//! println!("{} iterations: {:.4} -> {:.4}",
//!     result.iterations, result.start_cost, result.end_cost);
//! # assert!((model.x.value - 3.0).abs() < 1e-3);
//! ```
//!
//! The `#[arael(root)]` macro generates the `LmProblem<T>` impl the
//! solver needs. It also generates the
//! [`LmProblemInternals<T>`](simple_lm::LmProblemInternals) half (the
//! context forms of the evaluations, the routes a backend picks, the
//! walks of the structure, the elimination hints), which is inner API:
//! public, and free to change without a major version bump.
//!
//! ## [`LmConfig`](simple_lm::LmConfig) -- every field, with defaults
//!
//! | Field | Default | Meaning |
//! |---|---|---|
//! | `abs_precision` | `1e-6` | a step is small when the cost drops by less than this, or by less than `rel_precision` of the cost |
//! | `rel_precision` | `1e-4` | a relative cost drop below this is small |
//! | `max_iters` | `100` | the most iterations, damping retries included |
//! | `max_accepted_iters` | `None` | stop after this many accepted steps |
//! | `min_iters` | `5` | the stopping rules do not apply before this many accepted steps |
//! | `patience` | `3` | small steps in a row that stop the solve |
//! | `initial_lambda` | `1e-4` | the damping the solve starts with; small is close to Gauss-Newton, large to gradient descent |
//! | `lambda_floor` | `1e-12` (f32: its epsilon) | lambda never drops below this after an accepted step |
//! | `cost_threshold` | `0.0` | stop as soon as the cost is at or below this; `0.0` turns it off |
//! | `gradient_tolerance` | `None` | stop when `max_i |g_i| / sqrt(H_ii)` is at or below this. The only test for a stationary point |
//! | `parameter_tolerance` | `None` | stop when `|step| <= tol * (|x| + tol)`: the parameters stopped moving |
//! | `predicted_reduction_tolerance` | `None` | stop when the model predicts less than this fraction of the cost as the next gain |
//! | `min_diagonal` | `None` | floor under the damping scale, so a parameter with no curvature does not fail the solve |
//! | `driver` | `DefaultLambdaDriver` | how lambda moves between attempts (a [`LambdaDriver`](simple_lm::LambdaDriver)); `NielsenLambdaDriver` through `with_nielsen()` |
//! | `num_threads` | `1`, or `ARAEL_NUM_THREADS` | threads for the linear solve and the sweeps (needs the `threads` feature) |
//! | `assembly_threads` | `None` | a thread count for the cost and assembly sweeps alone; `None` uses `num_threads` |
//! | `time_limit` | `None` | wall-clock budget for the whole solve; overrides `min_iters` |
//! | `covariance` | `None` | assemble the parameter covariance at the solution, in this mode |
//! | `covariance_options` | `CovOptions::auto()` | how that covariance is assembled |
//! | `verbose` | `false` | log one line per attempt; turn it on first when debugging |
//! | `observer` | `None` | an [`LmObserver`](simple_lm::LmObserver) called once per damped attempt; can stop the solve |
//! | `gather_timing` | `false` | record per-phase timing and the per-attempt timeline in `LmResult::timing` |
//! | `device_observer_params` | `false` | with the LM loop on a GPU, read the parameters back after each accepted step for the observer |
//! | `device_driver_vectors` | `false` | with the LM loop on a GPU, read the gradient, the diagonal and the step back for the lambda driver |
//!
//! The solver stops when `iter >= min_iters` and the last `patience`
//! accepted steps were small; or on any of `iter >= max_iters`,
//! `cost <= cost_threshold`, a tolerance that is met, a spent
//! `time_limit`, or a driver or observer that stops it.
//!
//! The gradient and step tolerances are off unless set, and test what the
//! cost test cannot: the gradient test says the solve is at a stationary
//! point (a solve drifting along a gauge freedom has a flat cost and a live
//! gradient), the step test that the parameters stopped moving.
//!
//! Without `min_diagonal` a parameter of zero curvature fails the solve
//! (`Err` with [`SolveFailureKind::DegenerateDiagonal`](simple_lm::SolveFailureKind)).
//! Damping is `H[i,i] + lambda * max(H[i,i], min_diagonal)`, so the floor
//! keeps the damped system positive definite and the parameter does not
//! move. Such a parameter is unconstrained and its value means nothing:
//! constrain it, hold it fixed (`Param::fixed`), or leave the entity out.
//! Use the floor only where a residual can switch itself off for an
//! iteration (a `branch` guarding an undefined observation, a saturated
//! robust loss). A negative or NaN diagonal still fails the solve.
//!
//! A spent `time_limit` stops the solve wherever it is
//! (`LmStatus::TimeLimit`), returning the last accepted step. It is checked
//! before each assembly and each damped attempt, so one linear solve can
//! overrun it. When it is `None` (the default) the solver never reads the
//! clock.
//!
//! ## Threads
//!
//! Arael is single-threaded by default. With the `threads` feature the
//! cost evaluation, the assembly, the Schur reduction and the
//! factorization run on several threads:
//!
//! ```toml
//! arael = { version = "0.9", features = ["threads"] }
//! ```
//!
//! ```no_run
//! # use arael::simple_lm::LmConfig;
//! // 1 = sequential (default), n = n threads, 0 = every core
//! let cfg = LmConfig::<f64>::conservative().with_num_threads(4);
//! ```
//!
//! Without the feature, anything but 1 warns and stays sequential.
//! `ARAEL_NUM_THREADS` in the environment sets the default, and yields
//! to a count set in code. `assembly_threads` gives the cost and
//! assembly sweeps a count of their own.
//!
//! The factorization's dense kernels run on rayon's global pool; the
//! rest runs on arael's own worker threads, parked between sweeps for the
//! life of the process, until `arael::pool::shutdown()` joins them. A
//! threaded solve matches a sequential one to rounding. The threads chart
//! under [Benchmarks](#benchmarks) shows what the threads buy.
//!
//! The model is read from every thread at once, so a root must be
//! [`Sync`]. A root that is not, or that should stay sequential, opts
//! out and keeps its linear solve threaded:
//!
//! ```ignore
//! #[arael(root, seq)]
//! struct Scene { .. }
//! ```
//!
//! Threading has overhead: whether it helps depends on the model and its
//! number of parameters. Every solve's report says what the threads did.
//! In a solve on a GPU the threads run what stays on the host. See
//! [docs/SOLVERS.md](https://github.com/harakas/arael/blob/master/docs/SOLVERS.md#threads).
//!
//! ## Solving on a GPU
//!
//! [`SparseFaer`](simple_lm::SparseFaer) can run the assembly of the
//! gradient and the Hessian, the damped solves and the LM loop on a GPU.
//! It pays where the factor is large: bundle adjustment from a few
//! hundred cameras up, SLAM maps with landmarks. Pose graphs solve faster
//! on the host. A GPU solve agrees with a host solve to rounding.
//!
//! | Feature | GPU | Needed at run time |
//! |---|---|---|
//! | `gpu-cuda` | NVIDIA | the CUDA driver library and NVRTC |
//! | `gpu-hip` | AMD | the HIP runtime and hipRTC, from ROCm |
//! | `gpu-metal` | Apple silicon | macOS |
//! | `gpu-cpu` | none: the GPU kernels run on the host, for tests | a C++ compiler |
//!
//! The runtimes load their libraries at run time, so a build needs no
//! CUDA toolkit or ROCm, and a program built with a runtime runs on
//! machines without that GPU.
//!
//! ```ignore
//! use arael::simple_lm::{GpuDevice, MixedPrecision, SparseFaer, SparseFaerOptions};
//!
//! let gpu = GpuDevice::detect(); // the machine's GPU; None without one
//! // Metal has no f64: an f64 solve on it factorizes in f32 and refines
//! let mixed = gpu.as_ref().is_some_and(|g| !g.has_f64()).then(MixedPrecision::default);
//! let opts = SparseFaerOptions::default().with_gpu(gpu).with_mixed_precision(mixed);
//! let r = model.solve_with(&mut SparseFaer::from_options(&opts), &cfg)?;
//! if let Some(gpu) = &r.gpu {
//!     println!("gpu: {gpu}"); // what ran on it
//! }
//! ```
//!
//! - The block supernodal Cholesky and conjugate gradients on a formed
//!   reduced system run on the GPU. The other routes (the envelope, faer's
//!   scalar Cholesky, conjugate gradients on a reduced system that is not
//!   formed) run on the host, and a GPU given to them is not used: the solve
//!   warns, and [`GpuReport::work`](simple_lm::GpuReport::work) is `Unused`.
//! - When every constraint of the root has a device form the GPU also
//!   assembles and keeps the LM loop's vectors
//!   ([`GpuWork::Loop`](simple_lm::GpuWork::Loop)); otherwise the host
//!   assembles. An observer then gets the parameters, and a lambda driver
//!   the gradient, the diagonal and the step, only with
//!   `device_observer_params` and `device_driver_vectors`.
//! - [`GpuDevice::detect`](simple_lm::GpuDevice::detect) prefers a
//!   discrete GPU to an integrated one, then CUDA, HIP, Metal. Clones of a
//!   [`GpuDevice`](simple_lm::GpuDevice) share its GPU memory, so repeated
//!   solves allocate nothing.
//! - [`LmResult::gpu`](simple_lm::LmResult::gpu) says what ran on the GPU,
//!   and [`LmTiming::gpu`](simple_lm::LmTiming::gpu) the GPU's own time.
//!
//! See
//! [docs/SOLVERS.md](https://github.com/harakas/arael/blob/master/docs/SOLVERS.md#solving-on-a-gpu)
//! for precision, conjugate gradients on a GPU, the kernel cache and the
//! tools for reading and checking the GPU's kernels.
//!
//! ## Tuning for performance vs quality
//!
//! Every iteration costs an assembly, a factorization and a cost
//! evaluation, and the last iterations of a solve improve the cost
//! little. Tuning is choosing where to stop: measure on representative
//! inputs.
//!
//! - `verbose: true` first. The trace of a few real solves shows how
//!   many iterations the solve takes, whether it still improves near the
//!   end, and whether steps are rejected.
//! - `max_iters`: lower it to a margin above where the trace shows the
//!   cost stops improving.
//! - `rel_precision` and `abs_precision`: loosening them removes
//!   iterations from the end of a solve; tightening them buys precision.
//! - `patience`: a higher value is more careful about stopping.
//! - `initial_lambda`: small (`1e-6`) for a warm start near the
//!   minimum, larger (`1e-2` to `1e-1`) for a cold start far from it.
//!   If the first iterations are rejected, lambda was too small; if the
//!   cost barely moves in them, it was too large.
//! - `cost_threshold`: only with a known target cost.
//!
//! See
//! [docs/SOLVERS.md](https://github.com/harakas/arael/blob/master/docs/SOLVERS.md#tuning-for-performance-vs-quality)
//! for a procedure.
//!
//! ## The algorithm in brief
//!
//! On each iteration the solver:
//!
//! 1. linearises the residuals at the current `x`: `r`, `J`;
//! 2. damps: assembles `J^T J + lambda * diag(J^T J)` and `J^T r`;
//! 3. solves the damped system with the chosen backend. The Cholesky
//!    factorization fails when the matrix is not positive definite;
//! 4. accepts or rejects the step by comparing the new cost to the old:
//!    on a lower cost it accepts and divides lambda by 5, otherwise it
//!    rejects, multiplies lambda by 10 and tries again;
//! 5. repeats until a termination rule applies.
//!
//! Lambda acts as a trust-region radius: a large lambda gives small,
//! safe steps, a small one large Gauss-Newton steps.
//!
//! ## Verbose trace format
//!
//! With `verbose` each attempt logs one line (at info level, see
//! [`log`]; the default destination, stderr, prefixes it with
//! `[arael INFO] `):
//!
//! ```text
//! 3/0: 44.5679->44.5403 / 0.0276, lambda=2e-5 (step=91)
//! ```
//!
//! | Field | Meaning |
//! |---|---|
//! | `3/0` | the iteration and the retry counter; `0` is the first attempt at this linearization |
//! | `44.5679->44.5403` | the cost before and after the step |
//! | `0.0276` | the cost improvement |
//! | `lambda=2e-5` | the damping of this attempt |
//! | `(step=91)` | microseconds of this attempt |
//! | `(step=2194, factor=229, solve=16)` | with the block supernodal Cholesky: its factorization and its solves |
//! | `(step=899, gpu=889)` | on a GPU: the part of the attempt spent on the GPU's damped solve, timed on the host |
//!
//! A failed factorization logs a longer line:
//!
//! ```text
//! 5/0: Cholesky failed (damped matrix not positive-definite), lambda=1.6e-7 -> 1.6e-6 (step=177) [non-finite: grad=0 diag=0 x=0 matrix=0]
//! ```
//!
//! | Field | Meaning |
//! |---|---|
//! | `lambda=1.6e-7 -> 1.6e-6` | the damping of the failed attempt and of the next one |
//! | `non-finite: grad=N diag=N x=N matrix=N` | counts of NaN and infinite values in the gradient, the diagonal, the parameters and the matrix |
//!
//! With all counts 0 the matrix is finite and not positive definite at
//! this damping; the next attempt damps more. A non-zero count means the
//! model produces non-finite values: fix the model.
//!
//! ## Graduated optimisation
//!
//! When the residual groups of a model have very different weights
//! (inverse sigmas), start with the tight constraints loose and tighten
//! them over several solves. The cost surface is smoother at low
//! weights, so from a noisy initial estimate this converges more
//! reliably than solving the tight problem at once.
//!
//! Idiom: a scale field on the root, multiplied into the stiff
//! residuals, stepped per pass:
//!
//! ```ignore
//! struct Path { /* ... */ frine_isigma_scale: f32 }
//!
//! // in the constraint body:
//! let plain1 = ... * feature.isigma.x * path.frine_isigma_scale;
//!
//! // main loop:
//! for scale in [0.01, 0.1, 1.0] {
//!     path.frine_isigma_scale = scale;
//!     let result = path.solve_sparse(&cfg)?;
//! }
//! ```
//!
//! See `examples/loc_demo.rs` and `examples/slam_demo.rs`. Keep the
//! scale on a single root field and set it to the desired absolute
//! value each pass -- the constraint body reads the current value, so
//! the sigma you see is always the sigma in effect.
//!
//! # Parameter Covariance
//!
//! Recover the covariance of the estimated parameters, `Sigma = 2 H^-1`, from
//! the same Hessian the solver builds -- the dense inverse is never formed.
//! Ask for it in the config ([`LmConfig::with_covariance`](simple_lm::LmConfig::with_covariance))
//! and read it off the result ([`LmResult::covariance`](simple_lm::LmResult::covariance)),
//! an owned [`CovAssembly`](covariance::CovAssembly) to query.
//!
//! ```rust,ignore
//! use arael::covariance::{CovMode, Covariance};
//! let cfg = LmConfig::well_conditioned().with_covariance(CovMode::AllMarginals);
//! let r   = model.solve_sparse(&cfg)?;
//! let cov = r.covariance?;                          // the assembly the solve built
//! let sd  = cov.std_dev(&model.poses[0])?;          // 1-sigma per scalar (tangent coords)
//! let s   = cov.marginal_cov(&model.landmarks[3])?; // full covariance block
//! let p   = cov.marginal_cov(&model.poses[0].r2w)?; // one parameter field alone
//! let x   = cov.cross_cov(&model.poses[0], &model.landmarks[3]);
//!
//! let cov = model.assemble_covariance(CovMode::AllMarginals)?;   // without a solve
//! ```
//!
//! Query per-entity blocks by passing the entity itself: any [`Model`](model::Model)
//! reports its parameter span, so a pose, a landmark, or a whole collection works.
//! Covariances are in local tangent coordinates (rotation deltas are minimal 3-DOF
//! retractions). [`CovMode`](covariance::CovMode) picks the strategy: `PerQuery`
//! (factor, then solve per query -- a few entities), `AllMarginals` (a selected
//! inverse up front -- many or all marginals), and `TriDiagonal` (a band
//! forward/backward pass for a localization pose chain, no factorization).
//!
//! [`CovOptions`](covariance::CovOptions)
//! ([`LmConfig::with_covariance_options`](simple_lm::LmConfig::with_covariance_options), or
//! [`assemble_covariance_with`](covariance::Covariance::assemble_covariance_with)
//! on the model) chooses the elimination ordering
//! ([`CovOrdering`](covariance::CovOrdering)), and whether the block
//! supernodal Cholesky factorizes.
//! [`CovOrdering::Auto`](covariance::CovOrdering)
//! builds a symbolic factorization per candidate ordering to choose between
//! them; naming the ordering skips that work.
//! [`CovAssembly::plan`](covariance::CovAssembly::plan) reports what an
//! assembly picked.
//!
//! `H` must be non-singular: an unfixed gauge returns
//! [`NotPositiveDefinite`](covariance::CovError) -- anchor a pose or add a prior.
//! Full reference: `docs/COVARIANCE.md`.
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
//! - Turn on verbose mode. Set `verbose: true` on `LmConfig` and
//!   every LM step prints cost, lambda, and the step outcome. On a
//!   Cholesky rejection the line also reports non-finite counts for
//!   grad / diagonal / cur_x / matrix -- four quick signals that narrow
//!   the problem before any deeper digging (a non-positive diagonal is
//!   caught before the inner loop and fails the solve with
//!   `SolveFailureKind::DegenerateDiagonal`, naming the parameter):
//!
//!   ```ignore
//!   let cfg = arael::simple_lm::LmConfig::conservative().with_verbose(true);
//!   let result = model.solve_sparse(&cfg);
//!   ```
//!
//!   A healthy pass looks like steady cost drops with rising /
//!   stabilising step sizes and no Cholesky rejections -- see
//!   `examples/slam_demo.rs` run for a reference trace. The trace
//!   points at the rest of this list: a run of rejections before the
//!   first major cost reduction is the initial-damping entry; NaN /
//!   Inf counts and diag <= 0 each have their own entry below.
//!
//! - Initial damping. The `lambda` column in the verbose output
//!   is the LM damping, seeded by
//!   [`simple_lm::LmConfig::initial_lambda`] (1e-4 in
//!   [`conservative`](simple_lm::LmConfig::conservative)). Too small
//!   and early steps are near-Gauss-Newton: they overshoot from a
//!   rough start and get rejected, and lambda has to climb before
//!   progress begins. Too large and every step is a short
//!   gradient-descent crawl. Starting near a minimum (a re-solve
//!   after a small perturbation): 1e-8 to 1e-6. Starting far from
//!   one: 1e-3 or higher. The
//!   [`well_conditioned`](simple_lm::LmConfig::well_conditioned) and
//!   [`ill_conditioned`](simple_lm::LmConfig::ill_conditioned)
//!   presets cover the two regimes. When no single value works,
//!   `ill_conditioned`'s gain-ratio
//!   [`NielsenLambdaDriver`](simple_lm::NielsenLambdaDriver) adapts
//!   lambda per step instead of holding a schedule. Re-solving the
//!   same problem:
//!   [`continue_from`](simple_lm::LmConfig::continue_from) seeds
//!   `initial_lambda` from the previous solve's `final_lambda`.
//!
//! - Better initial values. LM is a local method: it linearises
//!   around the current parameters, so a starting point far from the
//!   optimum converges slowly, lands in the wrong minimum, or stalls
//!   -- and a robustifier makes the requirement stricter, since it is
//!   built to ignore far-out residuals. Seed from whatever structure
//!   you have (odometry, a closed-form estimate, the previous solve)
//!   before tuning anything else. See "Initialisation matters" below.
//!
//! - Run [`model.validate()`](simple_lm::LmProblem::validate). The
//!   model linter reports non-finite parameters, stale refs,
//!   unconstrained parameters, and derivative mismatches in one pass,
//!   instead of a solve failing on the first one.
//!
//! - Cost breakdown by label. Name your constraint attributes with
//!   `#[arael(constraint(hb, name = "drift", { ... }))]` so each group
//!   shows up under its own label in the sum-of-squares. Call
//!   `model.calc_cost_table(&params)` to get `HashMap<&'static str, T>`:
//!
//!   ```ignore
//!   use arael::model::JacobianModel;
//!   let table = model.calc_cost_table(&params);
//!   for (label, cost) in &table { println!("{:<20} {:.6}", label, cost); }
//!   ```
//!
//!   A single label dominating the total is usually the culprit --
//!   either an overly tight sigma, bad initial values for its inputs,
//!   or a constraint that's mathematically unsatisfiable. Read the
//!   small end too: a label that is missing, zero, or suspiciously
//!   small may be a constraint that never ran. Check its guard, that
//!   the collection it iterates is non-empty, and that its refs point
//!   where you think.
//!
//! - NaN or Inf residuals / derivatives. The verbose-mode output
//!   already tells you whether grad / matrix / params
//!   contain non-finite values at the failing step. If they do, walk
//!   the Jacobian to find the specific row:
//!
//!   ```ignore
//!   let j = model.calc_jacobian(&params);
//!   for row in &j.rows {
//!        if !row.residual.is_finite()
//!            || !row.entries.iter().all(|(_, v)| v.is_finite())
//!        {
//!            eprintln!("bad row cid={} label={}", row.constraint, row.label);
//!        }
//!   }
//!   ```
//!
//!   A NaN residual or partial derivative usually means a `sqrt`,
//!   `acos`, `asin`, or `atan2` saw a degenerate input (zero-length
//!   vector, both-zero arguments, `|x| > 1` for asin/acos).
//!   `arael_sym` ships `safe_sqrt`, `safe_asin`, `safe_acos`,
//!   `safe_atan2` that clamp / regularise at the singular point and
//!   produce non-diverging derivatives. Before reaching for them,
//!   though, prefer to redesign the constraint so the singularity
//!   can't be hit. A `safe_*` wrapper hides the degeneracy from
//!   the solver and may leave the residual insensitive to the
//!   parameters that should drive it out of the singular region;
//!   an equivalent constraint formulated on the right geometric
//!   quantity avoids the singularity entirely. E.g. match 3D
//!   landmarks to features in 3D space (compare world-frame
//!   directions or positions) instead of projecting through a
//!   camera model and computing 2D image-plane residuals. 3D is not
//!   automatically better: a badly chosen 3D quantity keeps the
//!   behind-camera and singularity problems. A well-chosen one has
//!   neither.
//!
//! - Derivatives that disagree with finite differences.
//!   [`check_gradients`](simple_lm::LmProblem::check_gradients)
//!   compares the assembled gradient against central differences of
//!   `calc_cost` and reports each component that disagrees:
//!
//!   ```ignore
//!   let d = model.check_gradients(&params);
//!   if !d.is_clean() { println!("{}", d); }
//!   ```
//!
//!   It catches a wrong `derivs = [...]` on an `#[arael::function]`
//!   or a wrong `deriv =` cache: the cost comes from the function
//!   value and the gradient from the declared derivative, so only a
//!   finite-difference comparison sees them disagree. The symptom is
//!   a solve that stalls or oscillates at a plausible-looking cost,
//!   because the step direction is not downhill.
//!   [`check_gradients_tol`](simple_lm::LmProblem::check_gradients_tol)
//!   loosens the comparison for noisy f32 costs.
//!   [`validate`](simple_lm::LmProblem::validate) runs the same
//!   check; use the standalone form for a custom tolerance or to
//!   re-check one fix.
//!
//! - Non-positive diagonal. A solve that fails with
//!   `SolveFailureKind::DegenerateDiagonal { param, fault }` is the loudest
//!   possible signal that some parameter is untouched by every constraint
//!   (indices left at `u32::MAX`) or is receiving a negative
//!   contribution. The error names the parameter. Either outcome is a
//!   bug distinct from f32 accumulation noise.
//!
//! - Gradient magnitude. After
//!   [`simple_lm::LmProblem::calc_grad_hessian_dense`], the maximum
//!   absolute gradient component should be small relative
//!   to the cost scale at a local minimum. A huge gradient with tiny
//!   cost means the parameter scaling is off -- one parameter moves
//!   cost several orders of magnitude more than another, which
//!   destabilises Levenberg-Marquardt.
//!
//! - Hessian health. The same `hessian` array should be finite and
//!   positive-semi-definite at a minimum (smallest eigenvalue >= 0
//!   modulo roundoff). A significantly-negative smallest eigenvalue
//!   means the Gauss-Newton approximation J^T J is a poor local fit
//!   -- often because constraints are ill-conditioned or cancel.
//!
//! - Stiffness. Ratios between the smallest and largest sigmas
//!   (or equivalently, smallest and largest eigenvalues of J^T J)
//!   that span many orders of magnitude make the problem numerically
//!   stiff. LM damping has to pick a lambda that suits both ends,
//!   which is hard at f32 precision. Keep isigmas comparable where
//!   you can; if a tight constraint dominates one direction, a gauge
//!   direction orthogonal to it will starve for signal. Starting
//!   with a loose scale and ramping up (graduated optimisation --
//!   see `examples/loc_demo.rs` and `slam_demo.rs` for the
//!   `frine_isigma_scale` pattern) helps LM climb a stiff problem
//!   without rejecting early steps.
//!
//! - Simpler math beats clever math. Reformulate residuals on the
//!   most natural geometric quantity. 3D direction / position errors
//!   are cheaper and better-conditioned than 2D reprojection errors;
//!   relative rotations compared as matrices or unit quaternions
//!   avoid Euler-angle gimbal lock; distances compared in squared
//!   form avoid `sqrt` derivatives near zero. Every nonlinear
//!   operation you remove is one less place for the residual /
//!   derivative to misbehave and one less source of stiffness.
//!
//! - Inspect the generated code. Use [`cargo expand`](https://github.com/dtolnay/cargo-expand)
//!   to see what the macro emitted for your constraint. See the
//!   next section for a walkthrough.
//!
//! - Rank / DOF. Call [`model::Jacobian::singular_values`] (or the
//!   full [`Jacobian::svd`](model::Jacobian::svd) for directions):
//!
//!   ```ignore
//!   let j = model.calc_jacobian(&params);
//!   let svs = j.singular_values();
//!   println!("{:?}", svs); // descending; near-zero entries count free DOF
//!   ```
//!
//!   Near-zero singular values count the degrees of freedom. If this
//!   is higher than you expect, the model is under-constrained. The
//!   right singular vectors (columns of [`SvdResult::v`](model::SvdResult))
//!   corresponding to near-zero singular values name the unconstrained parameter
//!   directions -- useful for identifying *which* parameters are free.
//!   SVD is always performed in f64 regardless of the model's element
//!   type, so rank detection stays reliable even for f32 models.
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
//! To see what the macros generated for your equations, expand them.
//! `#[arael::model]` does a lot: it
//! interprets the constraint body symbolically, differentiates it
//! against every reachable parameter, runs common-subexpression
//! elimination, and emits Rust code for three call paths
//! (`__compute_blocks`, `__build_blocks`, `calc_jacobian`).
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
//! ### What the expansion contains
//!
//! - `__build_blocks` -- where every declared `SelfBlock` /
//!   `CrossBlock` gets its place in the block store and its global
//!   parameter indices. A block absent here is not part of the
//!   Hessian.
//! - `__compute_blocks` -- the grad + block-Hessian accumulation
//!   path. Each constraint is a nested block with its own CSE'd body.
//! - `calc_jacobian` -- same body structure but builds a
//!   `JacobianRow` per residual instead of accumulating into the
//!   blocks. Generated only when you declare `#[arael(root, jacobian)]`.
//! - source markers -- doc comments like
//!   `/// arael: PointFrine[<name>] @ path/to/file.rs:NNN`
//!   pinpoint the constraint attribute each block came from.
//!
//! Expansion grows quickly (a full SLAM model runs to thousands of
//! lines). Use `sed -n` or a pager scoped to the method you care
//! about:
//!
//! ```bash
//! cargo expand --example slam_demo | sed -n '/fn __compute_blocks/,/^    fn /p'
//! ```
//!
//! ## Builds are slow
//!
//! Model code generation runs inside the `arael` proc-macros at compile
//! time, and a dev build compiles those macros unoptimized -- so a large
//! model can spend a long time expanding. Run just the macro crates in
//! release mode, without optimizing your own debug build, by adding to your
//! `Cargo.toml`:
//!
//! ```toml
//! [profile.dev.package.arael-macros]
//! opt-level = 3
//! [profile.dev.package.arael-sym]
//! opt-level = 3
//! [profile.release.package.arael-macros]
//! opt-level = 3
//! [profile.release.package.arael-sym]
//! opt-level = 3
//! ```
//!
//! `cargo arael setup` writes these into the workspace root for you.
//!
//! # Robustness and loss functions
//!
//! Given sensor readings stacked into a vector $L$, model parameters $M$
//! (poses, landmarks, etc.), and a prediction $\mu(M)$ of what the sensors
//! should report given $M$, Bayesian inference with $L \mid M \sim
//! \mathcal{N}(\mu(M), K_L)$ -- where $K_L$ is the sensor covariance
//! matrix -- leads to minimising the sum
//!
//! $$
//! S(M) = (L - \mu(M))^T K_L^{-1} (L - \mu(M)).
//! $$
//!
//! Whitening. Diagonalising $K_L = R D R^T$ and substituting $L^D =
//! R^T L$, $G(M) = R^T \mu(M)$ turns the quadratic form into a plain sum
//! of squares in units of standard deviations:
//!
//! $$
//! S(M) = \sum_i r_i^2, \qquad r_i = \frac{L_i^D - G_i(M)}{\sigma_i}.
//! $$
//!
//! The solver minimises $S(M)$ (the Gauss-Newton / LM step), and every
//! inner term $r_i$ is dimensionless -- a pure sigma count.
//!
//! Each $r_i^2$ grows as the *square* of the measurement error. With
//! proper covariances and no outliers a typical $|r_i|$ sits around $1$
//! (the whitening divides by the noise scale, so inliers cluster near
//! unity), and $\mathbb{E}[r_i^2] = 1$ per residual. A single bad
//! association at $10\sigma$ already contributes $100$ to the sum; at
//! $30\sigma$ it contributes $900$. A handful of bad measurements drown
//! out the signal from hundreds of good ones and pull the optimum off.
//!
//! Two mechanisms address this, freely combined per constraint: a block-
//! level M-estimator loss attached to the constraint, and the per-element
//! Starship wrapper written directly into the residual expression.
//!
//! ## Block-level loss
//!
//! A `loss` modifier on a constraint robustifies the whole residual block
//! at once: it takes the squared norm $s = \lVert r \rVert^2$ and replaces
//! it with $\rho(s)$, scaling the block's gradient and Hessian by the
//! weight $\rho'(s)$.
//!
//! ```ignore
//! #[arael(constraint(hb, loss = |s| loss_geman_mcclure(s, self.c2), {
//!     [(obs.u - proj.u) * obs.iw, (obs.v - proj.v) * obs.iw]
//! }))]
//! ```
//!
//! The closure argument is the block squared norm; the scale (`k2`, `c2`)
//! is squared, in the same chi-square units as `s` -- an inlier threshold
//! like the chi-square quantile 7.815 goes in unchanged. Five kernels ship
//! (`loss_huber`, `loss_soft_l1`, `loss_cauchy`, `loss_geman_mcclure`,
//! `loss_tukey`); `loss_select(kind, s, k2)` picks one of them by an
//! integer field at runtime and a `match` in the closure picks any set
//! of them, so one residual body serves every kernel a solve may want;
//! or write any differentiable expression -- `|s| s` is plain least
//! squares.
//! Unlike the per-element Starship wrapper (next subsection) this is a
//! standard M-estimator: the down-weighting depends only on the block
//! norm, so it is invariant to how the residual axes are oriented. Scaling
//! by $\rho'(s)$ keeps the Hessian positive semidefinite. See
//! `docs/SYM.md` for the kernel formulas.
//!
//! `examples/slam_demo_gm.rs` is the worked showcase: a visual SLAM
//! problem with 50% wrong feature associations, carried by a Geman-McClure
//! block loss on the feature and GPS residuals (`--loss cauchy` switches
//! the kernel).
//!
//! ## Starship robust error suppression
//!
//! Wrapping a residual $r$ in $\gamma \arctan(r / \gamma)$ is the
//! Starship method ([US Patent
//! 12,346,118](https://patents.google.com/patent/US12346118)) -- a way to
//! cap how much a single outlier can move the optimum without losing
//! smooth differentiability. This section explains what it does and why.
//!
//! The usual robust-loss fixes -- L1 ($|r|$) and Huber (quadratic near
//! zero, linear past a threshold) -- replace $r^2$ with something that
//! grows slower than quadratically, which limits but does not cap each
//! residual's contribution; a single very bad outlier can still pull the
//! solution. Their derivatives are also awkward: L1 has a kink at $r = 0$
//! (undefined derivative there), Huber has a kink at the quadratic-to-
//! linear transition (continuous but not smooth), and Gauss-Newton wants a
//! smooth Jacobian. We want a loss that is both fully bounded and smooth
//! everywhere.
//!
//! ### The suppression function
//!
//! We look for a function $\alpha(x)$ that behaves like $x$ in the normal
//! range but saturates for large inputs, so that $\alpha(x)^2$ contributes
//! a bounded amount $\Delta S_{\max}$ to the sum instead of an unbounded
//! $x^2$.
//!
//! A clean choice is
//!
//! $$
//! \alpha(x) = \gamma \arctan\frac{x}{\gamma}, \qquad \gamma = \frac{2 \sqrt{\Delta S_{\max}}}{\pi}.
//! $$
//!
//! The $\gamma$ value follows from the saturation requirement: as $|x| \to
//! \infty$, $\arctan(x/\gamma) \to \pm \pi/2$, so $\alpha(x)^2 \to (\gamma
//! \pi / 2)^2$; setting that equal to $\Delta S_{\max}$ and solving gives
//! the $\gamma$ above. Three further properties fall out:
//!
//! - $\alpha(x) \approx x$ for $|x| \sim 1$ -- small residuals pass
//!   through unchanged.
//! - $\alpha'(0) = 1$, so near the optimum the loss is indistinguishable
//!   from plain $r^2$.
//! - $\alpha(x)^2 \to \Delta S_{\max}$ as $|x| \to \infty$ -- no single
//!   residual can push the sum by more than $\Delta S_{\max}$.
//!
//! ![Starship suppression function](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/starship/capper.png)
//!
//! Left: $\alpha(x)$ (green) bends away from the identity $x$ (purple)
//! once $|x|$ moves past a few sigmas. Right: the *squared* contribution
//! -- the unbounded $x^2$ parabola vs the saturating $\alpha(x)^2$, capped
//! at $\Delta S_{\max}$. The cap is also smooth; Gauss-Newton still sees a
//! well-defined Jacobian everywhere.
//!
//! ### Applying it to a sum of residuals
//!
//! Replace each $r_i$ in the sum with $\alpha(r_i)$:
//!
//! $$
//! \hat{S}(M) = \sum_i \alpha(r_i)^2 = \sum_i \left[ \gamma \arctan\frac{L_i^D - G_i(M)}{\gamma \sigma_i} \right]^2.
//! $$
//!
//! In practice $\Delta S_{\max}$ in the range $[10, 25]$ (so $\gamma$
//! between roughly $2$ and $3$) suppresses genuine outliers hard without
//! biasing inlier-dominated regions. Since residuals are already sigma-
//! scaled, this corresponds roughly to saying "residuals past $3$ to
//! $5\sigma$ stop mattering".
//!
//! In arael this is exactly what you see in the demo constraint bodies:
//!
//! ```ignore
//! let plain_r = (a * e.x + b - e.y) / sigma;
//! gamma * atan(plain_r / gamma)
//! ```
//!
//! The symbolic-differentiation pipeline handles `atan`'s derivative
//! automatically; from the macro's point of view the residual is just
//! another expression. No special-case code, no outlier bookkeeping.
//!
//! ## Initialisation matters
//!
//! Gauss-Newton (and Levenberg-Marquardt) is a local method: each step
//! linearises the cost around the current $M$ and moves in the direction
//! that linearisation suggests. For any loss, you need a starting $M_0$
//! close enough to the optimum that the linearisation is informative.
//!
//! Every robustifier makes this requirement stricter -- that is its
//! purpose. The redescending block kernels (Geman-McClure, Tukey) and the
//! Starship cap all shrink the gradient of far-out residuals; for Starship
//! it falls off as $\alpha'(r) = 1 / (1 + \pi^2 r^2 / (4 \Delta
//! S_{\max}))$, so at the recommended $\Delta S_{\max} = 25$ a residual at
//! $5\sigma$ still carries about 29% of its least-squares pull and a
//! $10\sigma$ residual about 9% -- still usable. Once you get out to
//! $20\sigma$ and beyond it drops under 3% and those residuals are
//! effectively frozen. If $M_0$ puts many residuals that far out --
//! outliers by position, not by nature -- the solver has nothing to work
//! with and stalls, whichever mechanism is doing the suppression.
//!
//! The usual remedy is graduated optimisation: start with the
//! suppression loose (everything in the informative regime), solve, then
//! tighten across passes down to the target. The SLAM demos do this via a
//! `frine_isigma_scale` field stepped per pass -- `slam_demo` with the
//! per-element wrapper, `slam_demo_gm` with the block loss (re-
//! snapshotting landmark anchors between passes).
//!
//! ## What loss function should I use?
//!
//! Whichever works on your data -- there is no universally best kernel.
//! The shipped kernels and the Starship wrapper differ in how hard they
//! redescend, and the right choice depends on the contamination level, the
//! residual dimensionality, and how good the initial guess is; the scale
//! (`c2`, `gamma`) matters as much as the kernel itself. Kernels are
//! one-line changes, and `branch()` switches them at run time
//! (`slam_demo_gm --loss gm|cauchy` does that), so compare them on your
//! data.
//!
//! A loss is also not the whole story: pruning bad measurements outright
//! (gating by residual, dropping associations that fail a validity check)
//! and good initial conditions carry as much weight in reaching a good
//! solution as the loss does. A robustifier can only down-weight what the
//! initialisation puts within its reach.
//!
//! And a robust loss fixes one failure mode -- outliers, heavy tails in
//! the noise -- and only that. It does nothing about systematic error:
//! a constant sensor offset, a wrong scale, an unmodelled bias, a
//! miscalibrated extrinsic. A systematic error is present in every
//! measurement, so it reads as signal, not an outlier -- the loss
//! down-weights nothing and the bias flows straight into the estimate. A
//! redescending kernel can make it worse: if the bias pushes the true
//! solution to large residuals, the loss rejects the measurements that
//! carry the correction and settles into a wrong basin. The remedies are
//! different in kind -- calibrate or repair the sensor, improve the
//! measurement model, or add the offset as a parameter estimated jointly
//! with the state (observable when the geometry constrains it
//! independently). A loss is for a minority of bad measurements among
//! unbiased good ones, not a substitute for a correct model.
//!
//! # Examples
//!
//! The `examples/` directory is the primary place to see the API in use.
//! Each file is a runnable `cargo run --release --example <name>`.
//!
//! - [`linear_demo`](https://github.com/harakas/arael/blob/master/examples/linear_demo.rs)
//!   -- robust linear regression on noisy 2D data. Residual wrapped
//!   in `gamma * atan(r / gamma)` -- the Starship method
//!   (US12346118), same robustifier used by the feature constraints
//!   in loc/SLAM. Minimal single-struct model + LM fit, compared
//!   against plain closed-form least squares.
//! - [`robust_curve_fitting`](https://github.com/harakas/arael/blob/master/examples/robust_curve_fitting.rs)
//!   -- fit `y = exp(m*x + c)` to data with two gross outliers: a
//!   plain fit against a block Cauchy loss (`loss = |s|
//!   loss_cauchy(s, 0.25)`) and the per-element starship wrapper. The
//!   robust fits recover the true parameters; the plain fit is
//!   dragged off. Ports Ceres's robust_curve_fitting example.
//! - [`slam2d_simple_demo`](https://github.com/harakas/arael/blob/master/examples/slam2d_simple_demo.rs)
//!   -- minimal pedagogical 2D SLAM: pose = (x, y, gamma), single
//!   forward-facing camera reporting a bearing per landmark, diagonal
//!   odometry covariance, first pose held fixed at the origin facing east
//!   via `optimize = false` on its params (the fixed reference that makes
//!   the solution unique -- all measurements are relative). Constraint
//!   bodies use the symbolic
//!   `matrix2sym::rotation(angle)` surface. After the LM fit it recovers
//!   each landmark's covariance (`assemble_covariance`) and writes
//!   `slam2d_simple.eps` -- a colour-per-landmark plot of the trajectory,
//!   the bearing fan, ground-truth shadows, the GT-vs-estimate error
//!   lines, and 95% confidence ellipses around each landmark (visibly
//!   elongated along the radial direction, illustrating that depth is
//!   the unobservable dimension of bearing-only SLAM).
//! - [`slam2d_multi_demo`](https://github.com/harakas/arael/blob/master/examples/slam2d_multi_demo.rs)
//!   -- multi-run merge on a nested model tree
//!   (`Map { paths: Vec<Path>, landmarks }`): three GPS-anchored runs
//!   fused in one solve via bearings onto shared root-level landmarks
//!   (`ref = root.landmarks`, run-local odometry via
//!   `ref = parent.poses`). Writes `slam2d_multi.eps` with 95%
//!   ellipses.
//! - [`slam2d_align_demo`](https://github.com/harakas/arael/blob/master/examples/slam2d_align_demo.rs)
//!   -- builds one map from three runs the cheap, two-step way: each
//!   run is solved on its own, then a small second step lines the runs
//!   up with each other. Same scene as `slam2d_multi_demo` (which solves
//!   everything at once), so the two results can be compared; writes
//!   `slam2d_align.eps`.
//! - [`slam2d_direct_align_demo`](https://github.com/harakas/arael/blob/master/examples/slam2d_direct_align_demo.rs)
//!   -- the cheapest variant of the same merge: only the per-run
//!   corrections are optimized (9 parameters), with each run pair's
//!   landmark matches accumulating into one shared Hessian block
//!   (`constraint(parent.hb, ...)`); the landmarks are then fused in
//!   closed form. Same scene again; writes `slam2d_direct_align.eps`.
//! - [`slam_demo`](https://github.com/harakas/arael/blob/master/examples/slam_demo.rs)
//!   -- synthetic monocular SLAM: S-curve trajectory, 60 poses,
//!   240 landmarks, odometry + tilt + GPS + feature observations.
//!   Full verbose-LM trace across graduated isigma passes -- the
//!   reference for what a healthy solver run looks like. Built with a
//!   GPU feature (`--features gpu-cuda`), it solves on the machine's GPU
//!   and prints the solve report.
//! - [`slam_demo_gm`](https://github.com/harakas/arael/blob/master/examples/slam_demo_gm.rs)
//!   -- the same scene with other modelling choices: selectable Geman-McClure /
//!   Cauchy block loss (`--loss gm|cauchy`) instead of the per-element
//!   starship wrap, a `TransformParam` pose, anchored inverse-depth
//!   landmarks (`UnitVecParam` direction + inverse range, re-anchored
//!   between ramp passes), trig-free chord residuals, and the
//!   graduated ramp through one `LmSession`.
//! - [`loc_demo`](https://github.com/harakas/arael/blob/master/examples/loc_demo.rs)
//!   -- localisation with fixed known landmarks (no gauge freedom).
//!   Block-tridiagonal Hessian + band solver. Graduated-isigma
//!   optimisation via a root `frine_isigma_scale` field.
//! - [`loc_global_demo`](https://github.com/harakas/arael/blob/master/examples/loc_global_demo.rs)
//!   -- how to put `Param` fields on the root struct and have
//!   constraints consume them. Uses a system-global rigid transform
//!   (translation + 3-axis rotation applied to every pose) as the
//!   running example; every residual that reads the robot's world
//!   pose composes the globals before evaluating. Shows the two
//!   wiring shapes for pose<->root cross-Hessian pairs
//!   (`CrossBlock<Pose, Path>` on the constraint struct, and
//!   `[hb_pose, coo]` with no block field) and a
//!   `Path::optimise_center` pass that freezes pose
//!   params and optimises only the globals before the main sweep.
//! - [`plane_slam_demo`](https://github.com/harakas/arael/blob/master/examples/plane_slam_demo.rs)
//!   -- plane SLAM with a user-defined component: `UnitVec<T>`, a 2-DOF
//!   unit direction on the sphere, demonstrating
//!   `#[arael(component)]`.
//! - [`m3500_demo`](https://github.com/harakas/arael/blob/master/examples/m3500_demo.rs)
//!   -- the classic M3500 Manhattan-world pose-graph benchmark
//!   (Olson 2006): 3500 SE2 poses and 5453 relative-pose constraints
//!   from a g2o file, the between-factor written symbolically, solved
//!   with sparse faer LM. The same model backs `benchmarks/pgo`.
//! - [`bal_demo`](https://github.com/harakas/arael/blob/master/examples/bal_demo.rs)
//!   -- bundle adjustment on a real Bundle-Adjustment-in-the-Large
//!   Ladybug problem (49 cameras, 7776 points, 31843 observations,
//!   from the vendored file). The Snavely reprojection residual
//!   written symbolically; verbose LM with the Nielsen driver drives
//!   the solve to the same optimum as Ceres. Same model as
//!   `benchmarks/bal`.
//! - [`model_demo`](https://github.com/harakas/arael/blob/master/examples/model_demo.rs)
//!   -- minimal `#[arael::model]` walk-through showing how
//!   `Param`, `SimpleEulerAngleParam`, and the update cycle fit
//!   together.
//! - [`single_root_demo`](https://github.com/harakas/arael/blob/master/examples/single_root_demo.rs)
//!   -- single-struct model-and-root + a direct-composed sub-model,
//!   each carrying its own `SelfBlock<Self>`. The smallest example
//!   that exercises the "root has its own params" path.
//! - [`refs_demo`](https://github.com/harakas/arael/blob/master/examples/refs_demo.rs)
//!   -- `Ref<T>`, `refs::Vec`, `refs::Deque`, and `refs::Arena`
//!   behaviour: insertion, iteration, stable handles.
//! - [`jacobian_demo`](https://github.com/harakas/arael/blob/master/examples/jacobian_demo.rs)
//!   -- `#[arael(root, jacobian)]`, `#[arael(constraint_index)]`, and
//!   `calc_jacobian` / `calc_cost_table` walk-through. End-to-end
//!   reference for the instrumentation features referenced from
//!   "My solve doesn't converge".
//! - [`runtime_fit_demo`](https://github.com/harakas/arael/blob/master/examples/runtime_fit_demo.rs)
//!   -- curve fitting where the residual equation is a string parsed
//!   at runtime. Demonstrates `ExtendedModel` + robust loss on top
//!   of the symbolic front end.
//! - [`root_fit_demo`](https://github.com/harakas/arael/blob/master/examples/root_fit_demo.rs)
//!   -- curve fitting on the regular model system: the parameters
//!   live on the root, each measurement is a plain data entity whose
//!   constraint writes into the root's own `SelfBlock` through
//!   `root.hb`. The full constraint feature set (guards, block
//!   losses, named groups) in place of the `fit(...)` one-liner.
//! - [`user_function_demo`](https://github.com/harakas/arael/blob/master/examples/user_function_demo.rs)
//!   -- `#[arael::function]` for user-defined operators in
//!   constraint bodies. Form A purely symbolic
//!   `sigmoid(x) = 1 / (1 + exp(-x))` and Form B opaque numerical
//!   `my_safe_asin` with a closed-form symbolic derivative, both
//!   used in a single two-residual LM fit.
//! - [`sym_demo`](https://github.com/harakas/arael/blob/master/examples/sym_demo.rs)
//!   -- symbolic-math tour: expression building, automatic
//!   differentiation, CSE, pretty printing, parsing. No solver
//!   involvement; pure `arael-sym`.
//! - [`calc_demo`](https://github.com/harakas/arael/blob/master/examples/calc_demo.rs)
//!   -- `bc`-style REPL calculator built on `arael-sym`. Shows
//!   `parse_with_functions` + `FunctionBag` for user-defined
//!   functions, persistent history via rustyline.
//! - [`bench_band`](https://github.com/harakas/arael/blob/master/examples/bench_band.rs)
//!   -- benchmarks the band Cholesky backend against dense on the
//!   localisation model at increasing pose counts. Prints timing +
//!   speedup.
//! - [`bench_sparse`](https://github.com/harakas/arael/blob/master/examples/bench_sparse.rs)
//!   -- sparse Cholesky backends (faer) vs dense on SLAM.
//! - [`bench_investigate`](https://github.com/harakas/arael/blob/master/examples/bench_investigate.rs)
//!   -- deeper comparison of sparse backends on the
//!   SLAM model, with assembly vs solve breakdown and numeric
//!   cross-check of the solutions.
//! - [`gpu_lm_trace`](https://github.com/harakas/arael/blob/master/examples/gpu_lm_trace.rs)
//!   -- counts the CUDA launches, waits and copies of each attempt of a
//!   bundle adjustment solved on the GPU; run under
//!   `scripts/gpu_trace.py`, which needs NVIDIA's Nsight Systems
//!   (`--features gpu-cuda`).
//!
//! ## External and larger examples
//!
//! - [Pose-graph optimization in the browser](https://arael.mare.ee/pgo/)
//!   -- the JavaScript interface running the solver as WebAssembly: a
//!   pose graph with poses to drag, lock and fix, re-solved live.
//!   Source in [cxx-examples/m3500_demo](https://github.com/harakas/arael/tree/master/cxx-examples/m3500_demo).
//! - [arael-sketch](https://github.com/harakas/arael/blob/master/docs/ARAEL_SKETCH.md)
//!   -- the constraint-based 2D sketch editor,
//!   [live in the browser](https://sketch.mare.ee/): geometric
//!   constraints, parametric dimensions, undo, a command interface and
//!   an MCP server, all on arael.
//! - [rtabmap-arael](https://github.com/harakas/rtabmap-arael) --
//!   RTAB-Map with arael as a graph optimizer: pose-graph optimization
//!   and bundle adjustment with marginal covariance, measured against
//!   its Ceres, GTSAM, g2o and TORO backends. The models and the
//!   numbers are in
//!   [corelib/src/optimizer/arael](https://github.com/harakas/rtabmap-arael/tree/arael-optimizer/corelib/src/optimizer/arael).
//! - [slam_toolbox-arael](https://github.com/harakas/slam_toolbox-arael)
//!   -- ROS 2 slam_toolbox with an arael solver plugin reproducing its
//!   Ceres plugin's residual, measured against it on 2D pose-graph
//!   datasets. The model and the numbers are in
//!   [solvers/arael](https://github.com/harakas/slam_toolbox-arael/tree/arael-solver/solvers/arael).
//! - [colmap-arael](https://github.com/harakas/colmap-arael) --
//!   COLMAP with arael as its bundle adjustment backend: the incremental
//!   mapper's local and global bundle adjustments on the Ceres backend's
//!   problem (every camera model, rigs, loss functions, position priors),
//!   measured against Ceres and the GPU solver caspar. The model and the
//!   numbers are in
//!   [src/colmap/estimators/arael](https://github.com/harakas/colmap-arael/tree/arael-optimizer/src/colmap/estimators/arael).
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
//! The editor includes a command panel (`/` to toggle) with 79 scripting
//! commands, and an embedded MCP server (`--mcp`) that enables AI agents
//! like Claude Code to create and modify sketches programmatically.
//!
//! ![Dark mode](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/arael-sketch/docs/dark.png)
//!
//! # Crate structure
//!
//! - `arael` (this crate) -- the runtime: the `Model` trait and Hessian
//!   blocks, the Levenberg-Marquardt solver with dense/band/sparse backends,
//!   covariance recovery, rotation and transform parameters,
//!   vectors/matrices, cameras, g2o file I/O, validation
//! - `arael-sym` -- symbolic math engine: expression trees, differentiation,
//!   simplification, CSE, Rust/LaTeX code emission (`arael-sym-macros`
//!   supplies its `sym!` macro)
//! - `arael-macros` -- the procedural macros: `#[arael::model]`, constraint
//!   code generation, `#[arael::function]`
//! - `arael-faer` -- sparse extensions over faer: block CSC storage, the
//!   Schur-complement reduction, supernodal and envelope block Cholesky,
//!   conjugate gradients, a nested-dissection ordering, and the worker
//!   pool
//! - `arael-gpu` -- the GPU runtimes (CUDA, HIP, Metal, and the CPU device
//!   for tests) and the GPU solver's kernels, used through this crate's
//!   `gpu-*` features
//! - `cargo-arael` -- the `cargo arael` subcommand: `export` generates the
//!   C ABI, C++, Python and JavaScript (WebAssembly) interfaces, `kernels`
//!   writes the GPU kernels the roots generate
//! - `arael-sketch`, `arael-sketch-solver`, `arael-sketch-backend` -- the
//!   2D sketch editor: GUI, constraint-solver library, and headless
//!   command/MCP backend

#[macro_use]
pub mod log;
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
/// Runtime registry for user-defined functions declared via
/// `#[arael::function]`. Built from [`inventory`] submissions; see
/// [`user_fn::registry_bag`].
pub mod user_fn;
/// Re-export of the `inventory` crate. The `#[arael::function]` macro
/// expands to `::arael::inventory::submit!(...)` so user crates do
/// not need to depend on `inventory` directly.
#[doc(hidden)]
pub use inventory;
/// Re-export of the `arael-sym` symbolic math crate.
pub use arael_sym as sym;
/// A crate version: the three numbers and the tag after the dash.
/// `Display` writes the `Cargo.toml` form, `0.9.0-dev`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// The pre-release tag after the dash (`"dev"` in `0.9.0-dev`);
    /// empty for a release.
    pub pre: &'static str,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() {
            write!(f, "-{}", self.pre)?;
        }
        Ok(())
    }
}

const fn version_number(s: &str) -> u32 {
    let b = s.as_bytes();
    let mut i = 0;
    let mut v = 0u32;
    while i < b.len() {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

/// The version of this crate, from its `Cargo.toml`. Every generated
/// interface reports it: `arael_version()` in C, C++ and Python,
/// `araelVersion()` in JavaScript.
pub const VERSION: Version = Version {
    major: version_number(env!("CARGO_PKG_VERSION_MAJOR")),
    minor: version_number(env!("CARGO_PKG_VERSION_MINOR")),
    patch: version_number(env!("CARGO_PKG_VERSION_PATCH")),
    pre: env!("CARGO_PKG_VERSION_PRE"),
};
/// Offset into an assembled Hessian's value buffer: the width of every
/// scatter position and tile origin the solver stores. 32 bits by default,
/// which addresses 4e9 values (34 GB of `f64`); a problem past that needs the
/// alias widened in `arael-faer` and a rebuild.
pub use arael_faer::ValueIndex;
/// Model trait, parameter types, and Hessian blocks.
pub mod angle;
pub mod twofloat;
pub use twofloat::df32;
pub mod model;
pub mod twist;
/// The `twist` module's name through 0.8.
#[deprecated(since = "0.9.0", note = "renamed to `twist`")]
pub mod se3 {
    #[allow(deprecated)]
    pub use crate::twist::{se3, se3d, se3f};
}
pub mod transform;
pub mod unitvec;
/// Levenberg-Marquardt solver with dense, band, and sparse backends.
pub mod simple_lm;
pub mod threads;
pub mod store;
/// The worker threads of the sweeps and of the linear solve's own
/// threaded stages ([`arael_faer::pool`]).
pub use arael_faer::pool;
pub use threads::Context;
/// Parameter covariance recovery (`Sigma = 2 H^-1`) at the solution.
pub mod covariance;
#[cfg(feature = "gpu")]
mod device;
#[cfg(feature = "gpu")]
mod gpu_solver;
pub mod rank;
/// Pinhole camera model (`camera<T>`, `cameraf` / `camerad`) and
/// geometric utilities.
pub mod geometry;
/// .g2o pose-graph file I/O (SE2 and SE3:QUAT records).
pub mod g2o;
/// Model validation report types (`Diagnostic`, `Issue`).
pub mod validate;

/// Re-export Jacobian types for convenient access.
pub use model::{Jacobian, JacobianRow, jacobian_entries};
/// The `sym!` auto-clone macro for symbolic expressions (from `arael-sym`).
pub use arael_sym::sym;
/// Derive macro for the `Model` trait.
pub use arael_macros::Model;
/// Attribute macro: `#[arael::model]`.
pub use arael_macros::model;
/// Attribute macro: `#[arael::function]`.
pub use arael_macros::function;
/// Emit this crate's `arael_import!` macro carrying every `pub` model
/// struct defined above the invocation; other crates register them all
/// with `<crate>::arael_import!();`. See `docs/MODEL.md`.
pub use arael_macros::export_models;
#[doc(hidden)]
pub use arael_macros::__register_model;

/// serde_json for the interfaces cargo-arael generates: they save and load
/// a root that derives serde as JSON.
#[doc(hidden)]
pub use serde_json as __serde_json;

/// The device side of a model's generated code (feature `gpu`): arael-gpu
/// and the device assembly the generated code implements.
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub mod __gpu {
    pub use arael_gpu::*;
    pub use crate::device::*;
}

/// The device items of a model's generated code: kept when arael has its
/// `gpu` feature, dropped otherwise, so a model compiles either way.
#[cfg(feature = "gpu")]
#[doc(hidden)]
#[macro_export]
macro_rules! __gpu_items {
    ($($t:tt)*) => { $($t)* };
}
#[cfg(not(feature = "gpu"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __gpu_items {
    ($($t:tt)*) => {};
}

/// The device check (`ARAEL_DEVICE_CHECK`) at a root's assembly entry
/// points, `problem` at `params`: kept when arael has its `gpu` feature,
/// dropped otherwise.
#[cfg(feature = "gpu")]
#[doc(hidden)]
#[macro_export]
macro_rules! __device_check {
    ($problem:expr, $params:expr) => {
        $crate::__gpu::check_solve($problem, $params)
    };
}
#[cfg(not(feature = "gpu"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __device_check {
    ($problem:expr, $params:expr) => {};
}

/// One-stop imports for defining and solving models:
/// `use arael::prelude::*;`.
///
/// Curated, not exhaustive: it carries what a typical model-and-solve
/// program names -- parameter and block types, the rotation
/// parameterizations, the math aliases, the solve-side traits and
/// config -- and deliberately leaves out generically named or
/// specialist items (solver backends like `Dense`/`Band`, `lm_solve`,
/// `sym`), which read better explicitly
/// imported. `refs` is re-exported as a module
/// so collections stay qualified (`refs::Vec<Pose>` -- a bare `Vec`
/// would fight std's). Additions are curated: a generically named item
/// needs a strong case, since every prelude name is a potential glob
/// ambiguity downstream.
pub mod prelude {
    pub use crate::model::{
        Coo, CrossBlock, EulerAngleParam,
        ExtendedModel, JacobianModel, Model, Param, QuaternionParam,
        SelfBlock, SimpleEulerAngleParam,
    };
    pub use crate::covariance::{
        CovAssembly, CovError, CovMode, CovOptions, CovOrdering, Covariance,
    };
    pub use crate::refs::{self, Ref};
    pub use crate::simple_lm::{
        DefaultLambdaDriver, FitProblem, LambdaDriver, LmConfig, LmProblem, LmResult,
        LmSession, LmStatus, NielsenLambdaDriver, RootProblem, SolveFailure,
        SolveFailureKind, SolveResult,
    };
    // Inner API, in the prelude because a hand-written problem has to
    // name it to write its one empty impl.
    pub use crate::simple_lm::LmProblemInternals;
    pub use crate::angle::{AngleParam, AngleParamF};
    pub use crate::transform::{
        ScaledTransformParam, ScaledTransformParamF, TransformParam, TransformParamF,
        scaled_transform3, scaled_transform3d, scaled_transform3f,
        transform3, transform3d, transform3f,
    };
    pub use crate::unitvec::{UnitVecParam, UnitVecParamF};
    pub use crate::matrix::{matrix2d, matrix2f, matrix3d, matrix3f, matrixd, matrixf};
    pub use crate::quatern::{quaternd, quaternf};
    pub use crate::twist::{twist3, twist3d, twist3f};
    #[allow(deprecated)]
    pub use crate::twist::{se3, se3d, se3f};
    pub use crate::vect::{vect2d, vect2f, vect3d, vect3f, vectd, vectf};
}
