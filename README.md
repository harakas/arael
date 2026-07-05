# ARAEL

**Algorithms for Robust Autonomy, Estimation, and Localization**

A Rust framework for nonlinear optimization with compile-time symbolic differentiation. Define your model and constraints declaratively -- the macro system symbolically differentiates, applies common subexpression elimination, and generates compiled cost, gradient, and Gauss-Newton hessian (J^T J approximation) code.

## Contents

- [Features](#features)
- [Benchmarks](#benchmarks)
- [Scope](#scope)
- [Quick Example: Symbolic Math](#quick-example-symbolic-math)
- [Quick Example: Robust Linear Regression](#quick-example-robust-linear-regression)
- [SLAM Path Optimization](#slam-path-optimization)
- [Starship robust error suppression](#starship-robust-error-suppression)
- [Localization Demo](#localization-demo)
- [Examples](#examples)
- [Solvers](#solvers)
- [Runtime Differentiation](#runtime-differentiation)
- [Instrumentation and troubleshooting](#instrumentation-and-troubleshooting)
  - [My solve doesn't converge. What do I check?](#my-solve-doesnt-converge-what-do-i-check)
  - [Looking under the hood with `cargo expand`](#looking-under-the-hood-with-cargo-expand)
- [2D Sketch Editor](#2d-sketch-editor)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- **Symbolic math** -- expression trees with automatic differentiation, simplification, expansion, LaTeX/Rust code generation
- **Compile-time constraint code generation** -- write constraints symbolically, get compiled derivative code with CSE
- **Levenberg-Marquardt solver** -- with robust error suppression via the [Starship method (US12346118)](https://patents.google.com/patent/US12346118) `gamma * atan(r / gamma)` and switchable constraints (`guard = expr`)
- **Multiple solver backends** via `LmSolver` trait:
  - **Dense Cholesky** (nalgebra) -- fixed-size dispatch up to 9x9, dynamic for larger
  - **Band Cholesky** -- pure Rust O(n*kd^2) for block-tridiagonal systems (9.4x faster than dense at 500 poses)
  - **Sparse Cholesky** (faer, pure Rust) -- for general sparse hessians (66x faster than dense at 200 poses with 6% fill)
  - **Eigen SimplicialLLT** and **CHOLMOD** -- optional C++ backends via FFI (`--features eigen`, `--features cholmod`)
  - **LAPACK band** -- optional dpbsv/spbsv backend (`--features lapack`)
- **Indexed sparse assembly** -- precomputed position lists for zero-overhead hessian assembly after first iteration
- **f32 and f64 precision** -- `#[arael(root)]` for f64, `#[arael(root, f32)]` for f32 throughout
- **Model trait** -- hierarchical serialize/deserialize/update protocol for parameter optimization
- **Type-safe references** -- `Ref<T>`, `Vec<T>`, `Deque<T>`, `Arena<T>` for indexed collections with stable references
- **Runtime differentiation** -- parse equations from strings at runtime, auto-differentiate symbolically, and optimize via `ExtendedModel` + `TripletBlock` (used by the sketch editor for parametric expression dimensions)
- **User-defined functions** -- plug custom symbolic or native-eval operators into constraint bodies with `#[arael::function]`.
- **Hessian blocks** -- `SelfBlock<A>` and `CrossBlock<A, B>` for 1- and 2-entity constraints (packed dense); `TripletBlock` for 3+ entities (COO sparse)
- **Jacobian computation** -- `#[arael(root, jacobian)]` generates `calc_jacobian()` returning a sparse Jacobian matrix for DOF analysis and constraint diagnostics (see `examples/jacobian_demo.rs`)
- **Gimbal-lock-free rotations** -- `EulerAngleParam` optimizes a small delta around a reference rotation matrix
- **WASM/browser support** -- the sketch editor compiles to WebAssembly and runs in the browser via eframe/egui

## Benchmarks

Batch pose-graph optimization on city10000, the classic 10000-pose SLAM
benchmark (iSAM dataset: 30000 parameters, 20687 weighted constraints).
Best converging configuration per system, all verified to reach the
same minimum -- final costs are evaluated by one shared reference
function, so they are directly comparable. Apple M4 Pro, single
threaded:

| system | total time | iterations | final cost |
|--------|-----------:|-----------:|-----------:|
| **arael (LM, f32)** | **90.6 ms** | 9(9) | 511.99 |
| **arael (LM, f64)** | **106.4 ms** | 8(9) | 511.99 |
| [g2o](https://github.com/RainerKuemmerle/g2o) (GN) | 136.2 ms | 7 | 511.99 |
| [Ceres](http://ceres-solver.org) (LM) | 243.4 ms | 10(10) | 511.99 |
| [tiny-solver](https://crates.io/crates/tiny-solver) (GN) | 602.3 ms | 7 | 511.99 |
| [GTSAM](https://gtsam.org) (batch) | did not converge* | | |

\* GTSAM's batch LM/GN does not survive this dataset's odometry
initialization; its incremental ISAM2 solves it in 10.4 s of update
time. arael's iteration count is "accepted(total damped attempts)";
other systems report outer iterations, so per-iteration figures are
not directly comparable across systems.

Arael's iterations are extremely low cost: all derivative code is
generated and CSE-optimized at compile time, and the system matrix is
assembled through precomputed sparse positions rather than built from
scratch each iteration. On the smaller M3500 benchmark (10500
parameters) g2o's Gauss-Newton currently leads (24.5 ms vs arael's
28.5 ms) -- full tables for three dataset configurations, methodology,
and the cross-system validation harness:
[benchmarks/pgo](benchmarks/pgo/README.md).

## Scope

Arael is a **nonlinear optimization framework**, not a complete SLAM or state estimation system. The SLAM and localization demos show how to use arael as the optimizer backend, but a production SLAM pipeline would additionally need:

- **Front-end perception**: feature detection, descriptor extraction
- **Data association**: matching observed features to existing landmarks, handling ambiguous or incorrect matches
- **Landmark management**: initializing new landmarks from observations, merging duplicates, pruning unreliable ones
- **Keyframe selection**: deciding when to add new poses vs. discard redundant frames
- **Loop closure**: detecting revisited places, verifying loop closure candidates, and injecting constraints
- **Outlier rejection logic**: deciding which observations to reject
- **Marginalization / sliding window**: limiting optimization scope for real-time operation, marginalizing old poses while preserving their information
- **Map management**: spatial indexing, map saving/loading, multi-session map merging

Arael provides the compile-time-differentiated solver that sits at the core of such a system. Everything above is application-level logic that builds on top of it.

## Quick Example: Symbolic Math

```rust
use arael::sym::*;
use arael::sym;
use maplit::hashmap;

sym! {
    let (x, y) = symbols!(x, y);
    let f = sin(x) * y + pow(x, 2.0);

    println!("f(x, y)    = {}", f);           // x^2 + y * sin(x)
    println!("df/dx      = {}", f.diff(x));   // y * cos(x) + 2 * x
    println!("df/dy      = {}", f.diff(y));   // sin(x)

    let vars = hashmap!{ "x" => 2.0, "y" => 3.0 };
    println!("f(2, 3)    = {}", f.eval(&vars).unwrap()); // 6.7278...
}
```

The `sym!` macro auto-inserts `.clone()` on variable reuse, so you write natural math without Rust's ownership boilerplate.

See [docs/SYM.md](docs/SYM.md) for the full symbolic math reference.

## Quick Example: Robust Linear Regression

You describe the model as a Rust struct and the residual as an arael-sym expression; the macros do the rest.

- `#[arael::model]` auto-implements the `Model` trait for the struct: serialize / deserialize / update of every optimizable parameter, flat indexing into the residual vector, and all the hooks the solver needs.
- Every `Param<T>` field is an optimization variable. Plain fields (`data`, `sigma`, `gamma` here) are constants.
- `#[arael(fit(data, |e| ...))]` declares a least-squares fit: one residual per element of `data`, body written as a symbolic expression referencing model fields and the current data entry. The macro compiles the body into residual + gradient + Hessian code with symbolic differentiation and CSE.

The `gamma * atan(plain_r / gamma)` wrapper is the [Starship robust error-suppression method](https://patents.google.com/patent/US12346118) -- residuals up to ~gamma pass through linearly, beyond that they saturate, suppressing outlier influence while staying smoothly differentiable.

```rust
#[arael::model]
struct DataEntry { x: f32, y: f32 }

#[arael::model]
#[arael(fit(data, |e| {
    let plain_r = (a * e.x + b - e.y) / sigma;
    gamma * atan(plain_r / gamma)
}))]
struct LinearModel {
    a: Param<f32>,
    b: Param<f32>,
    data: Vec<DataEntry>,
    sigma: f32,
    gamma: f32,
}
```

The macro auto-generates `calc_cost()`, `calc_grad_hessian()`, and `fit()` methods with symbolically differentiated, CSE-optimized compiled code:

```rust
fn main() {
    let data = vec![
        DataEntry { x: -0.156, y: -0.094 },
        // ...
    ];

    let mut model = LinearModel::new(data, 0.01);

    // Initial values from ordinary least squares
    model.fit_linear_regression();
    println!("Linear regression: y = {}*x + {}", model.a.value, model.b.value);

    // Robust nonlinear fit -- suppresses outlier influence
    let result = model.fit_with(&LmConfig { verbose: true, ..Default::default() });
    println!("Robust fit: y = {}*x + {}", model.a.value, model.b.value);
}
```

The robust fit ignores outliers while tracking the inlier data:

![Linear Regression](docs/linear_regression.png)

See [docs/LINEAR.md](docs/LINEAR.md) for the full walkthrough. Full source: [examples/linear_demo.rs](examples/linear_demo.rs).

## SLAM Path Optimization

The earlier regression example fitted two scalar parameters against one residual. Real SLAM and bundle-adjustment problems have many coupled entities -- poses, landmarks, cameras -- with many constraint types between them. arael models the hierarchy as plain Rust structs, each annotated with `#[arael::model]` and one or more `#[arael(constraint(...))]` attributes. The macros walk the hierarchy at compile time, differentiate every residual symbolically, eliminate common subexpressions, and emit one fused `calc_cost` + `calc_grad_hessian` pair for the whole graph.

The demo ([examples/slam_demo.rs](examples/slam_demo.rs)) generates a synthetic S-curve trajectory with 60 poses and 240 point landmarks observed by 5 cameras. It handles 50% outlier associations with 30x pixel noise via robust suppression and graduated optimization. The solver uses faer sparse Cholesky (pure Rust) to exploit the hessian's sparsity structure.

Each entity owns its own parameters and its own `SelfBlock` -- the diagonal block of the Hessian for that entity. Constraints that touch a single entity accumulate into its self block; constraints that couple two entities (an odometry residual between two poses, a bearing residual between a landmark and a pose) accumulate into a `CrossBlock` between the pair. The assembled Hessian therefore mirrors the model hierarchy: one block row/column per entity, a self block on the diagonal, and a cross block off-diagonal wherever a constraint ties two entities together. Entities that never share a constraint remain exactly zero in that corner of the matrix -- which is where the sparsity comes from.

![Hessian Sparsity](docs/sparsity.png)

The pattern in the S-curve demo above shows pose-pose blocks (upper-left), pose-landmark coupling (off-diagonal), and landmark self-blocks (lower-right diagonal). The faer sparse Cholesky solver exploits this, achieving 66x speedup over dense at 200 poses.

A `Pose` is the robot's 6-DOF state at one timestep. Three constraint attributes stack on the same `hb_pose` Hessian block: a guarded GPS constraint (active only when GPS data is present), a drift regularizer that stabilises graduated optimization, and an accelerometer-based tilt constraint on roll and pitch. Every `Param<...>` is an optimization variable; `info` holds per-timestep measurements.

```rust
#[arael::model]
#[arael(constraint(hb_pose, guard = self.info.gps.is_some(), {
    // GPS constraint (guarded -- only when GPS data is present)
    let raw = pose.pos - pose.info.gps.pos;
    let whitened = pose.info.gps.cov_r.transpose() * raw;
    [gamma * atan(whitened.x * pose.info.gps.cov_isigma.x / gamma), ...]
}))]
#[arael(constraint(hb_pose, {
    // Drift regularizer -- prevents divergence during graduated optimization
    let pos_drift = pose.pos - pose.pos_value;
    [pos_drift.x * path.drift_pos_isigma, ...]
}))]
#[arael(constraint(hb_pose, {
    // Tilt sensor -- accelerometer constrains roll and pitch
    [(pose.ea.x - pose.info.tilt_roll) * path.tilt_isigma,
     (pose.ea.y - pose.info.tilt_pitch) * path.tilt_isigma]
}))]
struct Pose {
    pos: Param<vect3f>,
    ea: SimpleEulerAngleParam<f32>,  // precomputes sin/cos + rotation matrix
    info: PoseInfo,
    hb_pose: SelfBlock<Pose>,
}
```

A **frine** (project vocabulary) is a structure that ties one entity to one measurement. `PointFrine` is the frine for a point landmark: it binds a `PointLandmark` to a `PointFeature` -- the 2D detection in one of the pose's cameras that observed it. In factor-graph SLAM terms, a frine plays the role of a `Factor` (GTSAM) or an `Edge` (g2o). The measurement itself is pre-processed once at set-up time into a 3D direction ray in the camera frame (stored on `PointFeature`), so the solver never touches pixel coordinates or undistortion. Staying in 3D keeps derivatives smooth and sidesteps the projective singularities that show up when you differentiate through a pixel-space reprojection.

The residual transforms the landmark into the pose's frame and then into the camera's feature frame (`feature.mf2r`), and compares its direction to the stored measurement via two `atan2` bearings (azimuth and elevation). Each bearing is whitened by the feature's per-axis `isigma` and passed through the robust `gamma * atan(.../gamma)` wrapper for outlier tolerance.

The `#[arael(ref = ...)]` attributes declare which collection each reference resolves against -- `pose` from `root.poses`, `feature` chained off `pose.info.features` -- and the constraint uses a `CrossBlock<PointLandmark, Pose>` because it couples two entity types.

```rust
#[arael::model]
#[arael(constraint(hb, parent=lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_f = feature.mf2r.transpose() * (lm_r - feature.camera_pos);
    let plain1 = atan2(r_f.y, r_f.x) * feature.isigma.x;
    let plain2 = atan2(r_f.z, r_f.x) * feature.isigma.y;
    [gamma * atan(plain1 / gamma), gamma * atan(plain2 / gamma)]
}))]
struct PointFrine {
    #[arael(ref = root.poses)]          // resolved from root collection
    pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]  // chained resolution
    feature: Ref<PointFeature>,
    hb: CrossBlock<PointLandmark, Pose>,
}
```

`PosePair` is the odometry constraint between two consecutive poses -- a relative-motion residual whitened by a decomposed covariance. Another `CrossBlock`, this time Pose-to-Pose.

```rust
#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - cur.info.delta_pos;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * cur.info.delta_ea.rotation_matrix();
    let ea_err = (expected.transpose() * mr2w_cur).get_euler_angles();
    // ... whitened by decomposed covariance
}))]
struct PosePair {
    #[arael(ref = root.poses)]
    prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose>,
}
```

Finally, `Path` ties it all together. `#[arael(root)]` is what actually triggers code generation: the macro walks every constraint attribute on every reachable struct, resolves the refs, and emits `calc_cost()` / `calc_grad_hessian()` for the whole model hierarchy.

```rust
#[arael::model]
#[arael(root)]
struct Path {
    poses: refs::Deque<Pose>,
    landmarks: refs::Vec<PointLandmark>,
    pose_pairs: std::vec::Vec<PosePair>,
    gamma: f32,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    drift_lm_isigma: f32,
    tilt_isigma: f32,
}
```

See [docs/SLAM.md](docs/SLAM.md) for the full walkthrough.

## Starship robust error suppression

Both demos wrap every residual in $\gamma \arctan(r / \gamma)$. This is the **Starship method** ([US Patent 12,346,118](https://patents.google.com/patent/US12346118)) -- a way to cap how much a single outlier can move the optimum without losing smooth differentiability. This section explains what it does and why.

### Setup

Given sensor readings stacked into a vector $L$, model parameters $M$ (poses, landmarks, etc.), and a prediction $\mu(M)$ of what the sensors should report given $M$, Bayesian inference with $L \mid M \sim \mathcal{N}(\mu(M), K_L)$ -- where $K_L$ is the sensor covariance matrix -- leads to minimising the sum

$$
S(M) = (L - \mu(M))^T K_L^{-1} (L - \mu(M)).
$$

**Whitening.** Diagonalising $K_L = R D R^T$ and substituting $L^D = R^T L$, $G(M) = R^T \mu(M)$ turns the quadratic form into a plain sum of squares in units of standard deviations:

$$
S(M) = \sum_i r_i^2, \qquad r_i = \frac{L_i^D - G_i(M)}{\sigma_i}.
$$

The solver minimises $S(M)$ (the Gauss-Newton / LM step), and every inner term $r_i$ is dimensionless -- a pure sigma count.

### The outlier problem

Each $r_i^2$ grows as the *square* of the measurement error. With proper covariances and no outliers a typical $|r_i|$ sits around $1$ (the whitening divides by the noise scale, so inliers cluster near unity), and $\mathbb{E}[r_i^2] = 1$ per residual. A single bad association at $10\sigma$ already contributes $100$ to the sum; at $30\sigma$ it contributes $900$. A handful of bad measurements drown out the signal from hundreds of good ones and pull the optimum off.

The usual robust-loss fixes -- L1 ($|r|$) and Huber (quadratic near zero, linear past a threshold) -- replace $r^2$ with something that grows slower than quadratically, which limits but does not cap each residual's contribution; a single very bad outlier can still pull the solution. Their derivatives are also awkward: L1 has a kink at $r = 0$ (undefined derivative there), Huber has a kink at the quadratic-to-linear transition (continuous but not smooth), and Gauss-Newton wants a smooth Jacobian. We want a loss that is both fully bounded and smooth everywhere.

### The cap

We look for a function $\alpha(x)$ that behaves like $x$ in the normal range but saturates for large inputs, so that $\alpha(x)^2$ contributes a bounded amount $\Delta S_{\max}$ to the sum instead of an unbounded $x^2$.

A clean choice is

$$
\alpha(x) = \gamma \arctan\frac{x}{\gamma}, \qquad \gamma = \frac{2 \sqrt{\Delta S_{\max}}}{\pi}.
$$

The $\gamma$ value follows from the saturation requirement: as $|x| \to \infty$, $\arctan(x/\gamma) \to \pm \pi/2$, so $\alpha(x)^2 \to (\gamma \pi / 2)^2$; setting that equal to $\Delta S_{\max}$ and solving gives the $\gamma$ above. Three further properties fall out:

- $\alpha(x) \approx x$ for $|x| \sim 1$ -- small residuals pass through unchanged.
- $\alpha'(0) = 1$, so near the optimum the loss is indistinguishable from plain $r^2$.
- $\alpha(x)^2 \to \Delta S_{\max}$ as $|x| \to \infty$ -- no single residual can push the sum by more than $\Delta S_{\max}$.

![Starship suppression function](docs/starship/capper.png)

Left: $\alpha(x)$ (green) bends away from the identity $x$ (purple) once $|x|$ moves past a few sigmas. Right: the *squared* contribution -- the unbounded $x^2$ parabola vs the saturating $\alpha(x)^2$, capped at $\Delta S_{\max}$. The cap is also smooth; Gauss-Newton still sees a well-defined Jacobian everywhere.

### Using it

Replace each $r_i$ in the sum with $\alpha(r_i)$:

$$
\hat{S}(M) = \sum_i \alpha(r_i)^2 = \sum_i \left[ \gamma \arctan\frac{L_i^D - G_i(M)}{\gamma \sigma_i} \right]^2.
$$

In practice $\Delta S_{\max}$ in the range $[10, 25]$ (so $\gamma$ between roughly $2$ and $3$) suppresses genuine outliers hard without biasing inlier-dominated regions. Since residuals are already sigma-scaled, this corresponds roughly to saying "residuals past $3$ to $5\sigma$ stop mattering".

In arael this is exactly what you see in the demo constraint bodies:

```rust
let plain_r = (a * e.x + b - e.y) / sigma;
gamma * atan(plain_r / gamma)
```

The symbolic-differentiation pipeline handles `atan`'s derivative automatically; from the macro's point of view the residual is just another expression. No special-case code, no outlier bookkeeping.

### Initialisation matters

Gauss-Newton (and Levenberg-Marquardt) is a local method: each step linearises the cost around the current $M$ and moves in the direction that linearisation suggests. For any loss, you need a starting $M_0$ close enough to the optimum that the linearisation is informative.

Starship makes this requirement stricter. The gradient falls off as $\alpha'(r) = 1 / (1 + \pi^2 r^2 / (4 \Delta S_{\max}))$, so at the recommended $\Delta S_{\max} = 25$ a residual at $5\sigma$ still carries about 29% of its least-squares pull and a $10\sigma$ residual about 9% -- still usable. Once you get out to $20\sigma$ and beyond it drops under 3% and those residuals are effectively frozen. If $M_0$ puts many residuals that far out, the solver has nothing to work with and stalls. The usual remedy is **graduated optimisation**: start with a large $\Delta S_{\max}$ (loose cap, everything in the informative regime), solve, then shrink it across passes down to the target value. The SLAM demo does this via a `frine_isigma_scale` field stepped per pass.

## Localization Demo

Same model as SLAM but landmarks are fixed (known map). Since landmark positions are not optimized, there is no gauge freedom and absolute pose errors are meaningful. No GPS needed -- the known landmarks anchor the solution.

The frine constraint uses a **remote block** (`pose.hb_pose`) -- the hessian block lives on Pose, not on PointFrine, since only Pose has parameters. With only pose parameters, the hessian is block-tridiagonal (kd=11 for 6-param poses), so the band solver can be used for O(n) scaling instead of O(n^3) dense -- 9.4x faster at 500 poses.

See [examples/loc_demo.rs](examples/loc_demo.rs).

## Examples

The `examples/` directory is the primary place to see the API in use. Each file is a runnable `cargo run --release --example <name>`.

- **[bench_band](examples/bench_band.rs)** -- benchmarks the band Cholesky backend against dense on the localisation model at increasing pose counts. Prints timing + speedup.
- **[bench_investigate](examples/bench_investigate.rs)** -- deeper comparison of sparse backends on SLAM, with assembly vs solve breakdown and numeric cross-check of the solutions.
- **[bench_sparse](examples/bench_sparse.rs)** -- sparse Cholesky backends (faer) vs dense on SLAM.
- **[calc_demo](examples/calc_demo.rs)** -- `bc`-style REPL calculator built on `arael-sym`. Shows `parse_with_functions` + `FunctionBag` for user-defined functions, persistent history via rustyline.
- **[jacobian_demo](examples/jacobian_demo.rs)** -- `#[arael(root, jacobian)]`, `#[arael(constraint_index)]`, and `calc_jacobian` / `calc_cost_table` walk-through. End-to-end reference for the instrumentation features used in convergence debugging.
- **[linear_demo](examples/linear_demo.rs)** -- robust linear regression on noisy 2D data. Residual wrapped in `gamma * atan(r / gamma)` -- the [Starship method (US12346118)](https://patents.google.com/patent/US12346118), same robustifier used by the feature constraints in loc/SLAM. Minimal single-struct model + LM fit, compared against plain closed-form least squares.
- **[loc_demo](examples/loc_demo.rs)** -- localisation with fixed known landmarks (no gauge freedom). Block-tridiagonal Hessian + band solver. Graduated-isigma optimisation via a root `frine_isigma_scale` field.
- **[loc_global_demo](examples/loc_global_demo.rs)** -- how to put `Param` fields on the root struct and have constraints consume them. Uses a system-global rigid transform (translation + 3-axis rotation applied to every pose) as the running example; every residual that reads the robot's world pose composes the globals before evaluating. Shows the two wiring shapes for pose<->root cross-Hessian pairs (`CrossBlock<Pose, Path>` on the constraint struct, and a root-owned `TripletBlock` named via the `root.<field>` block spec) and a `Path::optimise_center` pass that freezes pose params and optimises only the globals before the main sweep.
- **[m3500_demo](examples/m3500_demo.rs)** -- the classic M3500 Manhattan-world pose-graph benchmark (Olson 2006): 3500 SE2 poses, 5453 relative-pose constraints read from a g2o file (vendored under `benchmarks/pgo/datasets/`, any 2D g2o file works). Between-factor written symbolically in ~10 lines (`matrix2sym::rotation`, `rad_diff`), gauge fixed by a guarded soft prior on pose 0, solved with sparse faer LM in ~30 ms. `--weighted` applies the file's information matrices; writes `m3500.eps`, an overlay of the drifted odometry input vs the optimized grid. The same model backs the cross-solver comparison in [benchmarks/pgo](benchmarks/pgo/README.md).
- **[model_demo](examples/model_demo.rs)** -- minimal `#[arael::model]` walk-through showing how `Param`, `SimpleEulerAngleParam`, and the update cycle fit together.
- **[refs_demo](examples/refs_demo.rs)** -- `Ref<T>`, `refs::Vec`, `refs::Deque`, and `refs::Arena` behaviour: insertion, iteration, stable handles.
- **[runtime_fit_demo](examples/runtime_fit_demo.rs)** -- curve fitting where the residual equation is a string parsed at runtime. Demonstrates `ExtendedModel` + robust loss on top of the symbolic front end.
- **[single_root_demo](examples/single_root_demo.rs)** -- single-struct model-and-root + a direct-composed sub-model, each carrying its own `SelfBlock<Self>`. The smallest example that exercises the "root has its own params" path.
- **[slam2d_simple_demo](examples/slam2d_simple_demo.rs)** -- minimal pedagogical 2D SLAM: pose = (x, y, gamma), single forward-facing camera reporting a bearing per landmark, diagonal odometry covariance, first pose held fixed at the origin facing east via `optimize = false` on its params (the fixed reference that makes the solution unique -- all measurements are relative). Constraint bodies use the symbolic `matrix2sym::rotation(angle)` surface. After the LM fit it computes the parameter covariance from the dense Hessian and writes `slam2d_simple.eps` -- a colour-per-landmark plot of the trajectory, the bearing fan, ground truth shadows, the GT-vs-estimate error lines, and 95% confidence ellipses around each landmark (visibly elongated along the radial direction, illustrating that depth is the unobservable dimension of bearing-only SLAM).
- **[slam_demo](examples/slam_demo.rs)** -- synthetic visual-inertial SLAM: S-curve trajectory, 20 poses, 40 landmarks, odometry + tilt + GPS + feature observations. Full verbose-LM trace across graduated isigma passes -- the reference for what a healthy solver run looks like.
- **[sym_demo](examples/sym_demo.rs)** -- symbolic-math tour: expression building, automatic differentiation, CSE, pretty printing, parsing. No solver involvement; pure `arael-sym`.
- **[user_function_demo](examples/user_function_demo.rs)** -- `#[arael::function]` for user-defined operators in constraint bodies. Form A purely symbolic `sigmoid(x) = 1 / (1 + exp(-x))` and Form B opaque numerical `my_safe_asin` with a closed-form symbolic derivative, both used in a single two-residual LM fit.

## Solvers

Levenberg-Marquardt with pluggable linear-algebra backends. Full
reference: [docs/SOLVERS.md](docs/SOLVERS.md).

**Default to `solve_sparse_faer_f32` (or `solve_sparse_faer` for f64).**
For most real problems the Hessian is sparse enough for sparse
Cholesky to be the right choice, and `faer` is pure Rust, no
external dependency, and handles the full sparsity pattern of a
SLAM-like problem.

| Backend | When |
|---|---|
| **`solve_sparse_faer[_f32]`** | **default**. Any non-trivial problem |
| `solve[_f32]` (dense) | toy problems (≤ 4 parameters) |
| `solve_band[_f32]` | **only** when the Hessian is genuinely block-tridiagonal with a known half-bandwidth `kd` |

`LmConfig` controls the solve -- convergence tolerances, iteration
caps, initial lambda, and `verbose` (turn it on first when
debugging). Defaults are a safe middle ground; production solves
usually want `max_iters` and `rel_precision` tuned for the
performance/quality trade-off that actually matters for the
problem. See [docs/SOLVERS.md](docs/SOLVERS.md) for the full field
reference and a recipe for picking them.

## Runtime Differentiation

Compile-time differentiation generates optimized Rust code with CSE at build time -- ideal when the model structure is fixed. But many applications need equations that are only known at runtime: user-typed formulas in a CAD parametric dimension, configuration-driven curve fitting, or symbolic constraints loaded from a file.

Arael supports this through **runtime differentiation**: parse an equation string with `arael_sym::parse`, symbolically differentiate once at setup with `E::diff`, then evaluate the expression tree numerically each solver iteration. The `ExtendedModel` trait and `TripletBlock` provide the integration point with the LM solver.

The sketch editor (`arael-sketch`) uses this extensively for parametric expression dimensions -- a user can type `d0 * 2 + 3` as a dimension value, and the solver constrains the geometry to satisfy the equation in real time, with full symbolic derivatives.

```rust
// Parse equation at runtime, differentiate symbolically
let expr = arael_sym::parse("a * x + b").unwrap();
let residual = expr - arael_sym::symbol("y");
let dr_da = residual.diff("a");  // symbolic derivative w.r.t. a
let dr_db = residual.diff("b");  // symbolic derivative w.r.t. b

// In ExtendedModel::extended_compute64(params, grad) -- each solver iteration:
for &(x, y) in &data {
    vars.insert("x", x);
    vars.insert("y", y);
    let r = residual.eval(&vars)?;
    let dr = vec![dr_da.eval(&vars)?, dr_db.eval(&vars)?];
    // writes 2*r*dr into `grad` AND pushes upper-triangle Hessian
    // into the TripletBlock -- one call, both done
    hb.add_residual(r, &param_indices, &dr, grad);
}
```

The demo accepts an arbitrary equation from the command line:

```bash
cargo run --example runtime_fit_demo                            # default: y = a * x + b
cargo run --example runtime_fit_demo -- "a * x^2 + b * x + c"  # quadratic
cargo run --example runtime_fit_demo -- "a * sin(x * b) + c"   # sinusoidal
```

Full source: [examples/runtime_fit_demo.rs](examples/runtime_fit_demo.rs).

## Instrumentation and troubleshooting

### My solve doesn't converge. What do I check?

0. **Turn on solver verbose mode first.** Set `verbose: true` on `LmConfig` and every LM step prints cost, lambda, and the step outcome. On a Cholesky rejection the line also reports non-finite counts for grad / diagonal / cur_x / matrix and a count of non-positive diagonal entries -- four quick signals that narrow the problem before any deeper digging:

    ```rust,ignore
    let cfg = arael::simple_lm::LmConfig::<f32> { verbose: true, ..Default::default() };
    let result = arael::simple_lm::solve_sparse_faer_f32(&x0, &mut model, &cfg);
    ```

    A healthy pass looks like steady cost drops with rising / stabilising step sizes and no Cholesky rejections -- see [examples/slam_demo.rs](examples/slam_demo.rs) for a reference trace. If verbose already reports NaN / Inf or `diag<=0`, skip to steps 2 / 3 below; otherwise continue to the cost-by-label breakdown.

1. **Cost breakdown by label.** Name your constraint attributes with `#[arael(constraint(hb, name = "drift", { ... }))]` so each group shows up under its own label in the sum-of-squares. Call `model.calc_cost_table(&params)` for a `HashMap<&'static str, T>` and log it. A single label dominating the total is usually the culprit -- either an overly tight sigma, bad initial values for its inputs, or a constraint that's mathematically unsatisfiable.

2. **NaN or Inf residuals / derivatives.** The verbose-mode output from step 0 already tells you whether grad / matrix / params contain non-finite values at the failing step. If they do, walk `model.calc_jacobian(&params).rows` to find the specific row. A NaN residual or partial derivative usually means a `sqrt`, `acos`, `asin`, or `atan2` saw a degenerate input (zero-length vector, both-zero arguments, `|x| > 1` for asin/acos). `arael-sym` ships `safe_sqrt`, `safe_asin`, `safe_acos`, `safe_atan2` that clamp / regularise at the singular point and produce non-diverging derivatives. **Before reaching for them, prefer to redesign the constraint so the singularity can't be hit.** A `safe_*` wrapper hides the degeneracy from the solver and may leave the residual insensitive to the parameters that should drive it out of the singular region; an equivalent constraint formulated on the right geometric quantity avoids the singularity entirely. E.g. match 3D landmarks to features in 3D space (compare world-frame directions or positions) instead of projecting through a camera model and computing 2D image-plane residuals -- the 3D formulation is simpler, better conditioned, and has no pixel-wraparound / behind-camera pathology.

3. **Non-positive diagonal.** The verbose-mode `diag<=0: N` counter at a Cholesky rejection is the loudest possible signal that some parameter is untouched by every constraint (indices left at `u32::MAX`) or is receiving a negative contribution. Either outcome is a bug distinct from f32 accumulation noise.

4. **Gradient magnitude.** After `calc_grad_hessian_dense`, the maximum absolute gradient component should be small relative to the cost scale at a local minimum. A huge gradient with tiny cost means the parameter scaling is off -- one parameter moves cost several orders of magnitude more than another, which destabilises Levenberg-Marquardt.

5. **Hessian health.** The same `hessian` array should be finite and positive-semi-definite at a minimum (smallest eigenvalue ≥ 0 modulo roundoff). A significantly-negative smallest eigenvalue means the Gauss-Newton approximation `J^T J` is a poor local fit -- often because constraints are ill-conditioned or cancel.

6. **Stiffness.** Ratios between the smallest and largest sigmas (or between the smallest and largest eigenvalues of `J^T J`) that span many orders of magnitude make the problem numerically stiff. LM damping has to pick a lambda that suits both ends, which is hard at f32 precision. Keep isigmas comparable where you can; if a tight constraint dominates one direction, a gauge direction orthogonal to it will starve for signal. Starting with a loose scale and ramping up (graduated optimisation -- see `loc_demo` / `slam_demo` for the `frine_isigma_scale` pattern) helps LM climb a stiff problem without rejecting early steps.

7. **Simpler math beats clever math.** Reformulate residuals on the most natural geometric quantity. 3D direction / position errors are cheaper and better-conditioned than 2D reprojection errors; relative rotations compared as matrices or unit quaternions avoid Euler-angle gimbal lock; distances compared in squared form avoid `sqrt` derivatives near zero. Every nonlinear operation you remove is one less place for the residual / derivative to misbehave and one less source of stiffness.

8. **Inspect the generated code.** Use [`cargo expand`](https://github.com/dtolnay/cargo-expand) to see what the macro emitted for your constraint body -- see [Looking under the hood](#looking-under-the-hood-with-cargo-expand) below.

9. **Rank / DOF.** Call `Jacobian::singular_values` (or the full `Jacobian::svd` for directions). Near-zero singular values count the degrees of freedom. If this is higher than you expect, the model is under-constrained. The right singular vectors (columns of `SvdResult::v`) corresponding to σ ≈ 0 name the unconstrained parameter directions -- useful for identifying *which* parameters are free. SVD is always performed in f64 regardless of the model's element type, so rank detection stays reliable even for f32 models.

### Looking under the hood with `cargo expand`

Mastering arael means being able to read what the macros actually generated for your equations. `#[arael::model]` does a lot: it interprets the constraint body symbolically, differentiates it against every reachable parameter, runs common-subexpression elimination, and emits Rust code for three call paths (`__compute_blocks`, `__set_block_indices`, `calc_jacobian`). [`cargo expand`](https://github.com/dtolnay/cargo-expand) (`cargo install cargo-expand`) prints the expansion exactly as the compiler sees it.

```bash
cargo expand --example single_root_demo
# or, for your own crate:
cargo expand --lib              # library
cargo expand --bin my_bin       # binary
cargo expand my_mod::MyModel    # a specific path
```

### Example: a one-line fix constraint

The single-root demo declares

```rust,ignore
#[arael(constraint(hb, name = "fix_x", {
    [(singleroot.x - 3.0) * singleroot.isigma]
}))]
struct SingleRoot {
    x: Param<f64>,
    y: Param<f64>,
    isigma: f64,
    /* ... */
}
```

`cargo expand --example single_root_demo` shows the macro emits a `__compute_blocks` method with a block like:

```rust,ignore
/// arael: SingleRoot[fix_x] @ examples/single_root_demo.rs:28
let __r_0       = singleroot.isigma * (singleroot.x.work() - 3.0);
let __dr_0_0    = singleroot.isigma;      // d/d x
let __dr_0_1    = 0.0;                    // d/d y
__item.hb.add_residual(
    __r_0 as f64,
    &[__dr_0_0 as f64, __dr_0_1 as f64],
    grad,
);
```

Things to notice:

- `singleroot.x.work()` -- each param access is rewritten to `work()` so the LM trial step is used in place of the stored value without mutating it.
- Derivatives for every param the constraint touches appear individually (`__dr_0_0`, `__dr_0_1`). The `0.0` entry for `y` is not elided because the index into `hb` is positional; dead rows fold out at optimisation time.
- The residual and the partials flow into the entity's Hessian block via `hb.add_residual(r, dr, grad)` -- one call per residual, accumulating `2*r*dr` into `grad` and `2*dr_i*dr_j` into the block's packed upper triangle.
- The `/// arael: ...` doc comment is a source marker pointing at the constraint attribute the block came from -- invaluable when the expansion runs to thousands of lines.

### Example: shared subexpressions

In a larger body -- say a landmark observation that builds a rotation matrix and reuses it across x/y/z residuals -- the macro runs CSE before emitting code, so you see lines like

```rust,ignore
let __cse_0 = cos(pose.ea.z.work());
let __cse_1 = sin(pose.ea.z.work());
let __cse_2 = __cse_0 * (lm.pos.x - pose.pos.x.work())
            + __cse_1 * (lm.pos.y - pose.pos.y.work());
// __cse_2 reused in __r_0, __r_1, and every __dr_* that needs it
```

Reading these tells you what the compiler *actually* has to evaluate -- useful for understanding the cost of a constraint, spotting accidental non-shared work, and sanity-checking that symbolic simplification collapsed things you expected it to.

### What to look for

- **`__set_block_indices`** -- where each `SelfBlock` / `CrossBlock` / `TripletBlock` gets its global parameter indices written into place. A block that isn't touched here is invisible to the solver (its `u32::MAX` sentinel causes every `add_residual` to silently skip) -- a common failure mode.
- **`__compute_blocks`** -- the grad + block-Hessian accumulation path. Each constraint is a nested block with its own CSE'd body.
- **`calc_jacobian`** -- same body structure but builds a `JacobianRow` per residual instead of accumulating into the blocks. Generated only when you declare `#[arael(root, jacobian)]`.
- **source markers** -- doc comments like `/// arael: PointFrine[<name>] @ path/to/file.rs:NNN` pinpoint the constraint attribute each block came from.

Expansion grows quickly (the single-root demo is ~800 lines; a full SLAM model is several thousand). Use `sed -n` or a pager scoped to the method you care about:

```bash
cargo expand --example slam_demo | sed -n '/fn __compute_blocks/,/^    fn /p'
```

## 2D Sketch Editor

An interactive constraint-based 2D sketch editor built on the arael optimization framework. Draw geometry, apply constraints, and the solver keeps everything consistent in real time.

[![Sketch Editor](docs/sketch.png)](https://sketch.mare.ee/)

[Try it in the browser](https://sketch.mare.ee/)

The sketch solver combines both differentiation modes:

- **Geometric constraints** (horizontal, coincident, parallel, tangent, etc.) use **compile-time differentiation** -- the macro generates optimized Gauss-Newton code with CSE for each constraint type.
- **Parametric dimensions** use **runtime differentiation** -- the user types an expression like `d0 * 2 + 3` as a dimension value, and the solver parses it, differentiates symbolically, and constrains the geometry to satisfy the equation in real time. Dimensions can reference each other, entity properties (`L0.length`, `A0.radius`), and arithmetic expressions. Broken references (deleted entities) are detected and the dimension falls back to its last computed value.

This makes the sketch editor a fully parametric constraint solver where changing one dimension propagates through all dependent expressions.

### Running (native)

```bash
cargo run -r -p arael-sketch
```

### Running (browser)

The sketch editor compiles to WebAssembly and runs in the browser.
Requires [trunk](https://trunkrs.dev/) (`cargo install trunk`) and the
`wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`):

```bash
cd arael-sketch
trunk build --release
python3 -m http.server -d dist 8080
# Open http://localhost:8080
```

### Tools

- **Line (L)**, **Circle (O)**, **Arc (A)**, **Point (P)** -- draw geometry with auto-snap to nearby points, endpoints, and curves
- **Dimension (D)** -- add length, distance, radius, angle, and point-to-line distance dimensions with draggable annotations. Supports numeric values and parametric expressions (`d0 * 2`, `L0.length + 3`).
- **Select (S)** -- click to select, drag to move entities, Backspace/Delete to remove
- **Dark/Light mode** toggle, **Save/Load** (JSON), **Undo/Redo** (Ctrl+Z/Ctrl+Shift+Z)

### Constraints

Horizontal (H), Vertical (V), Coincident (C), Parallel, Perpendicular, Equal length/radius, Tangent (T), Collinear, Midpoint (M), Symmetry (lines or points about a mirror line), Lock (K), Line style (X). Constraints are visualized as symbols on the geometry and can be selected and deleted.

### Example: Sketch Solver API

```rust
use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch::*;

let mut sketch = Sketch::new();

// Create a rectangle from 4 lines
let bottom = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.1));
let right  = sketch.add_line(vect2d::new(3.1, 0.0), vect2d::new(3.0, 2.1));
let top    = sketch.add_line(vect2d::new(2.9, 2.0), vect2d::new(0.1, 1.9));
let left   = sketch.add_line(vect2d::new(0.0, 2.1), vect2d::new(0.1, 0.1));

// Horizontal/vertical constraints
sketch.lines[bottom].constraints.horizontal = true;
sketch.lines[top].constraints.horizontal = true;
sketch.lines[left].constraints.vertical = true;
sketch.lines[right].constraints.vertical = true;

// Connect corners (a.p2 == b.p1)
sketch.coincident_ll21.push(CoincidentLL21 { a: bottom, b: right, hb: CrossBlock::new() });
sketch.coincident_ll21.push(CoincidentLL21 { a: right, b: top, hb: CrossBlock::new() });
sketch.coincident_ll21.push(CoincidentLL21 { a: top, b: left, hb: CrossBlock::new() });
sketch.coincident_ll21.push(CoincidentLL21 { a: left, b: bottom, hb: CrossBlock::new() });

// Fix bottom-left corner and set dimensions
sketch.lines[bottom].p1 = arael::model::Param::fixed(vect2d::new(0.0, 0.0));
sketch.lines[bottom].constraints.has_length = true;
sketch.lines[bottom].constraints.length = 4.0;
sketch.lines[left].constraints.has_length = true;
sketch.lines[left].constraints.length = 2.0;

// Solve -- all constraints satisfied simultaneously
sketch.solve();
// bottom: (0,0)->(4,0), right: (4,0)->(4,2), top: (4,2)->(0,2), left: (0,2)->(0,0)
```

The sketch solver uses Levenberg-Marquardt optimization with drift regularization and robust drag constraints. Geometric constraints are differentiated at compile time; parametric expression dimensions use runtime differentiation via `ExtendedModel`.

### Command Panel & Scripting

Press `/` to open the command panel. Full scripting support with 40+ commands for geometry creation, constraints, dimensions, parameters, introspection, and view control. Commands support expressions, coordinate references (`L0.p2`, `@dx,dy`), geometric functions (`midpoint(L0)`, `intersect(L0,L1)`), and vector arithmetic (`L0.p2 + normal(L0) * 3`).

See [arael-sketch-backend/docs/COMMANDS.md](arael-sketch-backend/docs/COMMANDS.md) for the full command reference.

### AI Agent Integration (MCP)

The sketch editor embeds an MCP (Model Context Protocol) server, enabling AI agents like Claude Code to create and modify sketches programmatically. The AI sends sketch commands and reads state through the standard MCP tool interface.

![Dark mode with AI-drawn geometry](arael-sketch/docs/dark.png)

*Dark mode with parameters panel, command history showing MCP agent connection, and geometry drawn by Claude Code.*

Start the editor with MCP enabled:
```bash
cargo run -r -p arael-sketch -- --mcp --mcp-allow-all
```

The `--mcp-allow-all` flag auto-approves OAuth connections from AI agents (recommended for local use). Without it, connections require manual approval in the GUI (not yet implemented).

Configure Claude Code (`~/.claude.json`):
```json
{
  "mcpServers": {
    "arael-sketch": {
      "type": "http",
      "url": "http://127.0.0.1:8585/mcp"
    }
  }
}
```

The MCP server exposes tools for executing sketch commands (`execute_command`, `execute_script`), querying state (`get_sketch_state`), and reading documentation (`get_help`). The `initialize` response includes a condensed command reference that the AI loads into context automatically. File operations (`save`, `load`) are blocked for security.

See [arael-sketch/](arael-sketch/) for the full implementation.

## Project Structure

```
arael/              Main library
  src/
    model.rs        Param<T>, Model trait, SelfBlock, CrossBlock, TripletBlock
    simple_lm.rs    LM solver, LmSolver trait, Dense/Band/Sparse backends, CooMatrix, CscMatrix
    refs.rs         Type-safe Vec<T>, Deque<T>, Arena<T>, Ref<T>
    vect.rs         vect2<T>, vect3<T>
    matrix.rs       matrix2<T>, matrix3<T>
    quatern.rs      quatern<T>
  cpp/
    eigen_sparse.cpp  Eigen SimplicialLLT + CHOLMOD FFI bridge (optional)

arael-sym/          Symbolic math library
  src/
    lib.rs          E type, constructors, operators
    diff.rs         Symbolic differentiation
    simplify.rs     Algebraic simplification
    cse.rs          Common subexpression elimination
    eval.rs         Evaluation, substitution, free variables
    fmt.rs          Display, LaTeX, Rust code generation
    geo.rs          Symbolic vectors/matrices (vect3sym, matrix3sym)
    linalg.rs       SymVec, SymMat, Jacobian
    parse.rs        Expression parser

arael-macros/       Procedural macros
  src/
    lib.rs          #[arael::model], sym!, field rewriting
    constraint.rs   Constraint code generation, CSE integration

arael-sketch-solver/ 2D constraint solver library
  src/
    lib.rs          Sketch root, solve(), entity management
    entities.rs     Point, Line, Arc types
    constraints.rs  40+ cross-constraint types
    dimensions.rs   Dimension annotations

arael-sketch/       Interactive sketch editor application
  src/
    main.rs         Entry points, EditorApp, core logic
    actions.rs      Action enum, undo-able operations
    history.rs      Undo/redo system
    tools.rs        Tool modes, selection, constraint types
    drawing.rs      Canvas rendering, grid, dimensions
    colors.rs       Color scheme (light/dark)
    geometry.rs     Coordinate transforms, snapping
```

## License

See [LICENSE.md](LICENSE.md).
