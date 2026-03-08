# SLAM Path Optimization

This example demonstrates a visual-inertial SLAM backend: jointly optimizing
robot poses and landmark positions from camera observations, GPS, wheel
odometry, and accelerometer tilt readings. The focus is on the optimization
(nonlinear least squares with symbolic differentiation); landmark discovery,
feature detection, and data association are outside the scope of this example
and are simulated with synthetic data.

The demo generates a synthetic S-curve trajectory with 60 poses and 240
point landmarks at 5-30m distance. 5 cameras provide 360-degree coverage.
Each landmark is visible from its anchor pose +/- 15 poses (75% probability),
creating a realistic sparse hessian structure. 50% of feature associations
are completely wrong (outliers with 30x pixel noise) -- the robust
`gamma * atan(r/gamma)` suppression handles this via graduated optimization.

The solver uses faer sparse Cholesky (pure Rust) to exploit the hessian
sparsity, with optional Eigen SimplicialLLT and CHOLMOD backends via C++ FFI.
Command-line arguments allow choosing the solver and problem size:

```bash
cargo run -r --example slam_demo -- --solver faer --poses 100 --landmarks 400
```

## Gauge Freedom

Without absolute reference, visual SLAM has 6 degrees of gauge freedom
(3 translation + 3 rotation). The entire solution can shift/rotate
arbitrarily without changing feature reprojection costs. Each sensor
fixes specific degrees of freedom:

- **Tilt sensor (accelerometer)** fixes roll (ea.x) and pitch (ea.y),
  making the path level. Without it the solution can tilt arbitrarily.
- **GPS** fixes translation and yaw (ea.z). Even with a systematic offset
  (all readings biased by ~2.5m), it still constrains yaw because the
  offset is constant while poses move -- relative GPS positions define
  heading.
- **Odometry** constrains relative motion between consecutive poses.
- **Drift constraints** are weak regularizers (sigma=1000m pos, 1800deg ea)
  preventing parameters from diverging during early optimization passes
  when feature constraints are scaled down.

## Hessian Sparsity

![Hessian Sparsity](sparsity.png)

The sparsity pattern shows the structure of a SLAM hessian: pose-pose blocks in the upper-left (odometry coupling consecutive poses), pose-landmark coupling in the off-diagonal rectangles (each landmark visible from nearby poses only), and landmark self-blocks along the lower-right diagonal. Sorting landmarks by anchor pose would produce an arrowhead pattern; here they are in insertion order. Sparse Cholesky exploits this structure for large speedups over dense solvers.

## Problem Structure

```
Path (root)
  poses: Deque<Pose>               -- 60 poses, 6 params each (pos + ea)
    pos: Param<vect3f>              -- 3D position (optimizable)
    ea: SimpleEulerAngleParam<f32>  -- euler angles: x=roll, y=pitch, z=yaw
    info: PoseInfo                  -- odometry, GPS, tilt, features
    hb_pose: SelfBlock<Pose>        -- GPS + drift + tilt hessian block
  landmarks: Vec<PointLandmark>     -- 240 landmarks, 3 params each
    pos: Param<vect3f>              -- 3D position (optimizable)
    frines: Vec<PointFrine>         -- observations linking to poses
      pose: Ref<Pose>               -- which pose observed this
      feature: Ref<PointFeature>    -- feature data
      hb: CrossBlock<PointLandmark, Pose>
    hb_drift: SelfBlock<PointLandmark>  -- landmark drift regularizer
  pose_pairs: Vec<PosePair>        -- 59 consecutive pose pairs
    prev: Ref<Pose>                 -- previous pose
    cur: Ref<Pose>                  -- current pose
    hb: CrossBlock<Pose, Pose>      -- odometry hessian block
  gamma: f32                        -- robust suppression parameter
  drift_pos_isigma: f32             -- pose position drift (1/1000m)
  drift_ea_isigma: f32              -- pose ea drift (1/deg2rad(1800))
  drift_lm_isigma: f32              -- landmark drift (1/1000m)
  tilt_isigma: f32                  -- tilt sensor (1/deg2rad(0.25))
```

Total: 1080 parameters (60 * 6 + 240 * 3). Configurable via `--poses` and `--landmarks`.

## Constraints

### GPS (guarded, on Pose)

Whitened GPS residual with robust suppression. Guarded -- only evaluated
when GPS data is present (`Option<GpsData>`):

```rust
#[arael(constraint(hb_pose, guard = self.info.gps.is_some(), {
    let gamma = path.gamma;
    let raw = pose.pos - pose.info.gps.pos;
    let rt_raw = pose.info.gps.cov_r.transpose() * raw;
    let p0 = rt_raw.x * pose.info.gps.cov_isigma.x;
    let p1 = rt_raw.y * pose.info.gps.cov_isigma.y;
    let p2 = rt_raw.z * pose.info.gps.cov_isigma.z;
    [gamma * atan(p0 / gamma),
     gamma * atan(p1 / gamma),
     gamma * atan(p2 / gamma)]
}))]
```

GPS covariance is decomposed as `K = R * diag(d) * R^T`, allowing
whitening: `whitened = diag(1/sqrt(d)) * R^T * raw`.

### Drift (on Pose and PointLandmark)

Weak regularizer anchoring parameters near their current `.value`.
Prevents divergence during early graduated passes:

```rust
#[arael(constraint(hb_pose, {
    let pos_drift = pose.pos - pose.pos_value;
    let ea_drift = pose.ea - pose.ea_value;
    [pos_drift.x * path.drift_pos_isigma,
     pos_drift.y * path.drift_pos_isigma,
     pos_drift.z * path.drift_pos_isigma,
     ea_drift.x * path.drift_ea_isigma,
     ea_drift.y * path.drift_ea_isigma,
     ea_drift.z * path.drift_ea_isigma]
}))]
```

### Tilt (on Pose)

Accelerometer constrains roll and pitch directly:

```rust
#[arael(constraint(hb_pose, {
    [(pose.ea.x - pose.info.tilt_roll) * path.tilt_isigma,
     (pose.ea.y - pose.info.tilt_pitch) * path.tilt_isigma]
}))]
```

### Feature observation (on PointFrine)

Links a landmark to a pose through a camera observation.
`parent=lm` names the parent PointLandmark variable:

```rust
#[arael(constraint(hb, parent=lm, {
    let gamma = path.gamma;
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - feature.camera_pos;
    let r_f = feature.mf2r.transpose() * r_r;
    let plain1 = atan2(r_f.y, r_f.x) * feature.isigma.x;
    let plain2 = atan2(r_f.z, r_f.x) * feature.isigma.y;
    let err1 = gamma * atan(plain1 / gamma);
    let err2 = gamma * atan(plain2 / gamma);
    [err1, err2]
}))]
```

Steps:
1. Transform landmark into robot frame: `lm_r = R^T * (lm - pos)`
2. Subtract camera offset: `r_r = lm_r - camera_pos`
3. Transform to feature frame: `r_f = mf2r^T * r_r`
4. Angular residuals: `atan2(r_f.y, r_f.x)` and `atan2(r_f.z, r_f.x)`
5. Weight by pixel angular resolution (`isigma`)
6. Robust suppression: `gamma * atan(plain / gamma)`

The feature frame `mf2r` has col0 = view direction from pose toward
landmark, col1/col2 = perpendicular axes for horizontal and vertical
angular error measurement.

### Odometry (on PosePair)

Constrains relative motion between consecutive poses:

```rust
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - cur.info.delta_pos;
    let pos_w = cur.info.delta_pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * cur.info.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = cur.info.delta_ea_cov_r.transpose() * ea_err;
    [pos_w.x * cur.info.delta_pos_cov_isigma.x,
     pos_w.y * cur.info.delta_pos_cov_isigma.y,
     pos_w.z * cur.info.delta_pos_cov_isigma.z,
     ea_w.x * cur.info.delta_ea_cov_isigma.x,
     ea_w.y * cur.info.delta_ea_cov_isigma.y,
     ea_w.z * cur.info.delta_ea_cov_isigma.z]
}))]
```

## Graduated Optimization

Feature isigma is scaled across multiple passes to avoid local minima:

1. **Pass 1** (scale=0.01): Features contribute almost nothing. GPS,
   odometry, tilt, and drift dominate. Finds roughly correct poses.
2. **Pass 2** (scale=0.1): Features start pulling. Outliers are suppressed
   by the robust cost function.
3. **Pass 3** (scale=1.0): Full feature weight. Fine-tunes the solution.
   Outliers are deep in the atan saturation region.

## Uncertainty Estimation

After optimization, the demo estimates landmark position uncertainty from the
inverse of the Gauss-Newton Hessian. The Hessian `H = 2 * J^T * J` is the
information matrix (the factor of 2 comes from our cost function `sum(r^2)`
without the conventional 1/2 prefactor). The parameter covariance matrix is:

```
Cov = (J^T * J)^{-1} = 2 * H^{-1}
```

### Global vs relative uncertainty

The raw diagonal block of `Cov` for a landmark gives its **global** position
uncertainty, which includes the gauge uncertainty (GPS offset, yaw) shared
by all parameters. This gauge floor dominates and makes all landmarks look
equally uncertain -- not useful.

The demo instead computes uncertainty **relative to the closest pose**, which
cancels the shared gauge. Given the full covariance with blocks `C_ll`
(landmark), `C_pp` (pose position), and `C_lp` (cross-covariance):

```
Cov_rel = C_ll - C_lp - C_pl + C_pp
```

The eigendecomposition of this 3x3 `Cov_rel` yields three eigenvalues whose
square roots are the semi-axis lengths of the **1-sigma uncertainty ellipsoid**
-- an ellipsoid describing the uncertainty of the landmark's position relative
to the nearby pose. The three eigenvectors give the principal directions of
uncertainty. Printed as `sigma=(a, b, c)` in descending order:

```
LM 227: |d|=0.017m  rel=0.30%  dist=5.8m  sigma=(0.417,0.059,0.006)m  frines=26
LM 224: |d|=0.054m  rel=0.21%  dist=25.9m  sigma=(1.876,0.319,0.020)m  frines=23
```

- The **largest** axis is depth (radial distance from observing cameras),
  the hardest direction to constrain from angular measurements alone.
  It scales roughly with landmark distance.
- The **middle** axis reflects the angular baseline of observations.
- The **smallest** axis is well-constrained by many observations from
  different poses.
- More observations (`frines`) and wider baselines produce smaller ellipsoids.

The dense Hessian inverse is O(n^3) and practical for the demo problem size
(n=1080). For larger problems, sparse selected-inversion or Schur complement
methods would be needed.

## Compile-Time Pipeline

```
Constraint expression (user writes)
    |
    v
Symbolic interpretation (proc macro interprets body)
    |
    v
Symbolic differentiation (d/d(each param))
    |
    v
Euler angles substitution (sin/cos -> precomputed fields)
    |
    v
Common Subexpression Elimination (iterative, factor-aware)
    |
    v
Rust code generation (precedence-aware, minimal parens)
    |
    v
Compiled evaluate code (inlined in calc_cost / calc_grad_hessian)
```

## Key Macro Attributes

| Attribute | Purpose |
|-----------|---------|
| `#[arael::model]` | Generate Model trait impl, Sym companion, PARAM_COUNT |
| `#[arael(root)]` | Trigger constraint code generation on root struct (f64 default) |
| `#[arael(root, f32)]` | Root with f32 precision (blocks, solver, all f32) |
| `#[arael(constraint(block, ...))]` | Define constraint expression on a struct |
| `#[arael(ref = path)]` | Auto-resolve Ref fields from a collection |
| `SimpleEulerAngleParam<T>` | Euler angles with precomputed sin/cos and rotation matrix |
| `EulerAngleParam<T>` | Gimbal-lock-free euler angles (delta around reference rotation) |
| `#[arael(skip)]` | Skip field in serialization |
| `guard = expr` | Conditional constraint evaluation |
| `parent = name` | Name the parent variable in constraint body |

## Viewing Generated Code

Use `cargo expand` to inspect the code the macros produce:

```bash
cargo expand --example slam_demo 2>&1 | less
```

The generated `calc_cost` method traverses all constraint collections,
evaluating CSE-optimized compiled code. Here is a simplified fragment
from the tilt constraint on Pose (the actual output has more constraints
and CSE intermediates):

```rust
fn calc_cost(&mut self, params: &[f64]) -> f64 {
    arael::model::Model::update64(self, params);
    let __self_ref = unsafe { &*(self as *const Self) };
    let mut __cost = 0.0 as f64;

    // Tilt constraint loop (simplified):
    for __item in self.poses.iter_mut() {
        let pose = &*__item;
        let path = &*__self_ref;
        let __r_0 = (pose.ea.work().x - pose.info.tilt_roll)
            * path.tilt_isigma;
        let __r_1 = (pose.ea.work().y - pose.info.tilt_pitch)
            * path.tilt_isigma;
        __cost += __r_0 * __r_0 + __r_1 * __r_1;
    }

    // Odometry constraint loop (with CSE intermediates):
    for __frine in self.pose_pairs.iter() {
        let prev = &self.poses[__frine.prev];
        let cur = &self.poses[__frine.cur];
        let path = &*__self_ref;
        // CSE: shared subexpressions extracted across all 6 residuals
        let __x19 = cur.pos.work().x - prev.pos.work().x;
        let __x20 = cur.pos.work().z - prev.pos.work().z;
        let __x21 = cur.pos.work().y - prev.pos.work().y;
        // Precomputed rotation matrix used directly:
        let __x6 = __x19 * prev.ea.rotation_matrix[0].x
            + __x21 * prev.ea.rotation_matrix[1].x
            - __x20 * prev.ea.sincos.0.y
            - cur.info.delta_pos.x;
        // ... more intermediates, residuals, cost accumulation
    }

    __cost
}
```

The `calc_grad_hessian_*` methods follow the same structure but also
compute symbolic derivatives (with CSE applied across both residuals
and their Jacobians) and accumulate them into `SelfBlock`/`CrossBlock`
hessian blocks via `add_residual()`.

## Full Source

See [examples/slam_demo.rs](../examples/slam_demo.rs).
