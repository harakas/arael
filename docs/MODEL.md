# Model Structure

This is the reference for every piece that appears in an `#[arael::model]`
declaration: parameter types, Hessian blocks, collection types, macro
attributes (struct-level, constraint-level, field-level), and the
patterns for placing constraints.

For an end-to-end walk-through see
[examples/single_root_demo.rs](../examples/single_root_demo.rs) (the
smallest complete model) and
[examples/slam_demo.rs](../examples/slam_demo.rs) (a full SLAM setup).

## Parameter types

Every field the solver is allowed to move during the solve must be a
parameter type. A field declared with a plain scalar / vector type is
treated as a constant.

| Type | Size | Use it when |
|---|---|---|
| `Param<f32>` / `Param<f64>` | 1 | scalar parameter |
| `Param<vect2<T>>` | 2 | 2D point / direction |
| `Param<vect3<T>>` | 3 | 3D position, velocity, linear vector |
| `SimpleEulerAngleParam<T>` | 3 | three independent Euler angles (roll, pitch, yaw) stored directly |
| `EulerAngleParam<T>` | 3 | "universal" Euler angles: parameters are a delta composed with a fixed reference rotation, avoiding parameterisation singularities for large-angle motion |

```rust,ignore
#[arael::model]
struct Pose {
    pos: Param<vect3f>,              // 3 scalar params
    ea: SimpleEulerAngleParam<f32>,  // 3 scalar params (roll, pitch, yaw)
    // info: plain non-Param data (sigmas, measurements) is fine
    info: PoseInfo,
    hb_pose: SelfBlock<Pose, f32>,   // mandatory; see below
}
```

Each parameter stores a current `value` and a per-iteration `work`
copy. The macro rewrites `pose.ea` in a constraint body to
`pose.ea.work()` so the LM trial step is evaluated without mutating
the stored value until the step is accepted.

### Initial values via `_value` suffix

Inside a constraint body, `pose.pos_value` (any `<field>_value`) resolves
to the original stored value of the parameter, i.e. the point the LM
trial-step is measured *against* -- not the trial step itself. Use it
to build drift / regularising residuals:

```rust,ignore
let pos_drift = pose.pos - pose.pos_value;
[pos_drift.x * path.drift_pos_isigma,
 pos_drift.y * path.drift_pos_isigma,
 pos_drift.z * path.drift_pos_isigma]
```

The drift measures how far the solver has pushed `pose.pos` away from
the seed the caller provided.

## Parameter control

Three ways to keep a parameter out of the solve:

1. **`Param::fixed(v)` at construction** -- immutable; the macro never
   emits indices for it. Typical use: problem-wide constants that are
   shaped like parameters (e.g. a known camera pose).
2. **Mutate `.optimize` at runtime** -- `pose.pos.optimize = false;`
   freezes a live parameter for the next solve call. Flip it back on
   to re-include. Use for staged optimisation (freeze subset, solve,
   unfreeze, solve again).
3. **The `_value` trick described above** -- the parameter still moves
   but the residual is anchored to its initial position via a drift
   constraint.

```rust,ignore
// (1) fixed at construction
let camera = Pose {
    pos: Param::fixed(known_position),     // never optimised
    ea:  Param::fixed(known_orientation),
    /* ... */
};

// (2) runtime freeze for a staged solve
for pose in path.poses.iter_mut() {
    pose.pos.optimize = false;
    pose.ea.optimize = false;
}
// ...solve only the remaining live params...
for pose in path.poses.iter_mut() {
    pose.pos.optimize = true;
    pose.ea.optimize = true;
}
```

See `Path::optimise_center` in
[examples/loc_global_demo.rs](../examples/loc_global_demo.rs) for a
real usage of runtime `.optimize = false` -- it freezes every pose,
solves only for the root-level global rigid transform, bakes the
result into the poses, then un-freezes.

## Hessian block types

The full Gauss-Newton Hessian is a **symmetric** block matrix, with
one block per (entity, entity) pair in the parameter vector. The
block at position `(Ei, Ej)` is the `NEi × NEj` matrix of second
partials; by symmetry `H[Ei, Ej] = H[Ej, Ei]^T`. arael stores each
unique block once and lets the accumulator fill in the transpose
when assembling into a dense / band / sparse matrix. Every
constraint that couples a given pair adds its `2 * dr_i * dr_j`
contribution to the same block:

- **Diagonal blocks (`Ei == Ej`)** live in each entity's
  `SelfBlock<Ei>` and are symmetric; only the upper triangle is
  stored. Every constraint touching `Ei`'s params writes there
  additively.
- **Off-diagonal blocks (`Ei != Ej`)** live in a `CrossBlock<Ei, Ej>`
  or in a `TripletBlock` that covers the pair. One `CrossBlock<A, B>`
  covers both `H[A, B]` and its transpose `H[B, A]` -- the
  accumulator writes both halves from the single stored rectangle.

Gradient contributions `2 * r * dr` go directly into the LM-provided
global gradient slice -- not into any block. Only Hessian entries
are stored block-wise.

Pick the block shape that matches the constraint body's parameter
reach:

| Type | Stores | Pick it when |
|---|---|---|
| **`SelfBlock<T>`** | grad + upper-triangular Hessian for entity T's own params | **mandatory on every params-having struct.** Holds the per-entity gradient and the (T, T) block of the Hessian |
| **`CrossBlock<A, B>`** | rectangular (A, B) cross Hessian only | **default for cross-entity Hessian pairs.** Packed in-place writes, cheap to assemble. One entry per unordered (A, B) entity pair in a constraint; (A, A) / (B, B) diagonals stay on each entity's SelfBlock |
| **`TripletBlock<T>`** | COO across-entity pairs | **always placed on the root** (declare one `hbt: TripletBlock<T>` on the root struct; constraints reach it via the `root.<field>` block spec). Two canonical uses: (1) the root has its own `Param` fields and constraints couple entity params with root params -- the (entity, root) cross pair lives in the root's TripletBlock; (2) runtime-parsed residuals via `ExtendedModel` that can't enumerate per-pair CrossBlocks statically -- `extended_compute*` writes into the root's TripletBlock directly. Never on a non-root struct. **Noticeably slower to assemble** than a multi-CrossBlock because every entry is a `Vec` push |

`SelfBlock<Self>` is required on every Model that has parameters --
failing to declare it is a compile-time error. Grad and diagonal
writes always land on each entity's `SelfBlock`; `CrossBlock` and
`TripletBlock` are cross-only storage.

```rust,ignore
// Entity with its mandatory SelfBlock.
#[arael::model]
struct Pose {
    pos: Param<vect3f>,
    ea:  SimpleEulerAngleParam<f32>,
    hb_pose: SelfBlock<Pose, f32>,   // required
}

// Constraint struct linking two entities via a CrossBlock.
#[arael::model]
#[arael(constraint(hb, { /* residuals involving prev and cur */ }))]
struct PosePair {
    #[arael(ref = root.poses)] prev: Ref<Pose>,
    #[arael(ref = root.poses)] cur:  Ref<Pose>,
    hb: CrossBlock<Pose, Pose, f32>, // only the (prev, cur) cross block
}
```

### Picking between multi-CrossBlock and TripletBlock

For N-entity residuals the macro accepts two shapes:

- **`constraint([hb_ab, hb_ac, hb_bc], { ... })`** -- one
  `CrossBlock<A, B>` field per unordered entity pair on the
  constraint struct. Packed rectangular storage, one
  `add_residual_cross` per pair.
- **`constraint(..., root.hbt, { ... })`** -- route across-entity
  pairs into a root-owned `TripletBlock<T>`. One COO accumulator on
  the root absorbs cross pairs from every constraint that
  references it. **The `TripletBlock` always lives on the root**
  -- don't put one on a constraint struct or an entity struct; the
  macro's `root.<field>` block spec is the only correct way to
  reach a `TripletBlock`.

```rust,ignore
// Multi-CrossBlock: explicit Hessian pair per unordered entity pair.
#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], { /* 3-line residual */ }))]
struct SymmetryLL {
    #[arael(ref = root.lines)] a: Ref<Line>,
    #[arael(ref = root.lines)] b: Ref<Line>,
    #[arael(ref = root.lines)] c: Ref<Line>,
    #[arael(cross = (a, b))] hb_ab: CrossBlock<Line, Line>,
    #[arael(cross = (a, c))] hb_ac: CrossBlock<Line, Line>,
    #[arael(cross = (b, c))] hb_bc: CrossBlock<Line, Line>,
}

// Root-owned TripletBlock: one COO accumulator on the root,
// referenced by constraints that couple an entity with root
// params (or where a per-pair CrossBlock layout doesn't fit).
#[arael::model]
#[arael(root)]
struct Path {
    poses: refs::Deque<Pose>,
    /* ... */
    hb:  SelfBlock<Path, f32>,
    hbt: TripletBlock<f32>,   // shared across-entity accumulator
}

#[arael(constraint(hb_pose, root.hbt, { /* residual touching pose + root */ }))]
struct Pose { /* ... hb_pose: SelfBlock<Pose, f32> ... */ }
```

**Prefer multi-CrossBlock whenever the set of cross-pairs is fixed
and dense.** TripletBlock carries a significant Hessian-assembly
penalty: every cross entry is a `Vec<(u32, u32, T)>` push (with
growth and no locality), vs CrossBlock's in-place write into a
pre-sized `NA * NB` rectangle at a known offset. The same N-entity
constraint assembles substantially faster through multi-CrossBlock
than through a TripletBlock, and the rectangular layout is also
friendlier to the CSC factorisation step that follows.

Reach for the root-owned TripletBlock in two canonical situations:

1. **The root has its own `Param` fields** and constraints couple
   per-entity params with root params. The (entity, root) cross pair
   has to live somewhere; a dedicated `CrossBlock<Entity, Root>` per
   entity type is verbose and scatters the cross storage, so the
   root TripletBlock is the clean place for it. The `loc_global_demo`
   example uses this: `hbt: TripletBlock<f32>` on `Path` absorbs
   every pose-to-globals cross pair emitted by the tilt and related
   constraints.
2. **Runtime-parsed residuals via `ExtendedModel`**. When the
   residual body is a user-supplied expression parsed at runtime,
   the macro cannot enumerate per-pair CrossBlocks statically.
   `ExtendedModel::extended_compute*` writes directly into the
   root's TripletBlock instead -- see
   [examples/runtime_fit_demo.rs](../examples/runtime_fit_demo.rs).

In both cases the triplet lives on the root, not on a constraint
struct.

**Caveat for case 1 -- root-level `Param`s destroy sparsity.** Every
constraint that reads a root param introduces an (entity, root)
cross pair in the Hessian. If *many* constraints read the same root
param -- which is the whole point of "global" root params -- the
root's rows and columns in the Hessian become dense (coupled to
every entity that touches them). Sparse Cholesky's fill-in grows
accordingly and solve times suffer. Use root `Param`s only when the
quantity is genuinely system-wide. Two canonical examples:

- **Frame corrections** -- rigid translation + rotation applied to
  *every* pose (the `loc_global_demo` pattern).
- **Global calibration** -- one-per-sensor quantities referenced by
  every observation from that sensor: camera intrinsics
  (`fx`, `fy`, `cx`, `cy`), lens-distortion coefficients, IMU bias
  and scale factors, magnetometer declination, barometric altitude
  reference. These live on the root naturally because there's one
  of them for the whole problem and every measurement reads them.

Prefer per-entity params whenever the quantity is local. A root
`Param` referenced by 1% of constraints is fine; one referenced by
90% of them will dominate factorisation cost.

## Collection types

Wrap entities in these when you have many of them:

| Type | Use it when |
|---|---|
| `refs::Vec<T>` | dense indexed list, contiguous storage, stable `Ref<T>` handles |
| `refs::Deque<T>` | like Vec but supports `push_front` / `push_back` (rolling pose history) |
| `refs::Arena<T>` | arbitrary insertion and deletion with stable handles |
| `Ref<T>` | a handle into the containing collection; dereferences via the parent struct |

A Model struct is "directly composed" if a child Model appears as a
plain field (e.g. `sub: Sub` -- see `single_root_demo.rs`). It's
"collection-composed" if it's wrapped in one of the containers above.

```rust,ignore
#[arael::model]
#[arael(root)]
struct Path {
    // collection-composed: many Pose instances, iterated by the macro
    poses: refs::Deque<Pose>,
    // direct composition: a single Sub entity as a plain field
    globals: Globals,
    hb: SelfBlock<Path, f64>,
}
```

## Struct-level macro attributes

| Attribute | Purpose |
|---|---|
| `#[arael::model]` | declare a Model; generates the Model trait impl (serialize / deserialize / update / accumulate_hessian) |
| `#[arael(root)]` | mark the top-level Model. Generates `LmProblem` impl, manages indices, owns the update cycle |
| `#[arael(root, f32)]` | scalar precision for the generated solver surface (default is f64). Produces `*_f32` methods |
| `#[arael(root, jacobian)]` | additionally emit `calc_jacobian(&params) -> Jacobian<T>` and `calc_cost_table(&params)` for diagnostics |
| `#[arael(root, fit(coll, \|e\| body))]` | shorthand: sum-of-squares fit of a residual body over one collection. Generates a one-line solver entry point |
| `#[arael(skip_self_block)]` | opt out of the mandatory `SelfBlock<Self>`. Reserved for Models whose parameters only appear inside constraints declared elsewhere (rare) |

Constraints can also appear on the root itself -- useful for
regularising root-level parameters (see `global_delta_drift` and
`global_rot_drift` on `Path` in `loc_global_demo.rs`).

```rust,ignore
// Root-level constraint pinning global_delta near its initial value.
#[arael::model]
#[arael(root, f32, jacobian)]
#[arael(constraint(hb, name = "global_delta_drift", {
    let d = path.global_delta - path.global_delta_value;
    [d.x * path.drift_pos_isigma,
     d.y * path.drift_pos_isigma,
     d.z * path.drift_pos_isigma]
}))]
struct Path {
    // ...
    global_delta: Param<vect3f>,
    drift_pos_isigma: f32,
    hb: SelfBlock<Path, f32>,
}
```

## Constraint attributes

The constraint body is symbolic Rust that the macro differentiates
against every parameter it reaches. Attach one or more of these to
any Model struct:

### Block-spec forms

```rust,ignore
#[arael(constraint(hb, { body }))]                    // single local block
#[arael(constraint([hb_ab, hb_ac, hb_bc], { body }))] // bracketed multi-block (N ≥ 2)
#[arael(constraint(pose.hb_pose, { body }))]          // remote SelfBlock (reach into Ref target)
#[arael(constraint(hb_pose, root.hbt, { body }))]     // self-primary + root-owned TripletBlock
```

Dotted names mean two different things depending on the first
segment:

- **`<ref_field>.<block>`** -- reach the target entity's SelfBlock
  through a `Ref<T>` field on this struct. Used by the SLAM
  PointFrine pattern: the constraint lives on PointFrine but writes
  grad / diagonal into Pose's own `hb_pose`.
- **`root.<triplet>`** -- (keyword `root`) point at a `TripletBlock`
  field on the root struct. The across-entity pair for
  (this entity, root) routes into the root's TripletBlock in COO.

```rust,ignore
// Remote SelfBlock: PointFrine lives on PointLandmark but writes
// pose's diagonal via pose.hb_pose; the (pose, path) cross-pair goes
// into a local CrossBlock<Pose, Path>.
#[arael::model]
#[arael(constraint([pose.hb_pose, hb_root], parent = lm, {
    /* residual involving lm, pose, feature, path */
}))]
struct PointFrine {
    #[arael(ref = root.poses)]         pose:    Ref<Pose>,
    #[arael(ref = pose.info.features)] feature: Ref<PointFeature>,
    hb_root: CrossBlock<Pose, Path, f32>,
}

// Self-primary + root-owned TripletBlock: tilt on Pose references
// path.global_rot, so the pose<->path cross pair needs somewhere to
// live. `root.hbt` names a TripletBlock field on the Path root.
#[arael(constraint(hb_pose, root.hbt, {
    let mr_global = path.global_rot.rotation_matrix();
    let mr2w_eff  = mr_global * pose.ea.rotation_matrix();
    let ea_eff    = mr2w_eff.get_euler_angles();
    [(ea_eff.x - pose.info.tilt_roll)  * path.tilt_isigma,
     (ea_eff.y - pose.info.tilt_pitch) * path.tilt_isigma]
}))]
struct Pose { /* ... hb_pose: SelfBlock<Pose, f32> ... */ }
```

### Modifiers

| Modifier | Purpose |
|---|---|
| `parent = <name>` | bind the parent iteration variable to `<name>` inside the body (default is `a_type.to_lowercase()`) |
| `name = "label"` | label the residual group. Shows up in `calc_cost_table` and `JacobianRow::label`. Useful for cost-breakdown diagnostics |
| `guard = <bool expr>` | evaluated once per iteration; when false the whole constraint is skipped for that iteration. Use for optional observations (has GPS this frame?) |
| `<var>: <Type>` | declare an extra binding so the body can refer to `<var>` as typed `<Type>`. Resolved via `Ref` / collection lookup |

```rust,ignore
// `parent = lm` so the body can refer to the enclosing PointLandmark
// as `lm`; `name = "feature_obs"` labels the residual group; `guard`
// skips the whole constraint when the flag is false.
#[arael(constraint([pose.hb_pose, hb_root],
    parent = lm,
    name = "feature_obs",
    guard = feature.enabled,
    {
        /* residual using lm.pos, pose.*, feature.*, path.* */
    }
))]
struct PointFrine { /* ... */ }
```

### Constraint placement

Where the attribute is attached decides what iterates over what:

| Attribute lives on | What iterates | Typical use |
|---|---|---|
| An entity struct (`Pose`) | root iterates `root.<collection of that entity>` | per-entity constraint: drift, tilt |
| A dedicated constraint struct (`PointFrine`) with `Ref<T>` fields | root iterates the collection of constraint structs (often nested: `landmark.frines.iter()`) | observation linking two or more entities |
| The root struct | fires once per solve | regularise root-level params, fix global DOF |

`Pose` can carry both kinds: one `#[arael(constraint(...))]` attribute
per residual group, mixed freely.

## Field-level macro attributes

| Attribute | Applies to | Purpose |
|---|---|---|
| `#[arael(ref = <path>)]` | `Ref<T>` field | where to resolve the Ref. Can be `root.<collection>` or `<other_ref>.<sub_collection>` (chain into a nested collection) |
| `#[arael(cross = (<refA>, <refB>))]` | `CrossBlock<T, T>` field | disambiguate *which* ref pair this CrossBlock serves when two local Refs share the same T |
| `#[arael(constraint_index)]` | `u32` field | receives a unique row id per constraint instance, useful for building per-constraint diagnostics / logs |
| `#[arael(skip)]` | any field | exclude from the Model's serialize / accumulate path. Use sparingly -- the macro already handles non-Param fields correctly |

```rust,ignore
#[arael::model]
struct PointFeature {
    pixel: vect2f,
    // Camera is a Ref<Camera>; we don't want the macro to walk it as
    // a nested Model, so skip it.
    #[arael(skip)] camera: Ref<Camera>,
    // ... measurement data ...
}

#[arael::model]
#[arael(constraint(hb, { /* ... */ }))]
struct PosePair {
    #[arael(ref = root.poses)] prev: Ref<Pose>,
    #[arael(ref = root.poses)] cur:  Ref<Pose>,
    // constraint_index: the macro writes the per-constraint row id
    // here so you can correlate log entries to this specific pair.
    #[arael(constraint_index)] ci: u32,
    hb: CrossBlock<Pose, Pose, f32>,
}
```

## User-defined functions (`#[arael::function]`)

Constraint bodies have a fixed set of built-in ops (arithmetic,
`sin` / `cos` / `exp` / `sqrt` / `clamp` / `safe_asin` / ..., vector
helpers). When your residual needs a custom function -- a factored-
out symbolic helper, or an opaque numerical routine with a known
closed-form derivative -- declare it with `#[arael::function]` and
use it in constraint bodies the same way you'd use `sin`.

Two forms, distinguished by the attributed fn's signature.

### Form A: purely symbolic

`fn name(x: E, ...) -> E { expr }` -- the body is an arael-sym
expression. The macro captures the body as an arael-sym source
string, re-parses it at constraint-expansion time, and inlines the
resulting `E` tree into the surrounding residual. Derivatives come
from arael-sym's own auto-diff.

```rust,ignore
use arael_sym::E;

#[arael::function]
fn sigmoid(x: E) -> E {
    1.0 / (1.0 + exp(-x))
}

#[arael::function]
fn square(x: E) -> E { x * x }

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "fit", {
    [(sigmoid(m.x) - m.target) * m.isigma,
     (square(m.y) - 9.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    y: Param<f64>,
    target: f64,
    isigma: f64,
    hb: SelfBlock<M>,
}
```

The body is stringified and handed to
`arael_sym::parse_with_functions`, so identifiers resolve against
arael-sym's parser rather than Rust's name resolution.

Optional `derivs = [expr, ...]` overrides auto-diff with an
explicit partial per parameter. Expressions are raw tokens, not
strings or closures.

### Form B: opaque numerical eval + symbolic derivatives

`#[arael::function(sym_name, derivs = [...])]` on a
`fn name_eval(x: f32, ...) -> f32` (or `f64`). The eval fn is
opaque numerical code the macro never inspects. The positional
`sym_name` names the symbolic sibling the macro emits for use
inside constraints; the sibling delegates residual evaluation to
the eval fn and uses the stashed `derivs` expressions for
gradient / Hessian assembly.

```rust,ignore
// `my_safe_asin` clamps its input before calling the libm asin
// and supplies a closed-form derivative that stays finite at the
// clamp edge. The `identity(...)` guard blocks the simplifier
// from reordering `1 - x*x + 1e-12` into `1 + 1e-12 - x*x` -- at
// |x| ~ 1 the subtraction already cancels most significant bits
// and the reordered form loses the 1e-12 floor. Same pattern as
// arael-sym's built-in `safe_asin`.
#[arael::function(my_safe_asin,
    derivs = [1.0 / sqrt(identity(1.0 - x * x) + 1e-12)])]
fn my_safe_asin_eval(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).asin()
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "inverse_sin", {
    [(my_safe_asin(m.x) - m.target) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    target: f64,
    isigma: f64,
    hb: SelfBlock<M>,
}
```

`derivs` is required in Form B -- one expression per scalar
parameter, same token shape as Form A derivs. Parameter names
inside the derivative expressions refer to the eval fn's own
parameters in declaration order, so the `x` in
`1.0 / sqrt(1.0 - x * x + 1e-12)` is the `x` from
`fn my_safe_asin_eval(x: f64)`. Derivative expressions may call
other registered `#[arael::function]`s, including each other and
themselves -- mutual recursion is resolved by a two-pass bag
build at constraint-expansion time.

### Ergonomics

- Parameter names in deriv expressions resolve to the attributed
  fn's own parameters, not to anything in the surrounding module.
- Numeric literals accept scientific notation (`1e-12`, `2.5E+2`).
- The sibling fn (Form A body, Form B positional name) is also
  callable from ordinary Rust with `E` arguments, so user fns
  compose with `ExtendedModel` / runtime `parse_with_functions`
  workflows for residuals that aren't known at compile time.
- Errors point at user source: bad signatures, mismatched deriv
  counts, and name collisions fire at attribute expansion;
  parse failures and arity mismatches fire at the call site
  inside the constraint body.

See [examples/user_function_demo.rs](../examples/user_function_demo.rs)
for a runnable two-form demo.

## Runtime differentiation (`ExtendedModel`)

For Models whose residuals aren't known at compile time (e.g. a
user-supplied expression parsed at runtime), implement
`ExtendedModel` in addition to the Model trait. The macro does not
generate the residual evaluation -- you do, by filling in:

```rust,ignore
fn extended_update64(&mut self, params: &[f64]);
fn extended_compute64(&mut self, params: &[f64], grad: &mut [f64]);
```

`extended_compute` evaluates residuals, writes directly into the
LM-provided `grad` slice, and accumulates Hessian contributions into
a `TripletBlock` on the Model (conventionally named `hb`). The
[runtime_fit_demo](../examples/runtime_fit_demo.rs) walks through the
full pattern: symbolic parse → compile-time differentiation of the
parsed expression → use inside the `extended_compute` body.
