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
| `QuaternionParam<T>` | 3 | a rotation-vector delta (not euler angles) composed with a unit-quaternion reference, renormalised on every re-center so it never drifts off SO(3) |
| `TransformParam<T>` | 6 | a rigid transform, such as a robot pose: a translation and a rotation moved together; the optimized delta is represented as a twist (se(3)), so a rotation correction carries the translation with it. Clear `optimize_translation` or `optimize_rotation` to hold either half |
| `UnitVecParam<T>` | 2 | a direction on the unit sphere, such as the normal of a mapped plane landmark: read and write `unit`, which stays unit length because the two parameters rotate a reference direction rather than move its components |

The three SO(3) parameterizations reach the same optimum; they differ only in
residual/Jacobian assembly cost (the linear solve is identical). Each pose
precomputes its rotation matrix and rotation Jacobian once per update, so when
many observations share a pose the exact parameterizations cost almost the same
as the naive one -- on the bearing-dense slam benchmark all three land within
~3% per iteration on assembly and ~0.3% on total solve time. Choose by geometry,
not speed. See the [rotation-parameterization comparison](../benchmarks/slam/README.md#rotation-parameterization-simple-vs-euler-vs-quaternion).

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
| **`TripletBlock<T>`** | COO across-entity pairs | **placed on the coupled co-entity** -- usually the root (declare one `hbt: TripletBlock<T>` on the root struct; constraints reach it via the `root.<field>` block spec), or on a containing parent for the `[hb, parent.<field>]` form. Canonical uses: (1) the root (or parent) has its own `Param` fields and constraints couple entity params with them -- the cross pair lives in that TripletBlock; (2) runtime-parsed residuals via `ExtendedModel` that can't enumerate per-pair CrossBlocks statically -- `extended_compute` writes into the root's TripletBlock directly. **Noticeably slower to assemble** than a multi-CrossBlock because every entry is a `Vec` push. When the constraint touches ONLY root params (the entity is pure data), skip the triplet entirely: name the root's SelfBlock as the primary block, `constraint(root.hb, ...)` -- dense writes, no COO |

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

### Heap-backed blocks: `BoxedSelfBlock` / `BoxedCrossBlock`

`SelfBlock` and `CrossBlock` store their Hessian **inline** as a fixed
`[T; M]` array embedded in the entity struct -- no allocation, best
cache locality. This is the right default.

`BoxedSelfBlock<T>` and `BoxedCrossBlock<A, B>` are drop-in twins that
hold the same block behind a single `Option<Box<...>>` instead. The
math is identical (they delegate to the inline block), so a solve is
bit-for-bit the same; only the storage differs. Swap the type and
nothing else changes:

```rust,ignore
struct Pose {
    pos: Param<vect3f>,
    hb_pose: BoxedSelfBlock<Pose>,   // heap-backed instead of SelfBlock<Pose>
}
```

Two reasons to opt in:

- **Reclaim assembly memory between solves.** The root gains a
  generated `release_blocks()` that frees every boxed Hessian in the
  tree. For a long-lived model that is solved occasionally, call it
  after each solve to hand the transient Hessian memory back; the next
  solve re-allocates on demand. Inline blocks can't do this -- their
  storage is part of the struct.

- **Optimize only part of the model tree.** A boxed block allocates
  its Hessian **only when it is active** -- i.e. at least one of its
  parameters is being optimized. Freeze a sub-tree with `Param::fixed`
  (every index becomes the `u32::MAX` sentinel) and its self-blocks,
  plus any cross-blocks whose *both* endpoints are frozen, stay
  unallocated. A sliding-window SLAM front-end that keeps the full
  history in an [`Arena`](#collection-types) but only optimizes the
  recent window pays Hessian memory for the active window alone.

Allocation is decided once, when the solver assigns block indices
(before the first `zero`/`add_residual`), so the choice is settled for
the whole solve. `block.is_allocated()` reports whether a block
currently holds storage -- useful for tests and diagnostics.

Prefer inline blocks unless you specifically want one of the two
behaviours above; the inline array avoids the pointer indirection and
the per-solve allocation of the active blocks.

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

#[arael(constraint([hb_pose, root.hbt], { /* residual touching pose + root */ }))]
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
   `ExtendedModel::extended_compute` writes directly into the
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
An `Option<T>` field holds zero or one entity: when `Some` its params
and constraints participate like any other entity, when `None` it
contributes nothing (like an empty collection). This works for
constraint entities too -- an optional per-pose observation
(`gps: Option<GpsObs>` on a pose) or an optional root-level constraint
(`loop_closure: Option<Tie>`) fires only when present. Whether the
Option is Some or None must not change during a solve (it shapes the
sparsity pattern, like a guard); flipping it between solves is fine.
A cross/triplet constraint struct as a PLAIN direct field is rejected
at compile time (hold it in a collection or an Option), as is the same
entity type held in more than one location.

Containers are recognized by their literal type name (`Vec`, `Deque`,
`Arena`, `Option`, `Ref`), so they must be spelled out with the element
type visible. An aliased container (`use refs::Vec as RVec` +
`pts: RVec<P>`) would read as an opaque data field and silently drop
`P`'s containment; the macro rejects any unrecognized generic wrapper
holding a model type, in either definition order. To hold model-typed
data deliberately outside the model (a stash, an undo buffer), mark the
field `#[arael(skip)]`. A non-generic alias (`type Poses =
refs::Vec<Pose>` + `poses: Poses`) hides the element type completely and
cannot be detected -- do not alias container types.

Composition nests to any depth: a root may hold `Vec<Path>` where each
`Path` is a Model with its own collections of entities and constraints.
The macro recurses serialize / accumulate / index-wiring through the
intermediate (often block-less) grouping structs, so entities and
constraints can live at any level below the root. A constraint that deep
resolves `root.<coll>` against the root and `parent.<coll>` against its
immediate containing sub-model (see `slam2d_multi_demo.rs`).

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
| `#[arael(root)]` | mark the top-level Model. Generates the `LmProblem` and `RootProblem` impls (unlocking LmProblem's `solve_with` / `solve_dense` / `solve_sparse`), manages indices, owns the update cycle |
| `#[arael(root, f32)]` | scalar precision for the generated solver surface (default is f64). Produces `*_f32` methods. Every block in the model (`SelfBlock` / `CrossBlock` / `TripletBlock`, whose scalar defaults to f64) must match this precision -- a mismatch is a compile error naming the struct and block field. Storage and `Param` precision are free to differ (an f32-storage model solving f64 casts at the boundary) |
| `#[arael(root, jacobian)]` | additionally emit `calc_jacobian(&params) -> Jacobian<T>` and `calc_cost_table(&params)` for diagnostics |
| `#[arael(root, fast_atan)]` | generated code calls `arael::utils::fast_atan` / `fast_atan2` (max error < 1e-6 rad) instead of libm atan / atan2 -- everywhere in residuals, gradients, Hessians, Jacobians. Derivatives are the exact rational forms either way. Alternatively call `fast_atan` / `fast_atan2` per site in a constraint body |
| `#[arael(root, marginalize(field, ...))]` | mark landmark-style fields (small blocks coupled to poses but never to each other) for the sparse solver to marginalize. Generates `RootProblem::marginalize_hint()` with the fields' parameter ranges, which `SparseFaer` reads off the model itself. Optional: without it the solver finds the marginalizable families in the model's coupling graph anyway. Use it to override that -- the named blocks are used as given, and must be mutually uncoupled |
| `#[arael(fit(coll, \|e\| body))]` / `fit64(...)` | shorthand: sum-of-squares fit of a residual body over one collection (f32; `fit64` is f64). Implements `FitProblem`, whose `fit()` / `fit_with()` run the dense LM round trip |
| `#[arael(skip_self_block)]` | opt out of the mandatory `SelfBlock<Self>`. Reserved for Models whose parameters only appear inside constraints declared elsewhere (rare) |

### Generic models

Any `#[arael::model]` struct except the root (and `fit` structs) may
take exactly one type parameter with an inline `Float` bound. Fields
are spelled generically -- `Param<vect2<T>>`, `quatern<T>`, bare
`Param<T>`, `SelfBlock<Pose<T>, T>`, `CrossBlock<Pose<T>, Lm<T>, T>` --
and one definition then serves f64 and f32 models alike. The root stays
concrete and picks the precision by instantiating the entities
(`poses: refs::Vec<Pose<f32>>`); a root must spell every use of an
entity identically (one instantiation per root). See
`examples/plane_slam_demo.rs` for a generic component and
`tests/generic_entity.rs` for a generic entity/constraint model with
f64 and f32 roots.

### Exporting models to other crates

A crate can share its models. After all `#[arael::model]` definitions
(macro expansion is top-down -- the bottom of lib.rs is the natural
place), emit the crate's import macro:

```rust,ignore
arael::export_models!();
```

Every `pub` model struct and enum defined above the invocation joins the
bundle. An importing crate registers all of them in one line, before
defining its own models over them:

```rust,ignore
use model_crate::{Pose, Frine};
model_crate::arael_import!();
```

After that the imported types work like local ones: component fields,
`Ref<Pose>` on local constraint structs, `CrossBlock<Pose, Local>`,
local roots over imported entities, at either precision. Importing the
same bundle twice (diamond dependencies) is harmless, and a model crate
that imports another and calls `export_models!()` re-exports what it
imported.

Rules:

- Exported structs need `pub` fields -- generated code in the importing
  crate reads them directly. A `pub` struct with a non-pub field still
  compiles but is excluded from the bundle; an importer that reaches for
  it gets an error naming the field. `#[arael(skip)]` fields may stay
  private.
- Roots and `fit(...)` structs are not importable: their generated
  solvers are already ordinary pub API.
- An imported constraint struct keeps its `root.<field>` resolution
  paths: the importing root must name its collections as the model
  crate's constraints expect.
- The bundle records each struct's param count; the importer recomputes
  it from the same tokens and fails the build on mismatch (incompatible
  arael-macros versions between the two crates).

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
#[arael(constraint(hb, { body }))]                      // single local block
#[arael(constraint([hb_ab, hb_ac, hb_bc], { body }))]   // bracketed multi-block (N ≥ 2)
#[arael(constraint(pose.hb_pose, { body }))]            // remote SelfBlock (reach into Ref target)
#[arael(constraint([hb_pose, root.hbt], { body }))]     // self-primary + root-owned TripletBlock
#[arael(constraint(root.hb, { body }))]                 // root's own SelfBlock (root params only)
#[arael(constraint(parent.hb, { body }))]               // containing parent's SelfBlock (parent params only)
#[arael(constraint([hb, parent.hbt], { body }))]        // self-primary + parent-owned TripletBlock
```

The positional form carries a single block only. Any N ≥ 2 block
list -- including the `(<local_self_block>, root.<triplet>)`
shape -- must use brackets so multi-block attributes have one
unambiguous syntax. Writing
`constraint(hb_a, hb_b, { body })` is rejected at macro expansion.

Dotted names mean two different things depending on the first
segment:

- **`<ref_field>.<block>`** -- reach the target entity's SelfBlock
  through a `Ref<T>` field on this struct. Used by the SLAM
  PointFrine pattern: the constraint lives on PointFrine but writes
  grad / diagonal into Pose's own `hb_pose`.
- **`root.<triplet>`** -- (keyword `root`) point at a `TripletBlock`
  field on the root struct. The across-entity pair for
  (this entity, root) routes into the root's TripletBlock in COO.
- **`root.<selfblock>`** as the PRIMARY block -- the constraint writes
  the root's own `SelfBlock<Self>` directly. For the "one shared
  parameter set, many observations" shape: the entity carries only
  data (no `Param`, no blocks), every param in the body is the
  root's, and the writes take the dense SelfBlock path (no COO). An
  entity with its own `Param` fields is rejected -- its (entity, root)
  cross pairs need the `[hb, root.<triplet>]` form. In bodies `root`
  names the root (`root.a`), as does the lowercased root type name.
  See examples/root_fit_demo.rs.
- **`parent.<selfblock>`** as the PRIMARY block -- the same shape one
  level down: the data-only entity nests in a collection (or Option)
  INSIDE the parameter-bearing entity and writes the parent's own
  `SelfBlock<Self>`. Every param in the body is the parent's, bound as
  the lowercased parent type name (or a `parent = <alias>`); the sweep
  iterates each parent instance's own observations, so every parent
  fits its own data. The parent may live at any depth below the root.
  If the containing parent turns out to be the root, use `root.hb`.
  See examples/robust_curve_fitting.rs -- observations nest directly
  under the `Curve` they fit, no container struct, no Ref.
- **`parent.<triplet>`** in the SECONDARY slot (`[hb, parent.<field>]`)
  -- the non-root analog of `[hb, root.<triplet>]`: the entity has its
  OWN params (SelfBlock primary), the body also touches the containing
  parent's params, and the (entity, parent) cross pairs go to a
  `TripletBlock<T>` field on that parent. Diagonals land on each
  side's own SelfBlock. The parent binds as its lowercased type name;
  exactly two block fields are allowed in this form. If the containing
  parent is the root, use `[hb, root.<field>]`.

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
#[arael(constraint([hb_pose, root.hbt], {
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
| `guard = <bool expr>` | when false the whole constraint contributes nothing, on every path. Use for optional observations (has GPS this frame?). Must not change during a solve -- see [Guards and optional data](#guards-and-optional-data----the-contract) |
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

### Guards and optional data -- the contract

Two rules govern `guard =` and `Option` reads. Both exist because the
solver discovers the sparsity pattern once, at the first assembly, and
every later iteration writes into that pattern.

**1. Guard values must not change during a solve.** A guard switches a
constraint's cells in or out of the pattern, so a guard that flips
mid-solve either writes outside the discovered pattern or leaves a
zero-curvature parameter (the solve fails DegenerateDiagonal). Set
guards before solving; flipping them BETWEEN solves is fine (call
`invalidate()` on a warm `LmSession`). For a residual that must switch
itself off numerically DURING a solve -- an observation that becomes
undefined at the current iterate -- use `branch()` in the body instead:
it changes the value, not the pattern.

**2. A body that reads through an `Option` sub-struct must be guarded
on it.** A residual is a number, so a `None` has no value to
propagate: the generated read is an unwrap, and the guard is what
keeps it from evaluating.

```rust,ignore
#[arael::model]
struct Extra { off: f64 }

#[arael::model]
#[arael(constraint(hb, guard = self.has_extra, {
    [(p.x - p.extra.off) * 2.0]     // read through the Option: guarded
}))]
struct P {
    x: Param<f64>,
    extra: Option<Extra>,
    has_extra: bool,                 // = extra.is_some(), set when building
    hb: SelfBlock<P>,
}
```

A guarded constraint is fully protected: when the guard is false the
body is not evaluated on ANY path (cost, gradient/Hessian, jacobian),
so the `None` is never touched. An unguarded read panics at solve time
on the first `None`, naming the field as the body spells it
(``optional `p.extra` is None -- guard the constraint reading it``).
Note the guard field is plain data the model carries -- rule 1 applies
to it like any guard.

This contract is about reading Option DATA inside a body. Holding a
whole ENTITY in an Option (`gps: Option<GpsObs>` -- see
[Collection types](#collection-types)) needs no guard: the macro wraps
that entity's sweeps itself, and a `None` contributes nothing.

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
| `#[arael(ref = <path>)]` | `Ref<T>` field | where to resolve the Ref: `root.<collection>` (a collection on the root), `parent.<collection>` (a collection on the immediate containing sub-model -- for a constraint nested below the root), or `<other_ref>.<sub_collection>` (chain into a nested collection) |
| `#[arael(cross = (<refA>, <refB>))]` | `CrossBlock<T, T>` field | disambiguate *which* ref pair this CrossBlock serves when two local Refs share the same T |
| `#[arael(compute = <expr>)]` | any data field | derived field: excluded from serialization, reassigned as `self.<field> = <expr>` on every `update()` (param names in the expression read their current working values). Example: `#[arael(compute = ea.rotation_matrix())]` caching a rotation matrix (see examples/model_demo.rs) |
| `#[arael(constraint_index)]` | `u32` field | receives a unique row id per constraint instance, useful for building per-constraint diagnostics / logs |
| `#[arael(skip)]` | any field | exclude from the model entirely: not serialized, not updated, and a skipped entity collection gets no constraint sweeps. Use sparingly -- the macro already handles non-Param fields correctly |

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

## Constraint-body language

The body inside `#[arael(constraint(..., { body }))]` is interpreted
by the macro (it is never compiled as ordinary Rust), symbolically
differentiated, CSE-optimized, and emitted as flat floating-point
code. The dialect:

- **Statements**: `let` bindings followed by ONE final array
  expression `[r0, r1, ...]` -- each element a residual. Macro calls
  (`assert!`, `println!`), item declarations, and non-final expression
  statements are compile errors (a stray `;`-terminated expression
  used to silently become an extra residual; it no longer compiles).
- **Variables**: the constraint's own entity -- reachable by the
  struct's lowercase name (e.g. `pose2` inside `Pose2`'s constraint) or
  by `self`, the same way a guard names it -- plus its `Ref` field
  names (`a`, `b`, ...), the `parent =` name if given, and the root --
  reachable by the root type's lowercase name or by the `root` keyword
  (`root.a` and `path.a` are the same param; a `Ref` field literally
  named `root` keeps its own meaning). `let` bindings shadow
  everything, constants included (Rust semantics).
- **Constants**: `pi`, `e`, `epsilon` resolve as named constants when
  not shadowed by a `let`.
- **`<field>_value`**: the last-committed value of a param field as a
  zero-derivative constant (see "Initial values via `_value`" above).
- **Scalar functions**: the arael-sym registry -- `sin cos tan asin
  acos atan sinh cosh tanh exp ln log2 log10 sqrt abs heaviside
  identity safe_sqrt safe_asin safe_acos atan2 pow safe_atan2 rad_diff
  rad_sum clamp branch` -- plus any `#[arael::function]` you define (next
  section). Derivative conventions for the `safe_*`/`heaviside`/
  `clamp`/`branch` family are documented in [SYM.md](SYM.md).
  `branch(q, a, b)` compiles to `if q >= 0.0 { a } else { b }`, so only
  the taken side runs and its derivative is the one selected.
- **Vectors / matrices / quaternions**: fields of runtime type
  `vect2*/vect3*/matrix2*/matrix3*/quatern*` dispatch through their
  symbolic companions -- arithmetic, `transpose`, `det`, indexing
  (`m[0]` row, `m[0].x` element), `rotation_matrix()`,
  `get_euler_angles()`, cross products (`%` or `.cross()`),
  `norm/square/unit`, and the static constructors
  (`matrix2sym::rotation(a)`, `matrix3sym::from_rows/from_cols/
  from_elements/rotation_from_euler_angles/rotation_from_axis_angle`,
  `vect2sym/vect3sym::from_components`, `quaternsym::identity/
  from_euler_angles/from_axis_angle`). Constructors are matched by
  path suffix, so any spelling ending in `matrix3sym::rotation_from_
  axis_angle` works. Full surface: [SYM.md](SYM.md), "Geometric
  Primitives".
- **Option fields**: reading through an `Option` struct field is an
  unwrap -- the constraint must be guarded so the body never evaluates
  when the field is `None`, or the first `None` panics at solve time.
  Contract and worked example:
  [Guards and optional data](#guards-and-optional-data----the-contract).
- **Ordering rule**: every entity struct must be defined BEFORE the
  root struct, in top-down file order -- the root's expansion consumes
  the stashed constraints. Violations are a macro error ("define it
  BEFORE the root"), enforced for every containment form at any depth:
  a type registering after a root that can reach it (through
  collections, Option fields, or nested holders) fails its own
  expansion naming that root. The same applies across modules within a
  crate (expansion order follows item order); use `export_models!` /
  `arael_import!` to share models across crates.
- **Errors**: body and attribute diagnostics are prefixed with the
  constraint's `file:line` (spans do not survive the macro's stash
  round trip, so the error arrow points at the root struct -- read the
  prefix).

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
  Mutually-referencing user fns (and forward references to fns
  declared later in the file or in a dependency) work at runtime
  via a registry populated through `inventory`; cross-crate
  composition works without re-declaration.
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
fn extended_update(&mut self, params: &[f64]);
fn extended_compute(&mut self, params: &[f64], grad: &mut [f64]);
```

`extended_compute` evaluates residuals, writes directly into the
LM-provided `grad` slice, and accumulates Hessian contributions into
a `TripletBlock` on the Model (conventionally named `hb`). The
[runtime_fit_demo](../examples/runtime_fit_demo.rs) walks through the
full pattern: symbolic parse → compile-time differentiation of the
parsed expression → use inside the `extended_compute` body.
