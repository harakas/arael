# The JSON model sidecar

A machine-readable description of a root model's layout, one JSON file
per root, for interface generators and external tooling. The schema is
an interface: additions are allowed, existing fields keep their
meaning, and the `schema` number bumps on breaking changes.

## Emitting

Set `ARAEL_SIDECAR_DIR` while the crate holding the root compiles:

```
ARAEL_SIDECAR_DIR=out cargo build
```

Every `#[arael(root)]` that expands writes `out/<Root>.json`. The
variable is not tracked by cargo's fingerprints, so a cached crate
does not re-emit -- touch a source file or use a fresh target dir to
force expansion. `cargo arael export` handles this automatically.

## Schema (version 1)

```json
{
  "schema": 1,
  "root": "Path",
  "precision": "f32",
  "types": { "<TypeName>": { ... }, ... },
  "constraints": [
    {"on": "PoseTie", "label": "PoseTie", "file": "src/model.rs", "line": 51}
  ]
}
```

`types` holds every model type reachable from the root, keyed by bare
type name. Each entry:

| Key | Meaning |
|---|---|
| `role` | `root`, `entity`, or `component` |
| `param_count` | total optimizable scalars, components folded in |
| `self_block` | name of the `SelfBlock<Self>` field, when one exists |
| `builtin` | `true` for arael's built-in components (`TransformParam`, `UnitVecParam`, `AngleParam`); their `fields` are empty and generators special-case them by name |
| `fields` | declared fields, in declaration order |

Each field has `name`, a `kind`, and kind-specific keys:

| `kind` | Extra keys | Meaning |
|---|---|---|
| `data` | `of` (type spelling), `symbolic` (optional, `true`) | plain scalar or math value (`f64`, `bool`, `vect3f`, `matrix3f`, `quaternd`, ...). Symbolic fields are computed reads in constraint bodies but stay settable data |
| `param` | `of` (inner type), `params` (scalar count) | `Param<T>` -- optimized value |
| `euler_param` | `variant` (`simple` / `universal` / `rotvec`), `scalar`, `params` | rotation parameter (`SimpleEulerAngleParam`, `EulerAngleParam`, `QuaternionParam`) |
| `component` | `of` | compound parameter field (`#[arael(component)]` type); its params fold into this struct's count |
| `struct` | `of` | direct-composed sub-model |
| `optional` | `of` | `Option<Entity>` -- zero or one instance |
| `collection` | `container` (`vec` / `deque` / `arena`), `of` (element type), `spelled` (full spelling) | entity collection |
| `ref` | `of` (target type), `target` (resolution path, e.g. `root.poses`) | `Ref<T>` handle |
| `self_block` / `cross_block` / `triplet_block` | `scalar`; cross adds `a`, `b` | Hessian block storage -- no external accessor |
| `skip` | `of` | excluded from the model (`#[arael(skip)]`, deriv caches, constraint indices) |
| `opaque` | `of` | present but not representable (e.g. `String`, user types) -- generators list these so nothing silently disappears |

`constraints` is informational (labels and source locations of the
constraint attributes on reachable types); it is not needed to build
an interface.

## Notes

- Field order is declaration order; type order is sorted by name.
  Output is deterministic for a given model.
- `collection.of` comes from the registry (reliable); `spelled` is the
  literal source spelling for generators that need the container
  flavor (`refs::Vec` vs `std::vec::Vec`).
- Emission never alters generated code -- with the variable unset the
  build is byte-identical.
