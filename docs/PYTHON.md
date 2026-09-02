# Python interface for arael models

An arael model is a tree of Rust structs: a root struct, collections
of entities under it, and each entity's fields -- parameters the
solver moves, data values it reads, refs to other entities. `cargo
arael export` turns such a crate into a Python package that mirrors
the tree: a class for the root, a view for each collection, a wrapper
for each entity with a property per field, and the solver calls. The
package is pure `ctypes` over the same C shim the tool generates for
C++; it needs CPython 3.9+ and nothing else. numpy is accepted
wherever a sequence is, never required.

## Generating the package

The same `cargo arael export` that writes `capi/` and `cxx/` also
writes `python/{ns}/`, named like the C++ namespace: one ffi/api
module pair per root and a vendored `arael/` subpackage (math value
types, camera, g2o reader, solver surface, array transport). Commit
the tree; `cargo arael check` reports drift.

## Using the model

Build the cdylib once (`cargo build --release -p <crate>-capi`), put
the package on the path, import. The cdylib is found through an
explicit `load(path)`, `$ARAEL_CAPI`, or the cargo build directories
next to the package. A crate with several roots gives one module per
root (`from cxx_mr import line, decay`).

The examples below use the cxx-tests fixture
(`cxx-tests/model/src/lib.rs`). Its root `Fit` has parameters `m`,
`c` and these collections:

| Rust field | element fields | Python view |
|---|---|---|
| `obs: std::vec::Vec<Obs>` | data `x`, `y` | `f.obs` |
| `items: refs::Vec<N>` | parameter `v`, data `t`, `w` | `f.items` |
| `poses: refs::Deque<Pose>` | position, euler angles, heading, targets | `f.poses` |
| `ties: std::vec::Vec<Tie>` | refs `a`, `b` into `poses`, `d`, `w` | `f.ties` |
| `marks: refs::Arena<N>` | as `items` | `f.marks` |

A first solve:

```python
import sys; sys.path.insert(0, "path/to/model/python")
from cxx_fit import fit

f = fit.Fit()                 # the root; owns the Rust model
for i in range(6):
    o = f.obs.push()          # a new Obs, returned as a wrapper
    o.x = float(i)            # fields are properties
    o.y = 2.0 * i + 1.0

cfg = fit.LmConfig.well_conditioned()
cfg.max_iters = 50
r = f.solve_sparse(cfg)       # LmResult; raises AraelError on failure
print(r.status, r.end_cost)
print(f.m, f.c)               # root parameters, read back
```

Calling `push()` and then setting each field separately is slow:
each of those lines is a call into the library, six per observation
here. Fine for six observations; for thousands, build the problem as
the next section shows.

## Building a problem

A problem is the root's collections filled in: push an element, set
its fields, tie elements together through refs. `push` takes the
fields as keywords, so an element costs one library call; when the
data already sits in arrays, the vectorized calls take them whole.

### One element at a time

#### `push` with keywords

Every property of the element is a keyword. An omitted one keeps the
Rust `Default`:

```python
o = f.obs.push(x=1.0, y=3.05)
n = f.items.push(t=1.5, w=2.0)   # v, the parameter, keeps its default
n = f.items.push(t=0.5)          # w stays at the model's Default
```

#### Math-valued fields

Any sequence, or an `arael.math` value:

```python
f.poses.push_back(pos=(0.0, 0.0, 0.0), target=[1.0, 0.5, 0.0])
f.poses.push_back(pos=np.array([0.1, -0.1, 0.05]), target=vect3d(2.0, 1.0, 0.0))
```

#### Parameters, optimize flags, rotations, angles

The keywords are the property names:

```python
f.poses.push_back(pos=(0.2, 0.2, 0.05), pos_optimize=False,
                  ea=(0.1, 0.2, 0.3), ea_optimize=False,
                  heading_angle=0.4, target_dir=(0.9, 0.4))
```

#### Quaternions, matrices, nested components

A quaternion as a `quaternd` or a 4-sequence `(t, x, y, z)`; a matrix
as rows. A nested component (`rig.gain`) keeps its per-element
accessors:

```python
rig = f.rigs.push(q=quaternd.from_euler_angles((0.1, 0.2, 0.3)),
                  ea_u=(0.15, -0.25, 0.6), target_g=1.75)
rig = f.rigs.push(q=(1.0, 0.0, 0.0, 0.0))
rig.gain.g = 0.25
vn = f.vns.push(v=[0.4, -0.1, 0.9, 0.0],
                h=[[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]], wp=0.7, w=1.3)
```

#### Refs and indices from the returned wrapper

The wrapper carries the key it was looked up by: `.ref` on a refs
container, `.index` on a `std::vec::Vec` (the other one raises
`TypeError`):

```python
a = f.poses.push_back(pos=(0.0, 0.0, 0.0))
b = f.poses.push_back(pos=(1.0, 0.5, 0.0))
t = f.ties.push(a=a.ref, b=b.ref, d=(1.0, 0.4, 0.0), w=3.0)
t.index      # 0
```

#### Deque and arena

A deque pushes at either end with the same keywords. An arena's
elements are addressed by ref only (its slots can be removed, so an
index would not stay valid), so its `push` returns the ref and the
element is reached through `marks[ref]`:

```python
front = f.poses.push_front(pos=(-1.0, 0.0, 0.0))   # front.ref == f.poses.front_ref()
m = f.marks.push(t=0.4, w=2.0)
f.marks[m].t                                       # 0.4
```

### Vectorized

The calls below take one field of every element of a collection as
one array: all the `x` of `f.obs`, all the positions of `f.poses`,
one library call each. A numpy array of the matching dtype is read
or written in place; any other sequence is packed first.

#### Many elements per call: `push_many`

A keyword is one value for every new element or a sequence with one
per element. It returns the index of the first new element:

```python
g = fit.Fit()
xs = np.arange(6.0)
first = g.obs.push_many(x=xs, y=2.0 * xs + 1.0)
g.items.push_many(t=[1.5, -0.3, 0.7], w=1.0)   # w broadcast to all three
g.items.push_many(n=2)                          # n needed when nothing is a sequence
g.poses.push_many(pos=np.zeros((3, 3)),         # (n, 3) for a vect3d
                  heading_angle=[0.1, 0.2, 0.3])
```

#### Ref arrays

`get_refs()` is the ref of every element as one `uint32` array of raw
handles, in index order; a ref keyword takes such an array, or a list
of refs:

```python
refs = g.poses.get_refs()                          # (3,) uint32
g.ties.push_many(a=refs[:-1], b=refs[1:], d=(1.0, 0.5, 0.0), w=3.0)
g.ties.push_many(a=[g.poses.ref_at(0)], b=[g.poses.ref_at(2)], w=[1.0])
```

#### Vectorized write: `set_<field>`

On an existing collection: a scalar broadcasts, a sequence sets one
per element, a strided numpy view is read as it is:

```python
g.obs.set_y(0.0)
g.obs.set_x(np.linspace(0.0, 1.0, 6))
X = np.arange(12.0).reshape(6, 2)
g.obs.set_x(X[:, 1])                               # a column of a 2-D array, no copy
g.items.set_v_optimize(np.array([True, False, True, False, True]))
g.poses.set_pos([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
```

#### Vectorized read: `get_<field>`

With numpy importable the result is a numpy array (`float64`,
`float32`, `int32`, `uint32` for refs, `bool`), shaped `(n,)` or
`(n, k)` for a math field; without numpy it is a flat ctypes array:

```python
x = g.obs.get_x()               # (6,) float64
P = g.poses.get_pos()           # (3, 3), one row per pose
ok = g.items.get_v_optimize()   # (5,) bool
a = g.ties.get_a()              # (3,) uint32 raw handles
```

#### Errors

An array of the wrong dtype raises `TypeError` rather than converting
(a silent copy could dominate the call); a wrong length raises
`ValueError`:

```python
g.obs.set_x(np.arange(6))       # TypeError: int64 for a float64 field
g.obs.set_x(np.zeros(5))        # ValueError: 6 expected
```

### Build, solve, read back

```python
h = fit.Fit()
xs = np.arange(6.0)
h.obs.push_many(x=xs, y=2.0 * xs + 1.0 + np.where(xs % 2 == 0, 0.05, -0.05))
h.items.push_many(t=[1.5, -0.3, 0.7], w=[1.0, 2.0, 0.5])
r = h.solve_dense(fit.LmConfig.well_conditioned())
v = h.items.get_v()             # the solved item parameters, one array
```

Measurements behind these forms: docs/dev/FAST_PYTHON.md.

## Collections

Every view has `len(view)`, `view[i]` (negative indices too),
iteration, `reserve`, `clear`, `truncate` and the `push`/`pop` family
of its container. A wrapper re-resolves its element on every access,
so a wrapper held across later pushes stays valid (unlike a C++
pointer):

```python
held = f.rigs[0]
for _ in range(200):
    f.rigs.push()
held.target_g = 1.5          # still the first rig
```

Refs exist on the refs-flavoured containers (`refs::Vec`, `Deque`,
`Arena`); a `std::vec::Vec` field has index access and iteration
only. A ref is a small typed handle: `.raw`, `.valid`, equality,
hashable; default-constructed it is null:

```python
r = f.items.ref_at(1)        # also first_ref() / last_ref(); front_ref() / back_ref() on a deque
f.items[r].t                 # lookup by ref
r in f.items                 # True while r addresses a live element
f.items.try_get(r)           # the element, or None for a stale or foreign ref
f.items.get_refs()           # every ref at once, a uint32 array of raw handles
fit.NRef().valid             # False
```

An arena removes elements, and its refs notice:

```python
m = f.marks.push(t=0.4, w=1.0)
f.marks.remove(m)
m in f.marks                 # False
list(f.marks.refs())         # the live refs, in slot order
```

An `Option<Entity>` field is the entity or `None`. `make_<field>`
creates it with the entity's fields as keywords, like `push`;
`clear_<field>` empties it:

```python
gps = pose.info.make_gps(pos=(7.0, 8.0, 9.0), isigma=2.5)
pose.info.gps.pos.y          # 8.0
pose.info.clear_gps()
pose.info.gps                # None
```

## Values

Math-typed fields (`vect2/3`, `matrix2/3`, `quatern`, `f`/`d`
variants) live in the vendored `arael.math`. The classes are the FFI
structs themselves, with the operators of the C++ headers: `*` dot,
`%` cross, `norm`, `unit`, `transpose`, `symmetric_eigen`, the
euler/quaternion conversions. A property takes any sequence in and
returns the math value:

```python
pose.pos = (0.1, -0.1, 0.05)
pose.pos.x                   # 0.1
q = quaternd.from_euler_angles((0.1, 0.2, 0.3))
q.rotation_matrix()
```

`vect<T, N>` and `matrix<T, R, C>` fields take any length-matching
sequence (rows for a matrix) and read back through the cached
factories `vectnd(n)` / `vectnf(n)` / `matrixnd(r, c)` /
`matrixnf(r, c)`: iterable, indexable, with `+ - *` (dot, matrix-
vector, matrix-matrix), `norm`, `transpose`:

```python
vn.v = [0.1, 0.2, 0.3, 0.4]
vn.h = [[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]]
vn.h[1][3]                   # 0.4
```

### Transforms

A `TransformParam` or `ScaledTransformParam` field is reachable two
ways. The flat properties, `frame.pose_translation`,
`frame.pose_rotation`, `frame.pose_optimize_translation`, and for the
scaled one `frame.st_scale` and `frame.st_optimize_scale`, are what
`push` keywords and columns use. The field itself, `frame.pose`, is a
live view: its parts read and write through, and it acts like a
transform, with the frame convention `r2w` reading as "robot to
world":

```python
frame.pose.translation = (0.3, -0.2, 0.5)
frame.pose.rotation = quaternd.from_euler_angles((0.1, 0.2, 0.3))
frame.pose.optimize_rotation = False
x_w = frame.pose * x_r                 # R x + t
x_r = frame.pose.inv() * x_w           # R^T (x_w - t)
d_w = frame.pose.rotate(d_r)           # a vector: R d_r, no translation
b2a = a.pose.inv() * b.pose            # composition: b's pose seen from a
c2w = frame.pose * frame.st            # rigid times scaled is scaled
```

`inv()` and compositions are plain values, `arael.transform3d` for a
rigid result and `arael.scaled_transform3d` when a scale is involved
(`f` variants for `f32` fields), with the same methods and `*`;
`frame.pose.to_transform()` and `frame.st.to_scaled_transform()` are
the snapshots. A scaled transform acts as `s (R x) + t` and its
inverse as `R^T (x - t) / s`; `rotate` never scales.

## Solving

`LmConfig` starts from a preset with the actual Rust values filled
in: `defaults()`, `conservative()`, `well_conditioned()`,
`ill_conditioned()`. Optional fields take a value or `None`:

```python
cfg = fit.LmConfig.well_conditioned()
cfg.max_iters = 50
cfg.gradient_tolerance = 1e-8      # or None
```

`solve_dense(cfg)`, `solve_sparse(cfg, opts=None)` and
`solve_band(kd, cfg)` (kd the half-bandwidth in scalar parameters)
return an `LmResult` for every healthy termination. `r.status` is an
`LmStatus`; `r.status.is_success()` and `r.status.as_str()` mirror
the Rust helpers, and success is not `status >= 0` (max_iters and
the time limit end on the Ok side without being successes).

A solver failure or a caught Rust panic raises
`AraelError(status, message)`. `e.partial` is the best accepted
`LmResult` when the solve got that far, `e.failure` the structured
cause: a `SolveFailure` with its `SolveFailureKind`, the parameter /
row / block indices (-1 where not applicable) and a `DiagonalFault`
for a degenerate diagonal.

```python
try:
    r = f.solve_dense(cfg)
except fit.AraelError as e:
    print(e.status, e.message, e.failure.kind if e.failure else None)
```

`SparseOptions()` holds the sparse backend's Rust defaults; set its
fields and pass it to `solve_sparse`: `schur` (`SchurPolicy`),
`ordering` (`FaerOrdering`), `envelope` (`EnvelopeMode`) and
`envelope_panel_width`, `supernodal`, `narrow_band`, `schur_solve`
(`SchurSolve`), and the block supernodal Cholesky knobs
`block_supernodal` (`BlockSupernodalMode`), `block_supernodal_batch`,
`block_supernodal_memory_lean`. An enum setter rejects a value
outside the enum with `ValueError`.

```python
so = fit.SparseOptions()
so.schur = fit.SchurPolicy.FORCE
so.ordering = fit.FaerOrdering.AMD
r = f.solve_sparse(cfg, so)
r.plan                       # the SchurPlan the backend used; None for dense and band solves
```

`LmSession(opts=None)` keeps the sparsity analysis (pattern,
ordering, symbolic factorization, Schur plan) across repeated solves
of the same structure; warm solves are bit-identical to cold ones. A
parameter-count change re-analyzes by itself; call `invalidate()`
after a structural change at the same count:

```python
sess = fit.LmSession()
r1 = sess.solve(f, cfg)
r2 = sess.solve(f, cfg)      # warm
```

The result owns the whole Rust-side solve and stays valid however
many solves follow. `r.report()` and `r.pretty_report()` render
status, costs, the timing breakdown and the backend's plan. With
`cfg.gather_timing = True`, `r.timing` holds the breakdown and
`r.steps` the per-attempt timeline (a list of `LmStep`). A warm
restart re-enters at the previous damping; the optimized parameters
already live in the model:

```python
cfg.initial_lambda = r.final_lambda
r = f.solve_dense(cfg)
```

`cfg.observer = fn` is called with an `LmIter` per damped attempt
(`it.lambda_`, `it.params_len`, `it.param(i)`, `it.param_list()`);
returning `False` stops the solve. `f.cost()` is the total cost at
the current values, `f.validate()` the diagnostic text (empty when
the model is clean), `f.last_error()` the last message.
`fit.set_log_level(fit.LogLevel.WARN)` quiets arael's diagnostics
process-wide (INFO is the default). A model is freed on garbage
collection; `free()` forces it.

## Covariance and diagnostics

`assemble_covariance(mode=CovMode.ALL_MARGINALS, ordering=
CovOrdering.AUTO, block_supernodal=BlockSupernodalMode.AUTO)`
prepares the covariance at the current (solved) parameters and
returns a view, or raises. `ordering` and `block_supernodal` decide
what the assembly costs, never what it is: `AUTO` builds a symbolic
factorization per candidate ordering to choose between them, naming
the ordering skips that. The view owns its assembly, is freed on
garbage collection (`free()` forces it), and a later assembly never
disturbs an older view. Entity arguments must come from the live
model:

```python
cov = f.assemble_covariance()
cov.marginal(f.items[0])            # 1 param -> float; 2 or 3 -> matrix2d / matrix3d; more -> row-major tuples
cov.conditional(f.items[0])         # all other parameters held fixed
cov.std_dev(f.poses[0])             # per-parameter standard deviations, every CovMode
cov.cross(f.poses[0], f.poses[1])   # row-major tuples
cov.plan()                          # CovPlan: the ordering kept, candidate_flops, symbolics_built, block_route
```

When the root is `#[arael(root, jacobian)]`, `f.cost_table()` is the
per-constraint cost breakdown as a dict (label -> that group's
robustified cost, summing to `f.cost()`), and `f.calc_jacobian()` an
owned snapshot for rank analysis: `num_residuals`, `num_params`,
`singular_values(column_normalised=False)` and `column_l2_norms()`
as lists.

## Support library

`arael.geometry` holds the pinhole camera (`cameraf` / `camerad`;
`Camera` is a legacy alias of `cameraf`). `arael.g2o` reads and
writes pose-graph files: `Dataset2` for SE2 with the
`iso_sqrt_info` / `eigen_sqrt_info` whitening accessors, `Dataset3`
for SE3:QUAT with sqrt-information Cholesky blocks; `to_g2o()` and
`save(path)` write the graph back out byte-identical to the Rust
writer.

## Notes

- Where C++ has `front()` / `back()` / `empty()` on a view, Python
  spells them `view[0]` / `view[-1]` / `len(view)`; where C++ returns
  `option<T>` from a method (`r.plan()`), Python has a property
  returning the value or `None` (`r.plan`).
- One model, one thread. The GIL is released around foreign calls, so
  a solve does not block other Python threads.
- Worked examples: `cxx-examples/<demo>/python/`, each next to the
  same demo's C++ driver; the m3500 twin matches the Rust and C++
  output digit for digit.
- The parity suite (`cxx-tests/runner/tests/python.rs`) builds the
  fixture problem through the generated package and compares every
  value exactly against the Rust mirror. Design notes:
  docs/dev/PYTHON.md; docs/CXX.md is the C++ side of the same surface.
