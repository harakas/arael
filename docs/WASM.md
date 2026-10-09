# The JavaScript interface

`cargo arael export` writes, beside the C++ and Python trees, a
wasm-bindgen crate under `wasm/` in the model crate: JavaScript classes
over the root model, shaped like the Python package, for a web page that
builds a model, sets its parameters, solves and reads the result back.
The whole solver runs in the page, in WebAssembly, on one thread.

[cxx-examples/m3500_demo](../cxx-examples/m3500_demo) has a working
page under `web/`: the M3500 pose graph composed from a g2o file,
solved through one session, drawn on a canvas, with poses to drag, lock
and fix. Its README has the build and serve commands.

## Prerequisites

The exporter:

```
cargo install cargo-arael
```

The target:

```
rustup target add wasm32-unknown-unknown
```

The wasm-bindgen CLI, at the exact version the generated manifest pins,
since the CLI and the crate must match; the manifest header repeats the
command with its version:

```
cargo install wasm-bindgen-cli --version 0.2.129
```

## Generating and building

```
cd model && cargo arael export
```

writes `wasm/Cargo.toml` (once; edit freely, delete to regenerate),
`wasm/src/lib.rs` (regenerated every time), `wasm/js/arael/g2o.js` (the
vendored g2o reader) and `wasm/.gitignore`. The crate is a workspace of
its own, so the model workspace's native builds never touch it.

The build is two commands, from `wasm/`:

```
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --weak-refs --out-dir pkg \
    target/wasm32-unknown-unknown/release/<crate>_wasm.wasm
```

`pkg/` then holds `<crate>_wasm.js`, the ES module a page imports, and
the `.wasm` beside it. Serve a directory that holds both the page and
`pkg/`, over http: modules and `fetch` do not load from a file URL.

```js
import init, { Graph, LmConfig } from "./model/wasm/pkg/m3500_demo_wasm.js";
await init();
const g = new Graph();
```

## Using the model

The class names are the Rust type names; methods and properties are the
Python names in camelCase (`refAt`, `solveSparse`, `rotAngle`). In a
crate exported from several roots every class carries its root's name
in front (`DecayCell`, `DecayLmConfig`).

**The root** is a class with a constructor: `new Graph()`. It carries
the root's own fields, one accessor method per collection
(`g.poses()`), the solve entries and the covariance assembly.

**A collection accessor** (`GraphPoses`, one class per collection
field) reaches the collection through the root every time it is used,
so it never goes stale:

| Method | Containers | Meaning |
|---|---|---|
| `length` | all | element count (a property) |
| `reserve(n)`, `clear()` | all | |
| `push()` | `refs::Vec`, `Vec` | a default element, handed back |
| `pushBack()`, `pushFront()` | `Deque` | the same, at either end |
| `push()` | `Arena` | a default element; returns its ref |
| `pushN(n)` | `refs::Vec`, `Vec` | `n` default elements; returns the first index |
| `at(i)` | `refs::Vec`, `Vec`, `Deque` | the element at an index; throws past the end |
| `refAt(i)` | `refs::Vec`, `Deque` | the ref of the element at `i` |
| `get(ref)`, `tryGet(ref)`, `contains(ref)` | `refs::Vec`, `Deque`, `Arena` | by ref; `get` throws on a stale one, `tryGet` returns undefined |
| `firstRef()`, `lastRef()` | `refs::Vec` | the ends' refs |
| `frontRef()`, `backRef()` | `Deque` | |
| `first()`, `next(ref)`, `last()`, `prev(ref)` | `Arena` | the live elements in order |
| `pop()`, `truncate(n)` | `refs::Vec`, `Vec` | |
| `popBack()`, `popFront()`, `truncate(n)` | `Deque` | |
| `remove(ref)` | `Arena` | |

A ref is a number; 4294967295 is none. Refs are what `Ref<T>` fields
take and give.

**An entity handle** (`Pose2`, the element classes) is what `push`,
`at` and `get` return. It holds where the element is, not a pointer,
and re-resolves it on every access, so growing the collection cannot
leave it dangling; an access after the element is gone throws. Every
field is a property:

| Field kind | Properties |
|---|---|
| `f64`, `f32` | a number |
| `bool`, `u32`, `i32` | a boolean, a number |
| `vect2d`, `vect3d` | a plain object `{x, y}`, `{x, y, z}` |
| `quaternd` | `{t, x, y, z}`, the scalar part first |
| `matrix2d`, `matrix3d`, `vect<T, N>`, `matrix<T, R, C>` | an array, or an array of row arrays |
| `Param<T>` | the value as above, plus `<name>Optimize` |
| rotation parameters | `<name>` (the vector or quaternion `.value`) and `<name>Optimize` |
| `TransformParam` | `<name>Translation`, `<name>Rotation`, `<name>OptimizeTranslation`, `<name>OptimizeRotation`; `ScaledTransformParam` adds `<name>Scale`, `<name>OptimizeScale` |
| `UnitVecParam` | `<name>Unit`, `<name>Optimize`, and the read-only `<name>UnitD0`, `<name>UnitD1` |
| `AngleParam` | `<name>Angle`, `<name>AngleOptimize`, and the method `<name>RotationMatrix()` |
| `Ref<T>` | a number |
| a sub-model or user component | a method `<name>()` returning its handle |
| `Option<T>` | `has<Name>()`, `make<Name>()`, `clear<Name>()`, and `<name>()` returning the handle or undefined |
| a nested collection | a method `<name>()` returning its accessor |

Setters take the same shapes the getters give.

**Bulk access.** For every settable value of an element type, the
collection accessor has `get<Name>N(start, n)` returning a typed array
of `n` elements' values, interleaved for vectors and matrices, and
`set<Name>N(start, values)` taking one. Numbers come as `Float64Array`,
refs and `u32` as `Uint32Array`, `i32` as `Int32Array`, booleans as
`Uint8Array`. `getRefsN(start, n)` gives the refs of a range. These are
what a drawing loop reads every frame:

```js
const n = g.poses().length;
const xy = g.poses().getPosN(0, n);   // x0, y0, x1, y1, ...
```

Arenas have no bulk access: an index means nothing in one.

## Solving

`LmConfig` starts from a preset with the Rust values filled in:
`new LmConfig()` or `LmConfig.defaults()`, `LmConfig.conservative()`,
`LmConfig.wellConditioned()`, `LmConfig.illConditioned()`. Every field
is a property (`cfg.maxIters = 50`); the optional ones take a number or
`undefined` (`cfg.gradientTolerance = 1e-8`). The observer is not
exported, nor are threads: `numThreads` stays 1 in WebAssembly.

`SparseOptions` mirrors the sparse backend's options, starting from the
Rust defaults; its enum fields carry the tags of the C ABI, named by the
exported enums (`SchurPolicy`, `SolveOrdering`, `EnvelopeMode`,
`SchurMethod`, `BlockSupernodalMode`; `opts.schur = SchurPolicy.Force`).
`LmStatus`, `ReducedOrdering`, `CovMode` and `CovOrdering` name the
other tags.

```js
const r = g.solveSparse(LmConfig.wellConditioned());
r.startCost; r.endCost; r.iterations; r.acceptedIterations;
r.status; r.statusName; r.statusText; r.isSuccess; r.finalLambda;
r.report();       // the text the Rust report() writes
r.timing();       // per-phase seconds, or undefined without gatherTiming
r.plan();         // the sparse backend's plan, or undefined
r.threads();      // what the solve's threads did; one thread in WebAssembly
r.steps();        // the per-attempt timeline, empty without gatherTiming
```

`solveDense(cfg)`, `solveSparse(cfg, opts?)` and `solveBand(kd, cfg)`
return an `LmResult` for every healthy termination and throw an `Error`
carrying the failure's message otherwise. A Rust panic is fatal to the
WebAssembly instance and is printed to the console first.

`LmSession` keeps a backend and what it learns about the problem's
structure, so every solve after the first skips the analysis:

```js
const s = new LmSession();          // or new LmSession(opts)
const r1 = s.solve(g, cfg);         // cold
const r2 = s.solve(g, cfg);         // warm
s.invalidate();                     // after any structural change
```

Call `invalidate()` after any structural change: without it a changed
parameter or block count panics, which is fatal to the instance, and a
change that keeps every count solves warm through stale analysis. The
rules are the Rust session's ([docs/SOLVERS.md](SOLVERS.md#what-a-solve-keeps----context-and-lmsession)).

`g.cost()` is the cost at the current parameters; `g.validate()` the
model's diagnostics as text, empty when clean.

## Covariance

```js
const cov = g.assembleCovariance();          // mode 1 AllMarginals
const cov = g.assembleCovariance(0);         // 0 PerQuery, 2 TriDiagonal
const cov = g.assembleCovarianceWith(mode, ordering, blockSupernodal);
cov.marginalPose2(p);       // Float64Array, row-major dim x dim (Rust's marginal_cov)
cov.conditionalPose2(p);
cov.stdDevPose2(p);         // Float64Array, one per parameter
cov.crossPose2Pose2(a, b);  // row-major
cov.plan();                 // what the assembly decided
```

One method per entity type, since JavaScript has no overloading; `p` is
an entity handle. The tags are the C ABI's.

## Memory

Every class instance lives on the WebAssembly side. Built with
`--weak-refs`, an instance is freed when JavaScript drops it; `.free()`
frees it now. A handle or accessor is small; the root holds the model.

## What is absent

Threads (`numThreads` and `assemblyThreads` are inert, and `threads()`
reports one), the observer callback, the per-constraint cost table and
`calcJacobian` of a `jacobian` root, the structured failure and the
partial result of a failed solve, the transform views of the C++ and
Python skins (the flat `<name>Translation` / `<name>Rotation` properties
are there), `setLogLevel`, `poolShutdown`, and the g2o loader.
