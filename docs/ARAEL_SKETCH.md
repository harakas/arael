# Arael Sketch -- a parametric 2D sketcher built on arael

The **arael-sketch\*** crates are an interactive constraint-based 2D
sketch editor built on the [arael](../README.md) optimization
framework. You draw geometry (points, lines, arcs, circles), apply
geometric constraints (horizontal, coincident, tangent, parallel,
perpendicular, ...) and parametric dimensions (lengths, radii,
angles, optionally expressions like `d0 * 2 + 3`), and the
Levenberg-Marquardt solver keeps everything consistent in real time.

[![Sketch Editor](sketch.png)](https://sketch.mare.ee/)

[Try it in the browser](https://sketch.mare.ee/)

Treat arael-sketch as **the big example** for arael: a real-world,
end-user application that exercises nearly every feature of the
framework. If you want to see how arael's macros, Hessian blocks,
Jacobian analysis, and both differentiation modes fit together on a
non-toy problem, this is the codebase to read.

## Why it is a good showcase for arael

The sketch solver combines **both differentiation modes** arael
offers:

- **Geometric constraints** (horizontal, coincident, parallel,
  tangent, etc.) use **compile-time differentiation** -- each
  constraint is a `#[arael(constraint(...))]` attribute on a struct
  in `arael-sketch-solver`. The macro symbolically differentiates
  every residual, applies common-subexpression elimination, and
  emits compiled Gauss-Newton code inside the usual `calc_cost` /
  `calc_grad_hessian` pair.
- **Parametric dimensions** use **runtime differentiation** -- the
  user types an expression such as `d0 * 2 + 3` as a dimension
  value. At set-up time the expression is parsed with
  `arael_sym::parse`, differentiated symbolically (`E::diff(...)`),
  and plugged into the solver via `ExtendedModel` and a
  `TripletBlock` on the root. Dimensions can reference each other,
  entity properties (`L0.length`, `A0.radius`), and arithmetic
  expressions; broken references (deleted entities) are detected and
  the dimension falls back to its last computed value.

This makes arael-sketch a fully parametric constraint solver where
changing one dimension propagates through all dependent expressions.

Beyond the two differentiation modes, the sketch uses:

- `SelfBlock` / `CrossBlock` for the dense per-entity Hessian blocks
  (see `Point`, `Line`, `Arc`, and the various coincidence / parallel
  / tangent structs).
- `TripletBlock` on the root for the runtime expression-dimension
  rows -- 3+-entity constraints written at runtime.
- `#[arael(root, jacobian)]` for DOF analysis, conflict detection,
  and the "blocker" diagnostics surfaced when a new constraint is
  rejected as redundant.
- `Ref<T>` / `refs::Vec<T>` handles that survive insertion and
  deletion across long edit sessions.
- WASM/browser support -- the GUI crate compiles unchanged to
  `wasm32-unknown-unknown` via eframe.

## Three-crate layout

After a recent split the editor is three independent crates, all
under the arael workspace. Pull in only what you need:

| Crate | Depends on | What lives there | Has egui? |
|---|---|---|---|
| [`arael-sketch-solver`](../arael-sketch-solver) | `arael` | `Sketch`, entities (`Point`, `Line`, `Arc`), constraints, dimensions, Jacobian, DOF analysis, blocker report | No |
| [`arael-sketch-backend`](../arael-sketch-backend) | `arael-sketch-solver` | `CommandContext`, `Action`, `History`, text-command parser, conflict detection, MCP server, geometry helpers | No |
| [`arael-sketch`](../arael-sketch) | `arael-sketch-backend` | `EditorApp` and eframe/egui GUI: drawing, tools, drag, selection, theme, params panel | Yes |

The boundary is hard: `arael-sketch-backend` has zero dependency on
`eframe`, `egui`, `egui_extras`, `egui_commonmark`, or `rfd`. Every
feature the GUI exposes -- drawing, constraining, dimensioning,
scripting, MCP -- is available to headless consumers through the
backend crate.

If you want to **embed the solver itself** (say, in a different GUI
or a batch tool), depend on `arael-sketch-solver`. If you want the
command language, undo/redo, and the MCP server without the GUI,
depend on `arael-sketch-backend`. The full editor including the
egui canvas is `arael-sketch`.

## Backend API quick tour

`arael-sketch-backend` is the layer most programmatic users will
touch. The public surface (from `arael-sketch-backend/src/lib.rs`):

```rust
pub use ids::{ConstraintId, CoincidentKind, MidpointKind,
               Selection, find_constraint_by_name};
pub use actions::{Action, resolve_dim_endpoint};
pub use history::{History, CursorState};
pub use conflicts::check_constraint_conflict;
pub use commands::{CommandContext, DRAG_PULL_WEIGHT};

pub mod geometry;      // arc/line math, hit-testing
pub mod earc_fit;      // elliptic arc from tangents + bulge
pub mod mcp_server;    // (not wasm) MCP HTTP server behind a wake callback
```

### Three layers, one `Action` alphabet

The sketch is always mutated through an `Action`. Three independent
layers build sequences of actions, but they all emit variants from
the same enum, which is what makes `History`'s snapshot model work
uniformly across GUI edits, scripted batches, and MCP tool calls:

1. **Raw `Action::apply`.** The lowest layer.
   `Action::AddLine { p1, p2 }` is literally
   `sketch.add_line(p1, p2)` -- no endpoint snapping, no
   coincidence, no solve. Stable semantics for programmatic
   callers. Actions are deliberately dumb; anything cleverer
   belongs in a higher layer. (See
   [`rectangle_actions`](../arael-sketch-backend/examples/rectangle_actions.rs)
   -- the rectangle's four corner coincidences are pushed
   explicitly because no layer below is doing it for you.)

2. **Command parser** (`cmd_add_line` and siblings in
   `arael-sketch-backend/src/commands.rs`). Walks the parsed text
   arguments, snaps endpoints against existing entities within a
   tolerance, and emits extra coincidence actions
   (`ApplyCoincidentPP`, `ApplyCoincidentLL21`,
   `ApplyCoincidentArcStart`, etc.) alongside the bare `AddLine`.
   The `[connected: L1.p1=L0.p2]` messages and the "Coincident
   constraint already exists" dedup you see in the
   [`rectangle_commands`](../arael-sketch-backend/examples/rectangle_commands.rs)
   example come from this layer.

3. **GUI tools** (`EditorApp::apply_snap_coincident` and friends in
   `arael-sketch/src/main.rs`). The mouse resolves to a richer
   `SnapTarget` enum than a parser can concisely name -- line body,
   line midpoint, arc body, arc midpoint, arc start/end/center --
   so the GUI dispatches the full ten-variant snap taxonomy into
   the matching actions (`ApplyLineP1OnLine`, `ApplyMidpointLP1`,
   `ApplyMidpointLP1Arc`, `ApplyLineP1OnArc`, ...). It also layers
   in auto-perpendicular (`ApplyPerpendicular`) when the drawn line
   crosses a host line at a right angle, gated by
   `has_perp_conflict` so a redundant perp is never pushed. All of
   it goes into one `begin_group()` frame so a single Ctrl+Z
   undoes the line *and* its auto-snaps *and* any auto-perps as
   one unit.

The command parser and the GUI are independent pipelines and do
duplicate the "add line then coincident-where-needed" shape --
deliberately, because each sees different input -- but they
converge on the same low-level `Action` alphabet. That is the key
invariant the whole backend leans on.

### Dispatch paths

Two dispatch paths into the sketch, both supported by the same
backend:

1. **Direct `Action::apply`**. Construct an `Action::Add...` /
   `Action::Apply...` enum variant and call `action.apply(&mut
   sketch)` to mutate the sketch and solve. Each `Action` is
   `serde`-encodable and round-trips through `History`, so you get
   undo/redo for free. This is what the GUI toolbar uses for
   single-click operations.

2. **Text commands through `CommandContext`**. Build a
   `CommandContext` around your sketch + history + session-var
   state, then feed it a command string. The parser handles coord
   literals, entity references (`L0.p2`, `A1.center`), geometric
   functions (`midpoint(L0)`, `intersect(L0, L1)`), vector
   arithmetic, and per-session variables. Under the hood each
   command still produces one or more `Action`s, so the history
   stack stays coherent whether you drove the sketch from the GUI,
   a script, or the MCP server.

   ```rust
   use arael_sketch_backend::{CommandContext, commands::execute};

   let mut ctx = CommandContext::new();
   execute(&mut ctx, "add_line 0,0 3,0");
   execute(&mut ctx, "add_line L0.p2 3,2");
   execute(&mut ctx, "coincident L0.p2 L1.p1");
   execute(&mut ctx, "horizontal L0");
   execute(&mut ctx, "length L0 3");
   ```

Full command reference (40+ commands for geometry, constraints,
dimensions, parameters, introspection, view control, explain / DOF
diagnostics): see
[`arael-sketch-backend/docs/COMMANDS.md`](../arael-sketch-backend/docs/COMMANDS.md).

Additional backend facilities worth knowing about:

- **`History` + `CursorState`.** Snapshot/restore of the whole
  sketch via bincode, with groupable actions so a multi-step
  operation (for instance, a rectangle tool pushing four
  `AddLine`s, four coincidences, and four flags in one go) undoes
  as one unit. The GUI's Ctrl+Z and the MCP's implicit atomicity
  both ride on this.

  `History` exists for four reasons:

  1. **Undo/redo.** Every GUI click, drag, and command line funnels
     through `history.push(action, sketch, cursor)`; Ctrl+Z and
     Ctrl+Shift+Z are the undo/redo methods on this struct.
  2. **Atomicity.** `begin_group()` tags subsequent pushes with the
     same group id so a multi-step operation undoes as one frame.
     `Action::Drag { snapshot }` wraps a whole drag trajectory in a
     single reversible unit.
  3. **Cursor restoration.** Each frame stores where the
     command-panel cursor was and which tangent it pointed along,
     so undoing a `move` puts the typing cursor back where it was.
  4. **Snapshot storage, not replay.** Every push serialises the
     post-apply sketch via bincode; undo/redo deserialises the
     snapshot and calls `sketch.solve()` once. The action log is
     *not* replayed forward, because `Action::apply` is
     non-deterministic in general (drag trajectories, solver
     starting points, helper-point ordering) and a forward replay
     could diverge. Snapshots make undo/redo exact.

  > **Only `Action`s are tracked.** Any mutation you want to be
  > reversible must go through an `Action`. Raw mutations like
  > `sketch.lines[r].p1 = Param::fixed(..)` or pushing constraint
  > structs directly into `sketch.*` collections bypass `History`
  > entirely, are invisible to undo/redo, and will not come back on
  > a redo. The raw `Sketch` API is there for headless batch work
  > where history does not matter (see `rectangle_solver`); the
  > moment you want undo/redo, always go through the `Action` enum
  > (see `rectangle_actions`, which uses `Action::LockLineP1`
  > instead of a direct `Param::fixed` assignment precisely so the
  > pin survives undo/redo).
- **`check_constraint_conflict(sketch, action)`.** Cheap pre-apply
  validation -- catches self-reference, impossible orientations
  (horizontal and vertical on the same line), and transitive
  redundancy before the solver sees the new constraint. Used by the
  `explain` command and by the interactive validation loop.
- **Blocker analysis (DOF rejection).** When a constraint is
  rejected because it does not reduce DOF, the solver's
  `Sketch::analyze_blockers` reports which existing constraints are
  the minimum set responsible. The GUI flashes those constraints in
  pink; headless consumers get the names in the error string.
- **MCP server.** `mcp_server::start(addr, verbose, allow_all,
  wake)` spawns an axum + tokio server on a background thread,
  hosting the Streamable HTTP MCP protocol (2025-03-26 spec). The
  `wake: Arc<dyn Fn() + Send + Sync>` callback is invoked on each
  inbound request so the host can drain the command channel (the
  GUI passes `ctx.request_repaint`; a bare-metal host could pass a
  no-op). The server exposes `execute_command`, `execute_script`,
  `get_sketch_state`, and `get_help` tools, and serves
  `COMMANDS.md` plus the live sketch as MCP resources.

## Running (native)

```bash
cargo run -r -p arael-sketch
```

Common flags: `--empty` (start blank instead of with a demo),
`--nogui --stdout --script path/to/script.cmd` (headless script
run), `--mcp --mcp-allow-all` (start the MCP server), `--dark` (dark
theme).

## Running (browser)

The editor compiles to WebAssembly and runs in the browser unchanged
via eframe. Requires [trunk](https://trunkrs.dev/)
(`cargo install trunk`) and the `wasm32-unknown-unknown` target
(`rustup target add wasm32-unknown-unknown`):

```bash
cd arael-sketch
trunk build --release
python3 -m http.server -d dist 8080
# Open http://localhost:8080
```

## Tools

- **Line (L)**, **Circle (O)**, **Arc (A)**, **Point (P)** -- draw
  geometry with auto-snap to nearby points, endpoints, and curves.
- **Dimension (D)** -- length, distance, radius, angle, and
  point-to-line distance dimensions with draggable annotations.
  Numeric values and parametric expressions (`d0 * 2`,
  `L0.length + 3`) are both accepted.
- **Select (S)** -- click to select, drag to move entities,
  Backspace/Delete to remove.
- **Dark/Light mode** toggle, **Save/Load** (JSON),
  **Undo/Redo** (Ctrl+Z / Ctrl+Shift+Z).

## Constraints

Horizontal (H), Vertical (V), Coincident (C), Parallel,
Perpendicular, Equal length/radius, Tangent (T), Collinear, Midpoint
(M), Symmetry (lines or points about a mirror line), Lock (K), Line
style (X). Constraints are visualised as symbols on the geometry and
can be selected and deleted.

Internally these are plain Rust structs in `arael-sketch-solver`
with a `#[arael(constraint(...))]` attribute describing the residual,
e.g. `Parallel`, `Perpendicular`, `CoincidentLL21`, `TangentLA`,
`Midpoint`, etc. Each one owns a `CrossBlock` (for two-entity
constraints) or `SelfBlock` (for single-entity flag-style
constraints). Some constraints internally introduce a **helper
point** (named `Pc<n>`) to bridge into entity centres or midpoints;
helper points are invisible to the user and the command interface.

## Runnable examples

Three end-to-end rectangle examples, one per crate, demonstrating
each layer of the API:

| Example | Crate | Shows |
|---|---|---|
| [`rectangle_solver`](../arael-sketch-solver/examples/rectangle_solver.rs) | `arael-sketch-solver` | Raw `Sketch` API: add lines, flag H/V, push `CoincidentLL21` structs, set length flags, call `sketch.solve()`. |
| [`rectangle_actions`](../arael-sketch-backend/examples/rectangle_actions.rs) | `arael-sketch-backend` | `Action` enum + `History`: same rectangle built through `Action::AddLine` / `ApplyHorizontal` / `ApplyCoincidentLL21` / `AddDimension` / `LockLineP1`, then undo+redo. |
| [`rectangle_commands`](../arael-sketch-backend/examples/rectangle_commands.rs) | `arael-sketch-backend` | `CommandContext` + text commands: same rectangle driven by the command parser, exactly the syntax the `/` panel and MCP agents use. |

Run them with:

```bash
cargo run -r -p arael-sketch-solver   --example rectangle_solver
cargo run -r -p arael-sketch-backend  --example rectangle_actions
cargo run -r -p arael-sketch-backend  --example rectangle_commands
```

## Sketch Solver API in detail

The solver crate can be used directly, independently of the command
layer or the GUI. Good for unit tests, batch tooling, or embedding in
another application. This is the code inside
[`rectangle_solver`](../arael-sketch-solver/examples/rectangle_solver.rs):

```rust
use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch_solver::*;

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
sketch.coincident_ll21.push(CoincidentLL21 { a: bottom, b: right, nid: 0, cid: 0, hb: CrossBlock::new() });
sketch.coincident_ll21.push(CoincidentLL21 { a: right, b: top,    nid: 0, cid: 0, hb: CrossBlock::new() });
sketch.coincident_ll21.push(CoincidentLL21 { a: top,    b: left,  nid: 0, cid: 0, hb: CrossBlock::new() });
sketch.coincident_ll21.push(CoincidentLL21 { a: left,   b: bottom, nid: 0, cid: 0, hb: CrossBlock::new() });

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

The sketch solver uses Levenberg-Marquardt optimization with drift
regularization and robust drag constraints. Geometric constraints
are differentiated at compile time; parametric expression dimensions
use runtime differentiation via `ExtendedModel` and a root-mounted
`TripletBlock`.

## Command panel & scripting

Press `/` in the GUI to open the command panel. Full scripting
support with 40+ commands for geometry creation, constraints,
dimensions, parameters, introspection, and view control. Commands
support expressions, coordinate references (`L0.p2`, `@dx,dy`),
geometric functions (`midpoint(L0)`, `intersect(L0,L1)`), and vector
arithmetic (`L0.p2 + normal(L0) * 3`).

See [arael-sketch-backend/docs/COMMANDS.md](../arael-sketch-backend/docs/COMMANDS.md)
for the full command reference.

Scripts can also be run headless:

```bash
cargo run -r -p arael-sketch -- --empty --nogui --stdout --script robot.cmd
```

This path boots the command context, runs every line, prints results
to stdout, and exits. No GUI is initialised at all. Ideal for
regression tests and CI-side sketch generation.

## AI agent integration (MCP)

`arael-sketch-backend` embeds an **MCP (Model Context Protocol)
server**, enabling AI agents like Claude Code to create and modify
sketches programmatically. The agent sends sketch commands and reads
state through the standard MCP tool interface.

![Dark mode with AI-drawn geometry](../arael-sketch/docs/dark.png)

*Dark mode with parameters panel, command history showing MCP agent
connection, and geometry drawn by Claude Code.*

Run the native editor with:

```bash
cargo run -r -p arael-sketch -- --mcp --mcp-allow-all
```

The MCP server is decoupled from egui: it takes a wake callback
(`Arc<dyn Fn() + Send + Sync>`) instead of a handle to the GUI. The
editor supplies `ctx.request_repaint`; a headless host (for example,
a non-GUI MCP bridge) would supply whatever drains its own command
channel. That decoupling is what lets the entire backend live in a
crate with zero GUI dependencies.

## Further reading

- [arael README](../README.md) -- the framework behind it all:
  macros, solvers, Jacobian, Starship method, SLAM and localization
  demos.
- [docs/SLAM.md](SLAM.md) -- a large-scale end-to-end example of
  compile-time differentiation.
- [docs/SOLVERS.md](SOLVERS.md) -- Levenberg-Marquardt,
  `LmConfig`, and backend selection.
- [arael-sketch-backend/docs/COMMANDS.md](../arael-sketch-backend/docs/COMMANDS.md)
  -- full command reference.
