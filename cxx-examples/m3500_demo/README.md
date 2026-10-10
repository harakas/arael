# m3500_demo -- the C++ and Python editions

The twins of `examples/m3500_demo.rs`: the classic M3500
Manhattan-world 2D pose graph -- 3500 poses, ~5450 relative SE2
measurements, gauge fixed by a soft prior on pose 0.

The split: the model and the solver are Rust (`model/`, with its
generated interfaces from `cargo arael export`); loading the g2o file
(the vendored `arael` g2o reader in each language), composing the
graph, and reporting are `cxx/main.cpp` / `python/main.py`. There is
no randomness, so all editions match the Rust example digit for
digit.

The vendored dataset under `benchmarks/pgo/datasets/` is the default;
pass a path to run any other 2D g2o file. `--weighted` uses the
dataset's sqrt-info weights, `--dump out.txt` writes the solved
poses, `VERBOSE=1` prints solver iteration lines. Writes `m3500.eps`
(before = gray, after = black).

C++ (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S cxx -B cxx/build
cmake --build cxx/build
./cxx/build/m3500_demo
```

Python (needs the capi cdylib built once):

```
cargo build --release -p m3500-demo-capi
python3 python/main.py
```

Browser (`web/`): the same graph solved in the page as WebAssembly,
through the generated JavaScript interface
([docs/WASM.md](../../docs/WASM.md)). The actions are buttons in a
column on the left, each with a tooltip and its key. Drag a pose and
a lock, a soft prior, follows the pointer and stays where it is
released; `L` locks the selection in place, `F` fixes it (its
parameters leave the solve),
`C` clears; Ctrl+Z (Cmd+Z on a Mac) undoes any of these, Shift with
it or Ctrl+Y redoes; a click selects without locking; shift-click and
shift-drag select; a locked or fixed pose
is picked before the plain poses around it; the links are colored by
their cost, gray to red, red being the worst percent; the wheel zooms,
a middle double-click fits the view; any 2D g2o file can be loaded. The
gauge is a very weak lock on pose 0, so every pose moves under a drag.
Every re-solve runs through one `LmSession`.
Build the wasm crate once (the target and the pinned CLI are named in
`model/wasm/Cargo.toml`), then serve this directory; `web/datasets`
links to the vendored g2o files under `benchmarks/pgo/datasets/`:

```
cd model/wasm
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --weak-refs --out-dir pkg \
    target/wasm32-unknown-unknown/release/m3500_demo_wasm.wasm
cd ../..
python3 -m http.server
# http://localhost:8000/web/
```

After changing the model, regenerate the interfaces:

```
cd model && cargo arael export
```
