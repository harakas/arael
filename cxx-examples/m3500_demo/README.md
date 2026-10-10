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

Browser (`web/`, live at [arael.mare.ee/pgo](https://arael.mare.ee/pgo/)):
the same graph solved in the page as WebAssembly,
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
their cost on a log scale, gray at the median and red at the worst
percent (the `Log color` toggle makes it linear); the wheel zooms,
a middle double-click fits the view; the vendored M3500, Intel Lab,
City10000 and ais2klinik sets are a dropdown and any 2D g2o file can
be loaded. The
gauge is a very weak lock on pose 0, so every pose moves under a drag.
Every re-solve runs through one `LmSession`.
`web/build.sh` builds the wasm crate (the target and the pinned CLI
are named in `model/wasm/Cargo.toml`). `web/` is then the whole site:
its `pkg`, `arael` and `datasets` entries link to the built module,
the loader and the vendored g2o files under `benchmarks/pgo/datasets/`.
Serve it, or copy it with the links dereferenced:

```
web/build.sh
python3 -m http.server -d web
# http://localhost:8000/
cp -rL web/ /srv/www/arael-demo       # a deployable copy
```

After changing the model, regenerate the interfaces:

```
cd model && cargo arael export
```
