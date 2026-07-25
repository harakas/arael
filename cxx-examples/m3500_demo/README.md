# m3500_demo, the C++ edition

The C++ twin of `examples/m3500_demo.rs`: the classic M3500
Manhattan-world 2D pose graph -- 3500 poses, ~5450 relative SE2
measurements, gauge fixed by a soft prior on pose 0.

The split: the model and the solver are Rust (`model/`, with its
generated `capi/` and `cxx/` interface from `cargo arael export`);
loading the g2o file (`arael::g2o::Dataset2` from the vendored
header), composing the graph, and reporting are C++ (`main.cpp`).
There is no randomness, so the results match the Rust example
digit for digit.

Build and run (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S . -B build
cmake --build build
./build/m3500_demo
```

The vendored dataset under `benchmarks/pgo/datasets/` is the default;
pass a path to run any other 2D g2o file. `--weighted` uses the
dataset's sqrt-info weights, `--dump out.txt` writes the solved poses,
`VERBOSE=1` prints solver iteration lines. Writes `m3500.eps`
(before = gray, after = black).

After changing the model, regenerate the interface:

```
cd model && cargo arael export
```
