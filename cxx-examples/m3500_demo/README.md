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

After changing the model, regenerate the interfaces:

```
cd model && cargo arael export
```
