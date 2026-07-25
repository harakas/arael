# slam2d_simple_demo, the C++ edition

The C++ twin of `examples/slam2d_simple_demo.rs`: a robot drives an
arc with drifting odometry while a forward camera reports bearings to
building corners; SLAM recovers the path and the corners together.

The split: the model and the solver are Rust (`model/`, with its
generated `capi/` and `cxx/` interface from `cargo arael export`);
composing the problem, reading the results, and plotting are C++
(`main.cpp`). The synthetic world uses its own RNG, so the numbers
differ slightly from the Rust example's -- same shape, same behavior.

Build and run (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S . -B build
cmake --build build
./build/slam2d_simple_demo
```

Prints per-pose and per-landmark errors against ground truth and
writes `slam2d_simple_cxx.eps` (gray = ground truth, blue = solved
path, red = solved corners).

After changing the model, regenerate the interface:

```
cd model && cargo arael export
```
