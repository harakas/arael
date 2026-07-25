# slam2d_simple_demo -- the C++ and Python editions

The twins of `examples/slam2d_simple_demo.rs`: a robot drives an arc
with drifting odometry while a forward camera reports bearings to
building corners; SLAM recovers the path and the corners together.

The split: the model and the solver are Rust (`model/`, with its
generated `capi/`, `cxx/` and `python/` interfaces from `cargo arael
export`); composing the problem, reading the results, and plotting
are the driver's job -- `cxx/main.cpp` and `python/main.py` do the
same things in their own language. Each synthetic world uses its own
RNG, so the numbers differ between editions -- same shape, same
behavior.

Both print the solver's pretty report (status, cost, per-phase
timing, the accept/reject timeline), per-pose and per-landmark errors
against ground truth, and plot the map with 95% covariance ellipses
in the Rust example's style (`slam2d_simple_cxx.eps` /
`slam2d_simple_py.eps`).

C++ (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S cxx -B cxx/build
cmake --build cxx/build
./cxx/build/slam2d_simple_demo
```

Python (needs the capi cdylib built once):

```
cargo build --release -p slam2d-simple-capi
python3 python/main.py
```

After changing the model, regenerate the interfaces:

```
cd model && cargo arael export
```
