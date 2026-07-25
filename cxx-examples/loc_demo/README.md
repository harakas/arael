# loc_demo -- the C++ and Python editions

The twins of `examples/loc_demo.rs`: localization against a known map
-- the same synthetic S-curve world as the SLAM demos, but the
landmarks are fixed data, so there is no gauge freedom and absolute
pose errors are meaningful. Feature residuals wear the Starship
robustifier (`gamma * atan(r / gamma)`); a graduated three-pass ramp
loosens then tightens the feature weights.

The Hessian is block-tridiagonal (fixed map, no loop closures), so
the solves run on the band solver (`solve_band(11)` -- half-bandwidth
2*6 - 1 with 6-parameter poses), and the last pose's 1-sigma
uncertainty comes from the TriDiagonal covariance mode + `std_dev`.

The split: the model and the solver are Rust (`model/`, with its
generated interfaces from `cargo arael export`); composing the
problem, the ramp, and the reports are `cxx/main.cpp` /
`python/main.py`. Each world uses its own RNG -- same shape, same
behavior, numbers differ.

C++ (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S cxx -B cxx/build
cmake --build cxx/build
./cxx/build/loc_demo
```

Python (needs the capi cdylib built once):

```
cargo build --release -p loc-demo-capi
python3 python/main.py
```

After changing the model, regenerate the interfaces:

```
cd model && cargo arael export
```
