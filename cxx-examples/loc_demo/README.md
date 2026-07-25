# loc_demo, the C++ edition

The C++ twin of `examples/loc_demo.rs`: localization against a known
map -- the same synthetic S-curve world as the SLAM demos, but the
landmarks are fixed data, so there is no gauge freedom and absolute
pose errors are meaningful. Feature residuals wear the Starship
robustifier (`gamma * atan(r / gamma)`); a graduated three-pass ramp
loosens then tightens the feature weights.

The Hessian is block-tridiagonal (fixed map, no loop closures), so the
solves run on the band solver (`solve_band(11)` -- half-bandwidth
2*6 - 1 with 6-parameter poses), and the last pose's 1-sigma
uncertainty comes from `CovMode::TriDiagonal` + `std_dev`.

The split: the model and the solver are Rust (`model/`, with its
generated `capi/` and `cxx/` interface from `cargo arael export`);
composing the problem, the ramp, and the reports are C++ (`main.cpp`,
with `arael::Camera` from the vendored geometry header). The world
uses its own RNG, so the numbers differ from the Rust example's --
same shape, same behavior.

Build and run (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S . -B build
cmake --build build
./build/loc_demo
```

After changing the model, regenerate the interface:

```
cd model && cargo arael export
```
