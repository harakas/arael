# slam_demo_gm, the C++ edition

The C++ twin of `examples/slam_demo_gm.rs`: a synthetic
visual-inertial SLAM problem -- an S-curve trajectory watched by 5
cameras with 360-degree coverage, GPS, wheel odometry, and
accelerometer tilt, with 50% outlier feature associations at 30x pixel
noise. Feature and GPS residuals wear a Geman-McClure block loss;
landmarks use the anchored inverse-depth parameterization. A graduated
three-pass ramp loosens then tightens the feature weights, re-anchoring
landmarks between passes.

The split: the model and the solver are Rust (`model/`, with its
generated `capi/` and `cxx/` interface from `cargo arael export`);
composing the problem, the ramp, and the error reports are C++
(`main.cpp`, with `arael::Camera` from the vendored geometry header for
the synthetic world). The world uses its own RNG, so the numbers differ
from the Rust example's -- same shape, same behavior.

Build and run (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S . -B build
cmake --build build
./build/slam_demo_gm
```

Options: `--solver <dense|sparse>`, `--loss <gm|cauchy>`, `--poses N`,
`--landmarks N`, `--seed N`; `SINGLE_PASS=1` skips the ramp (and fails
here -- with half the observations wrong, the ramp is what carries the
landmarks into their inlier basins). The solver runs verbose; per-pass
iteration lines come from the Rust side.

Prints per-pose and relative-pose errors against ground truth, and per
landmark the position error plus the 95% uncertainty ellipsoid axes
from the covariance API -- the landmark x pose cross-covariance cancels
the shared gauge uncertainty, leaving uncertainty relative to the pose.

After changing the model, regenerate the interface:

```
cd model && cargo arael export
```
