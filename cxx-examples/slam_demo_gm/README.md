# slam_demo_gm -- the C++ and Python editions

The twins of `examples/slam_demo_gm.rs`: a synthetic visual-inertial
SLAM problem -- an S-curve trajectory watched by 5 cameras with
360-degree coverage, GPS, wheel odometry, and accelerometer tilt,
with 50% outlier feature associations at 30x pixel noise. Feature and
GPS residuals wear a Geman-McClure block loss; landmarks use the
anchored inverse-depth parameterization. A graduated three-pass ramp
loosens then tightens the feature weights, re-anchoring landmarks
between passes.

The split: the model and the solver are Rust (`model/`, with its
generated interfaces from `cargo arael export`); composing the world,
the ramp, and the error reports are `cxx/main.cpp` /
`python/main.py`, with the vendored `arael` Camera for the synthetic
world. Each world uses its own RNG -- same shape, same behavior,
numbers differ.

Options: `--solver <dense|sparse>`, `--loss <gm|cauchy>`, `--poses N`,
`--landmarks N`, `--seed N`; `SINGLE_PASS=1` skips the ramp (and
fails here -- with half the observations wrong, the ramp is what
carries the landmarks into their inlier basins). The solver runs
verbose. Both editions report per-pose and relative-pose errors, and
per landmark the position error plus the 95% uncertainty ellipsoid
axes from the covariance API (the landmark x pose cross-covariance
cancels the shared gauge uncertainty).

C++ (needs cmake, a C++17 compiler, and a Rust toolchain):

```
cmake -S cxx -B cxx/build
cmake --build cxx/build
./cxx/build/slam_demo_gm
```

Python (needs the capi cdylib built once):

```
cargo build --release -p slam-demo-gm-capi
python3 python/main.py
```

After changing the model, regenerate the interfaces:

```
cd model && cargo arael export
```
