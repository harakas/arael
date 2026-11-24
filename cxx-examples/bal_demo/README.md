# bal_demo -- bundle adjustment over the C++ interface, with or without a GPU

The C++ twin of `examples/bal_demo.rs`: a BAL ("Bundle Adjustment in the
Large") problem -- cameras with a rotation, a translation, a focal length
and two radial distortion coefficients, points, and pixel observations --
solved on the host or a GPU. It shows the GPU options of the generated
C++ interface ([docs/CXX.md](../../docs/CXX.md#solving-on-a-gpu)).

The model and the solver are Rust (`model/`, with its generated
interface from `cargo arael export`); loading the file, composing the
scene, choosing the route and reporting are `cxx/main.cpp`. Every run
solves the reduced (Schur) system with the block supernodal Cholesky,
the route the GPU runs, so a run with a GPU and one without take the
same route and agree to rounding.

The vendored `benchmarks/bal/datasets/problem-49-7776-pre.txt` is the
default; `benchmarks/bal/fetch_datasets.sh` fetches the larger Ladybug
problems.

The build needs a Rust toolchain, CMake and a C++17 compiler. A run on
a GPU needs, for `--gpu cuda`, an NVIDIA GPU with the CUDA driver
library and NVRTC (CUDA's run-time compiler); for `--gpu hip`, an AMD
GPU with the HIP runtime and hipRTC from ROCm.

```
cmake -S cxx -B cxx/build -DARAEL_GPU=cuda    # or hip, or cuda,hip; leave it out to build without a GPU
cmake --build cxx/build
./cxx/build/bal_demo --gpu cuda --compare
```

Each solve prints one line: its wall time, the iterations, the cost and
the RMS (root mean square) pixel error before and after, and what ran on
the GPU. `--compare` adds a run without the GPU:

```
BAL problem: 49 cameras, 7776 points, 31843 observations
gpu       ... ms  22 iterations (22 accepted)  cost 1701824.9214 -> 26689.4532  RMS 7.3106 -> 0.9155 px  [reduced, GPU assembly, solves and loop on NVIDIA GeForce RTX 4090 (sm_89)]
cpu       ... ms  22 iterations (22 accepted)  cost 1701824.9214 -> 26689.4532  RMS 7.3106 -> 0.9155 px  [reduced]
final cost relative difference: 1.80e-14
```

The first GPU solve of a process also opens the GPU and loads the
kernels, and the first run on a machine compiles them (they are kept in
a disk cache); `--repeat 2` shows a second solve.

| option | effect |
|---|---|
| `--gpu cuda`, `--gpu hip` | solve on that GPU (the interface built with that runtime in `-DARAEL_GPU`); no other value is taken |
| `--device N` | the GPU's index (0) |
| `--f32-factor` | factor in f32 and refine against the f64 matrix |
| `--df32` | keep the GPU's wide scalar in df32 (two f32) instead of f64 |
| `--no-gpu-assembly` | assemble on the host, solve on the GPU |
| `--repeat N` | solve N times from the initial state |
| `--compare` | add a run without the GPU and print the relative difference of the final costs |
| `--threads N` | threads for the host's assembly and linear solve |
| `--timing` | print each solve's phases: analysis, assembly, linear solve (on a GPU, an attempt's whole GPU work), cost, advance |

`VERBOSE=1` prints the solver's iteration lines. The exit code is 0 when
every solve converged.

After changing the model, regenerate the interface:

```
cd model && cargo arael export
```
