# bal_demo -- bundle adjustment over the C++ interface, CPU or GPU

The C++ twin of `examples/bal_demo.rs`: a BAL ("Bundle Adjustment in the
Large") problem -- cameras with a rotation, a translation, a focal length
and two radial distortion coefficients, points, and pixel observations --
solved on the CPU or a GPU. It is the test bed of the GPU options of the
generated interfaces (docs/CXX.md, GPU).

The model and the solver are Rust (`model/`, with its generated
interface from `cargo arael export`); loading the file, composing the
scene, choosing the route and reporting are `cxx/main.cpp`. Every run
solves the reduced (Schur) system with the block supernodal Cholesky,
the route the GPU runs, so a CPU and a GPU run compute the same thing.

The vendored `benchmarks/bal/datasets/problem-49-7776-pre.txt` is the
default; `benchmarks/bal/fetch_datasets.sh` fetches the larger Ladybug
problems.

```
cmake -S cxx -B cxx/build -DARAEL_GPU=cuda    # or hip; leave it out for a CPU-only build
cmake --build cxx/build
./cxx/build/bal_demo --gpu cuda --compare
```

```
BAL problem: 49 cameras, 7776 points, 31843 observations
gpu      1143.2 ms  22 iterations (22 accepted)  cost 1701824.9214 -> 26689.4532  RMS 7.3106 -> 0.9155 px  [reduced, GPU factor, GPU assembly and loop]
cpu       260.0 ms  22 iterations (22 accepted)  cost 1701824.9214 -> 26689.4532  RMS 7.3106 -> 0.9155 px  [reduced]
final cost relative difference: 6.45e-14
```

The first GPU solve of a process includes compiling the kernels;
`--repeat 2` shows a second one.

| option | effect |
|---|---|
| `--gpu cuda`, `--gpu hip` | solve on that GPU (the interface built with `-DARAEL_GPU` of the same name) |
| `--device N` | the GPU's index (0) |
| `--f32-factor` | factor in f32 and refine against the f64 matrix |
| `--df32` | keep the GPU's wide values in df32 (two floats) instead of f64 |
| `--no-gpu-assembly` | assemble on the CPU, factor on the GPU |
| `--repeat N` | solve N times from the initial state |
| `--compare` | add a CPU run and print the relative difference of the final costs |
| `--threads N` | CPU threads for the assembly and the linear solve |
| `--timing` | print each solve's phases: analysis, assembly, linear solve (on the GPU route an attempt's whole device work), cost, advance |

`VERBOSE=1` prints the solver's iteration lines. The exit code is 0 when
every solve converged.

After changing the model, regenerate the interface:

```
cd model && cargo arael export
```
