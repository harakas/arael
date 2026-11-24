# devmap

A viewer of what a GPU solve does on the device: the kernels in launch
order, the device memory they read and write (how much, in what pattern,
in which scalar types), and the copies between host and device. It is for
finding duplicated data, index data read over and over, memory nothing
reads, and types wider than needed.

Open `index.html` in a browser (from disk, no server) and pick a case at
the top left. The cases are generated (see Regenerating); until they are,
the viewer shows `data/sample.js`, a small hand-made case.

## Terms

- case: one traced solve of one problem, on a GPU or on the CPU device
  (arael's GPU kernels compiled for the CPU, feature `gpu-cpu`).
- probed case: a case on the CPU device with its access probe: every read
  and write of every launch is counted.
- timed case: a case on a GPU: every launch is timed on the device; reads
  and writes are not counted.
- phase: a part of the solve the solver marks (setup, linearize, attempt,
  ...); a stage is a part marked inside a phase.
- step: consecutive launches of one kernel in the same marked part
  (`k_syrk x93` is 93 launches), or one copy between host and device.
- block: one device allocation. section: a named part of a block, given by
  the code that owns the block: the model's data on the device (its image)
  has a section per column, the plans a section per array, the work
  blocks a section per vector.
- plan: the integer data the Schur reduction and the factorization write
  to the device at setup, which tells their kernels where to read and
  write.
- unit: one kernel source file compiled with one set of defines; the
  defines fix the kernel's scalar types.

## Views

- Overview: the phases of a solve between host and device, with the
  uploads, the read-backs (where the host waits) and the totals.
- Shares: what each kernel takes of an iteration (or of a phase, or of the
  whole solve), by area: a treemap of phase, stage and kernel, and the same
  kernels ranked with their figures. The area is the device time on a
  timed case, or the bytes read or written on a probed one.
- Pipeline: the steps of a phase in order, with the dataflow between them.
  Click a step for the kernel's documentation, arguments, measured
  accesses and source. A step of several launches lists each one under
  Launches: its place in the phase, its grid, the arguments that differ
  between them and, on a timed case, its time.
- Sequence: the launches and copies of a phase one by one, in the order
  one occurrence of the phase issues them: the kernel with its number
  among its step's launches, its stage, its grid, the arguments that
  differ between the step's launches, and on a timed case its time and
  the device time elapsed. Click a row: all the launches of its step are
  marked.
- Access matrix: steps against sections: who reads and writes what, when.
- Memory: every block cut into its sections, by type, with the column
  `computable` (below). Click a section for its words, the first 64 and
  the last 16, with the steps between neighbours.
- Kernels: every kernel with its unit's scalar types and its totals.
- Findings: checks computed from the data.

The column `computable` says of an index section (u32, u64) what a kernel
could compute instead of reading it, or `no`:

| example | meaning |
|---|---|
| `27 i` | word i is 27 i |
| `9; from 138: 3` | the words are 9, and from word 138 on they are 3 |
| `2/3 (?, 3, 3348 + 3 r)` | records of three fields, r the record's index: two of the three fields are computable, `?` is the one that is not |
| `= 9 x image.C0_S0` | the words are 9 times those of another section |

## Cases

`gen/devmap.py` writes each case to `data/<case>.js` and lists them in
`data/cases.js`; the generated files are not in the repository.

| case | what it shows |
|---|---|
| `*-cpu` | Probed: the counts, distinct elements and pattern of every read and write. Small problems. |
| `*-cuda`, `*-hip`, `*-metal` | Timed, at the benchmark's size: the launch sequence, the buffers and their sizes, and each launch's time on the device. Which sections a kernel reads or writes comes from its signature; the amounts read `n/m` (not measured). |

A timed case borrows its access patterns from the probed case of the same
benchmark and model (`gen/devmap.py --patterns-from`): the pattern
measured for the same kernel and section, shown in italics. An access the
kernel's signature allows and the probed case never saw (a failure flag,
an unused column) has a dashed cell in the access matrix. Kernels the
probed problem does not launch are listed in the case's findings: nothing
is measured for them.

The counts are those of the host compiler's code (clang, no
vectorization): a GPU compiler may keep in a register what this one
reloads, or the reverse. Patterns and volumes carry over; exact counts do
not.

A GPU's launches are timed one at a time (`ARAEL_GPU_TRACE_TIME=1`,
`Device::time_launches`): each runs alone between two device events, a
captured graph's launches one by one. A kernel's time is its own; the
solve's wall time, where launches overlap and the host works in between,
is not their sum.

## Regenerating

`gen/run.sh` runs the bal and slam benchmarks with the trace on and turns
each trace into a case. It needs the benchmarks built with the trace, the
CPU device and the GPU runtime to trace:

```sh
(cd benchmarks/bal  && cargo build -r --features gpu-cpu,gpu-cuda,gpu-trace --target-dir target-trace)
(cd benchmarks/slam && cargo build -r --features gpu-cpu,gpu-cuda,gpu-trace --target-dir target-trace)
tools/devmap/gen/run.sh cpu cuda
```

For an AMD GPU build with `gpu-hip` in place of `gpu-cuda` and run
`tools/devmap/gen/run.sh hip`; on a Mac, with `gpu-metal` and
`tools/devmap/gen/run.sh metal`, where the f64 cases take the f32 factor,
refined, as Metal has no f64. How to build and run the benchmarks on a GPU:
[benchmarks/README.md](../../benchmarks/README.md).

The problems, each with the f32 and the f64 model:

- probed (`cpu`): bal Ladybug-49 and Ladybug-138 (`BAL_CPU_SIZES`, default
  `49 138`), slam with 120 poses (`SLAM_CPU_POSES`, default `120`); an
  empty list runs none.
- timed (`cuda`, `hip`, `metal`): bal Ladybug-372, slam with 900 poses.

Ladybug-49 is in the repository; Ladybug-138 and Ladybug-372 are not:
download `problem-138-19878-pre.txt.bz2` and
`problem-372-47423-pre.txt.bz2` from
https://grail.cs.washington.edu/projects/bal/data/ladybug/ into
`benchmarks/bal/datasets` and decompress them (`bunzip2`). The slam scene
is generated.

The probed cases need `clang++` (`ARAEL_CPU_PROBE_CXX` names another).
Traces go to `/tmp/devmap-traces` (`DEVMAP_TRACES`); `gen/regen.sh` builds
the cases again from them without running the benchmarks.
`BAL_CPU_ATTEMPT_SIZES` adds bal problems probed for one attempt, for
those too large to probe a whole solve:

```sh
BAL_CPU_SIZES= SLAM_CPU_POSES= BAL_CPU_ATTEMPT_SIZES=372 tools/devmap/gen/run.sh cpu
```

## Testing the viewer

`test/smoke.sh` opens every case and view in a headless Chromium, light
and dark, selects a step, a kernel and a section in each, reports page
errors and the page's error strip, and writes screenshots (`test/shots`,
or `--out DIR`; `--cases a,b`; `--shots all|first|none`). It needs Node
and Playwright with its Chromium in `~/.local/share/devmap-browser`
(`DEVMAP_PLAYWRIGHT` names another directory):

```sh
cd ~/.local/share/devmap-browser && npm init -y && npm install playwright && npx playwright install chromium
tools/devmap/test/smoke.sh
```

When that directory has `libs/usr/lib/x86_64-linux-gnu` (system libraries
unpacked from their packages, for a system that lacks Chromium's),
`smoke.sh` puts it first on the library path.

## Tracing another program

Build arael with the `gpu-trace` feature and run with
`ARAEL_GPU_TRACE=<file>`: every device operation of the process is written
there, one JSON object per line (`arael-gpu/src/trace.rs`). On the CPU
device (`GpuDevice::cpu()`, feature `gpu-cpu`) `ARAEL_CPU_PROBE=1` adds
the measured accesses. Then:

```sh
tools/devmap/gen/devmap.py solve.trace --id my-case --title "my model, f32, cuda"
```

A trace holds every solve of the process; the case is the one with the
most attempts (`--solve N` picks another).

## How it works

- `arael_gpu::trace::Traced` wraps a `Device` and records allocations,
  copies, launches with their arguments, graph captures and replays, and
  the parts the solver marks (`Device::scope_begin`).
- The code that owns a block names its sections (`Device::describe`).
- The CPU device's probe (`arael-gpu/src/cpu_probe.rs`) compiles the
  kernels with clang's load and store tracing and counts every access per
  launch and section.
- `gen/devmap.py` merges launches into steps, resolves kernel parameters
  and types from the unit's source and defines, classifies the access
  patterns, derives the dataflow and runs the checks. Formats:
  `SCHEMA.md`.
