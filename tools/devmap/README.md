# devmap

A map of what a device (GPU) solve does: the kernels in launch order, the
device memory they read and write, how much and in what pattern, and in
which scalar types. For looking around: finding duplicated data, plan
words read over and over, sections nobody reads, types wider than needed.

Open `index.html` in a browser (from disk, no server). Pick a case at the
top left; generate the cases first (Regenerating, about 16 minutes).

## Views

- Overview: the phases of a solve (setup, linearize, attempt, ...) between
  host and device, with the uploads, the read-backs (where the host waits)
  and the totals.
- Shares: what each kernel takes of an iteration (or of a phase, or of the
  whole solve), by area: a treemap of phase, stage and kernel, and the same
  kernels ranked with their figures. The area is the device time on a
  timed case (a GPU's), or the bytes read or written on a probed one.
- Pipeline: the steps of a phase in order, with the dataflow between them.
  Click a step for the kernel's documentation, arguments, measured
  accesses and source. A step of several launches (`k_syrk x93`) lists
  each one under Launches: its place in the phase, its grid, the
  arguments that differ between them and, on a timed case, its time.
- Sequence: the launches and host copies of a phase one by one, in the
  order one occurrence of the phase issues them (the Pipeline shows a
  kernel's launches as one step): the kernel with its number among its
  step's launches, its stage, its grid, the arguments that differ between
  the step's launches, and on a timed case its time and the device time
  elapsed. Click a row: all the launches of its step are marked.
- Access matrix: steps against sections: who reads and writes what, when.
- Memory: every device block cut into its named sections, by type. The
  column `computable` says of an index section (u32, u64) what a kernel
  could compute instead of reading it: a formula in the word's index i
  (`27 i`, `9; from 138: 3`), per field in the record's index r for
  records (`2/3 (?, 3, 3348 + 3 r)`: two of three fields), and other
  sections that hold the same words or a multiple of them
  (`= 9 x image.C0_S0`); `no` where neither. Click a section for the words
  themselves, the first 64 and the last 16, with the steps between
  neighbours.
- Kernels: all kernels with their units' types and totals.
- Findings: checks computed from the data.

## Cases

A case is one solve of one problem on one device (`data/*.js`, listed in
`data/cases.js`). The cases are generated from traces and are not in the
repository (see Regenerating): until they are there, the viewer shows
`data/sample.js`, a small hand-made case.

| case | what it shows |
|---|---|
| `*-cpu` | The CPU device with its access probe: every read and write of every launch is measured (counts, distinct elements, pattern). Small problems. |
| `*-cuda`, `*-hip` | A GPU at the benchmark's size: the launch sequence, the buffers and their sizes, and each launch's time on the device. A GPU does not count accesses: which sections a kernel reads or writes is from its signature, amounts read `n/m` (not measured). |

A GPU's case borrows its access patterns from the probed case of the same
bench and model (`gen/devmap.py --patterns-from`): the pattern measured
for the same kernel and section, shown in italics. An access the kernel's
signature allows and the probed case never saw (a failure flag, an unused
column) has a dashed cell in the access matrix. Kernels the probed problem
does not launch are listed in the case's findings: nothing is measured for
them.

The measured counts are those of the host compiler's code (clang, no
vectorization): a GPU compiler may keep in a register what this one
reloads, or the reverse. Patterns and volumes carry over; exact counts do
not.

A GPU's launches are timed one at a time (`ARAEL_GPU_TRACE_TIME=1`,
`Device::time_launches`): each runs alone between two device events, a
captured graph's launches one by one. A kernel's time is its own; the
solve's wall time, where launches overlap and the host works in between,
is not their sum.

## Regenerating

```sh
(cd benchmarks/bal  && cargo build -r --features gpu-cpu,gpu-cuda,gpu-trace --target-dir target-trace)
(cd benchmarks/slam && cargo build -r --features gpu-cpu,gpu-cuda,gpu-trace --target-dir target-trace)
tools/devmap/gen/run.sh cpu cuda
```

The probed CPU cases need `clang++` (`ARAEL_CPU_PROBE_CXX` names another).
Traces go to `/tmp/devmap-traces` (`DEVMAP_TRACES`); `gen/regen.sh` builds
the cases again from them without running the benches. The probed
problems are Ladybug-49 and Ladybug-138 (`BAL_CPU_SIZES="49 138"`, about
1.5 and 5 minutes a solve) and slam with 120 poses (`SLAM_CPU_POSES`).

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

A `libs/` directory there (system libraries unpacked from their packages)
is put on the library path when the system lacks them.

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
  the scopes the solver opens (`Device::scope_begin`).
- Each owner of a block names its sections (`Device::describe`): the
  model's image by column, the reduction's and the factor's plans by
  array, the work blocks by vector.
- The CPU device's probe (`arael-gpu/src/cpu_probe.rs`) compiles the
  kernels with clang's load and store tracing and counts every access per
  launch and section.
- `gen/devmap.py` merges launches into steps, resolves kernel parameters
  and types from the unit's source and defines, classifies the access
  patterns, derives the dataflow and runs the checks. Formats:
  `SCHEMA.md`.
