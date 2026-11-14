# devmap data formats

Two formats: the trace a traced solve writes (JSON lines), and the case the
viewer loads (one JavaScript file per case). `gen/devmap.py` turns the
first into the second.

## 1. Case (what the viewer loads)

A case is one solve of one problem on one device: its device memory, the
kernels, and the steps of each phase of the solve. A case file calls
`DEVMAP.add({...})` with the object below; `data/cases.js` lists the case
files (`DEVMAP_FILES = ["bal-49-f32-cpu.js", ...]`), so the viewer works
from `file://` with no server. `gen/devmap.py` writes both; without
`data/cases.js` the viewer loads `data/sample.js`.

```
{
  schema: 1,
  id: "bal-49-f32-cpu",
  title: "bal Ladybug-49, f32 model, CPU device, probed",
  meta: {
    problem: "bal Ladybug-49", model: "f32", runtime: "cpu", device: "cpu",
    wide: "f64",          // the route's wide scalar: f64 or df32
    probe: true,          // measured access data present (CPU device)
    timed: true,          // the launches were timed on the device (time_us in the steps)
    patterns_from: "bal-49-f32-cpu",   // optional, a case without measured accesses: the probed
                                       // case its borrowed patterns come from
    commit: "faf42737", generated: "2026-11-13",
    notes: ["..."]        // free text lines
  },

  // compile units: one kernel source compiled with one set of defines
  units: [
    { id: 0, name: "schur", file: "arael-gpu/kernels/schur.cpp",
      defines: "#define SCHUR_F32 1\n...",
      types: { V: "f32", W: "f64", R: "f32" },   // typedefs resolved
      kernels: ["k_schur_elim", "k_schur_w"] }
  ],

  // kernels by name
  kernels: {
    k_schur_w: {
      unit: 0, file: "arael-gpu/kernels/schur.cpp", line: 172,
      doc: "the comment block above the kernel",
      params: [
        { name: "plan", ctype: "const u32*", type: "u32", ptr: true, const: true },
        { name: "no", ctype: "u32", type: "u32", ptr: false, const: true }
      ],
      source: "the kernel's text"            // optional
    }
  },

  // device memory: blocks cut into named sections
  blocks: [
    { id: 3, name: "schur.work", owner: "GpuSchur", bytes: 123456,
      doc: "the reduction's values",
      sections: [
        { i: 0, name: "h", at: 0, bytes: 4000, elem: 4, count: 1000,
          ty: "f32",        // the stored scalar: u32, f32, f64, df32, bytes
          role: "V",        // the kernel-side typedef, or ""
          group: "",        // optional grouping label (e.g. "structure")
          record: 3,        // optional: the elements come in records of this many
          doc: "H's tile values",
          // optional, index sections (u32, u64): what kind of sequence the words are (below)
          content: { kind: "arithmetic", n: 1000, base: 0, stride: 3 },
          // with content: the share of the words a kernel could compute from their position
          // (a record counts by its fields) and what computes them: i the word's index, r the
          // record's, ? a field nothing computes. fields: a record section's formulas, a field each
          computable: 1, formula: "3 i",
          about: "Content: word i is 3 i.",   // with content: the classification in words
          // optional, index sections: the words as they were on the device when the block
          // was described. head: all of them when at most 80, else the first 64; tail: the
          // last 16
          sample: { head: [0, 3, 6, 9], tail: [2991, 2994, 2997] } }
      ] }
  ],

  // optional: index sequences (a section's words, or one field of its records: field, -1 for
  // the section's own words) that hold the same n words or a multiple of them. A member's
  // words are scale / scale' times another member's
  copies: [ { n: 85217, members: [ { block: 2, section: 9, field: -1, scale: 9 },
                                    { block: 4, section: 16, field: -1, scale: 1 } ] } ],

  // scope documentation, by path
  scopes: { "attempt": { title: "Attempt", doc: "one damped solve" },
            "attempt/reduce": { title: "Schur reduction", doc: "..." } },

  // phases: the top-level scopes, in order of first occurrence
  phases: [
    { id: "attempt", title: "Attempt", doc: "...",
      times: 19,               // occurrences in the solve
      variants: 1,             // distinct step sequences seen (the steps are the most frequent one's)
      steps: [ STEP, ... ] }
  ],

  // how the phases follow each other, for the overview diagram
  flow: [ { phase: "setup" }, { phase: "start" },
          { loop: "LM iterations", body: [
              { phase: "linearize" },
              { loop: "attempts until a step is accepted", body: [ { phase: "attempt" } ] },
              { phase: "accept" } ] },
          { phase: "finish" } ],

  // dataflow: who produced what a step reads
  edges: [ { from: "s12", to: "s15", block: 3, section: 0 },
           { from: "host", to: "s3", block: 3, section: 1 } ],

  findings: [
    { level: "warn",            // bug, warn, info
      title: "short", text: "longer",
      refs: [ { kind: "section", block: 3, section: 4 },
              { kind: "kernel", name: "k_schur_w" },
              { kind: "step", id: "s15" } ] }
  ]
}
```

`content` says what kind of sequence an index section's words are:

| kind | fields | meaning |
|---|---|---|
| `constant` | `value` | every word is `value` |
| `arithmetic` | `base`, `stride` | word i is `base + i * stride` |
| `pieces` | `pieces`: `[first index, base, stride]` each | at most four such stretches, eight words each at least |
| `records` | `period`, `declared`, `fields` | records of `period` words; each field (words j, j + period, ...) has a kind of its own, without `n`. `declared`: the section's owner gave the width; else it was found from the data |
| `monotone`, `other` | `runs`, `distinct`, `gcd`, `hash` | non-decreasing, or neither |

`n` is the word count. `runs`: the stretches of consecutive words with one
step between neighbours (0, 3, 6, 9, 13, 17 is two runs; one run would be
arithmetic). `gcd`: the greatest common divisor of the words. `hash`: of
the words divided by `gcd`: sequences of equal length and hash hold the
same words or a multiple of them (`copies`).

```
{ kind: "records", n: 142269, period: 3, declared: true, fields: [
    { kind: "monotone", runs: 30011, distinct: 47423, gcd: 9, hash: "19e9fab072285dfc" },
    { kind: "constant", value: 3 }, { kind: "arithmetic", base: 3348, stride: 3 } ] }
computable: 0.667, formula: "(?, 3, 3348 + 3 r)", fields: ["?", "3", "3348 + 3 r"]
```

A STEP is one of:

```
// kernel launches: consecutive launches of one kernel in one scope are one step
{ id: "s15", kind: "launch", kernel: "k_schur_w",
  scope: ["attempt", "reduce"],   // the scope path, the phase first
  launches: 1,                    // launches merged into this step
  grid: [31], threads: [256], shared: 0,   // geometry of the first launch
  geoms: 1,                       // distinct geometries among the merged launches
  lane: "main",                   // or "side"
  graph: true,                    // runs inside a captured graph
  interleaved: true,              // optional: its launches alternate with other kernels' (a factor's levels),
                                  // merged into one step per kernel in first-launch order
  args: [ { param: "plan", block: 2, at: 0, section: 0 },
          { param: "no", value: "7776" } ],
  access: [                       // per (block, section) touched
    { block: 3, section: 0,
      src: "measured",            // or "static" (from const-ness and COL macros)
      r: { n: 46656, bytes: 186624, unique: 46656, lo: 0, hi: 186623,
           threads: 7776, sizes: [4],
           seq: 0.83, same: 0.0,  // fractions of a thread's accesses adjacent to / equal to its previous one
           pattern: "serial" },   // coalesced, serial, gather, scatter, broadcast, strided, mixed
      w: null }
  ],
  time_us: 12.5 }                 // optional (a timed trace): the mean device time of one of its
                                  // launches, in microseconds, over the phase's occurrences

// host to device copy
{ id: "s3", kind: "upload", scope: ["start"], block: 4, section: 1, at: 16, bytes: 19332,
  sections: [1, 2],               // optional: every section the copy covers (section is the first)
  via: "pageable" }               // or "pinned"

// device to host copy: the host waits for the device here
{ id: "s40", kind: "download", scope: ["attempt"], block: 4, section: 8, at: 0, bytes: 64 }
```

`access[].r` and `.w` fields: `n` accesses, `bytes` moved, `unique`
distinct elements, `lo`..`hi` the byte range touched within the section,
`threads` distinct device threads, `sizes` access widths in bytes.

A static entry has no counts, only a pattern:

```
r: { pattern: "unknown" }                    // nothing known but the direction
r: { pattern: "gather", borrowed: true }     // the pattern measured for this kernel and section
                                             // in the probed case (meta.patterns_from)
w: { pattern: "unknown", unseen: true }      // the signature allows the access; the probed case's
                                             // launches of the kernel never made it
```

Patterns:

| pattern | meaning |
|---|---|
| coalesced | adjacent threads touch adjacent elements, each element about once |
| serial | each thread walks a contiguous run |
| strided | regular, but not adjacent |
| gather / scatter | index-driven reads / writes, irregular |
| broadcast | every thread reads the same few elements |
| mixed | none of the above dominates |

## 2. Trace (what a traced solve writes)

Set `ARAEL_GPU_TRACE=<file>` (arael built with the `gpu-trace` feature):
every device operation of the process is appended as one JSON object per
line. `ARAEL_CPU_PROBE=1` with the CPU device adds the measured accesses.
See `arael-gpu/src/trace.rs` for the events.
