#!/usr/bin/env python3
"""Count what a GPU program asks of the CUDA driver, phase by phase.

Runs the command under `nsys profile --trace=cuda`, exports the trace, and
reports per phase: kernel launches, blocking synchronizations, copies each
way (count and MB), device and pinned-host allocations and frees, module
loads. These are the budgets of docs/dev/GPU.md section 7.6.

Phases are cut at marker copies: host-to-device copies of MARKER + id bytes
(id < 4096), which arael-gpu's examples write between phases (see
arael-gpu/examples/repeat_solves.rs). Without markers the whole run is one
phase.

    scripts/gpu_trace.py -- target/release/examples/repeat_solves
    scripts/gpu_trace.py --keep out/trace -- ./target/release/bal-bench

The profiler needs no special permission for API tracing (unlike ncu's
hardware counters). Needs nsys on PATH or at /usr/local/bin/nsys.
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

MARKER = 1_000_003
MARKER_RANGE = 4096

# CUDA driver and runtime API names, by what they cost
CATEGORIES = {
    "launch": ("cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernel", "cuGraphLaunch"),
    "sync": ("cuEventSynchronize", "cuStreamSynchronize", "cuCtxSynchronize", "cudaDeviceSynchronize",
             "cudaStreamSynchronize", "cudaEventSynchronize"),
    "dev alloc": ("cuMemAlloc_v2", "cuMemAllocAsync", "cuMemAllocPitch_v2", "cuMemCreate", "cudaMalloc"),
    "dev free": ("cuMemFree_v2", "cuMemFreeAsync", "cudaFree"),
    "pinned alloc": ("cuMemAllocHost_v2", "cuMemHostAlloc", "cuMemHostRegister_v2", "cudaMallocHost", "cudaHostAlloc"),
    "pinned free": ("cuMemFreeHost", "cuMemHostUnregister", "cudaFreeHost"),
    "module load": ("cuModuleLoadData", "cuModuleLoadDataEx", "cuLibraryLoadData"),
    "graph inst": ("cuGraphInstantiate", "cuGraphInstantiate_v2", "cuGraphInstantiateWithFlags", "cuGraphInstantiateWithParams"),
}


def nsys_binary():
    for cand in (shutil.which("nsys"), "/usr/local/bin/nsys", "/usr/local/cuda/bin/nsys"):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("nsys not found")


def profile(cmd, out):
    nsys = nsys_binary()
    run = subprocess.run([nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                          "--force-overwrite", "true", "-o", out] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write("".join(l for l in run.stdout.splitlines(True) if not l.startswith(("Collecting", "Generating", "Generated", "Processing", "[", "\t"))))
    if run.returncode != 0:
        sys.exit(f"the profiled command failed ({run.returncode})")
    subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", out + ".sqlite", out + ".nsys-rep"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out + ".sqlite"


def phase_name(mid):
    if mid is None:
        return "before the first marker"
    if mid == MARKER_RANGE - 1:
        return "end"
    phase, kind = (mid - 1) // 2, (mid - 1) % 2
    return f"structure {phase // 16} solve {phase % 16} {'setup' if kind == 0 else 'attempts'}"


def report(db_path):
    db = sqlite3.connect(db_path)
    api = db.execute("SELECT r.start, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id ORDER BY r.start").fetchall()
    copies = db.execute("SELECT start, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
    markers = [(t, b - MARKER) for t, k, b in copies if k == 1 and MARKER <= b < MARKER + MARKER_RANGE]
    bounds = [(None, None)] + markers  # (start time, id)
    which = {}
    for name, calls in CATEGORIES.items():
        for c in calls:
            which[c] = name

    def phase_of(t):
        lo = 0
        for i, (mt, _) in enumerate(bounds[1:], 1):
            if t >= mt:
                lo = i
            else:
                break
        return lo

    cols = ["launch", "sync", "h2d", "h2d MB", "d2h", "d2h MB", "dev alloc", "dev free", "pinned alloc", "pinned free", "module load", "graph inst"]
    counts = [dict.fromkeys(cols, 0) for _ in bounds]
    for t, name in api:
        cat = which.get(name)
        if cat:
            counts[phase_of(t)][cat] += 1
    for t, kind, b in copies:
        if kind == 1 and MARKER <= b < MARKER + MARKER_RANGE:
            continue
        p = counts[phase_of(t)]
        if kind == 1:
            p["h2d"] += 1
            p["h2d MB"] += b / 1e6
        elif kind == 2:
            p["d2h"] += 1
            p["d2h MB"] += b / 1e6
    width = max(len(phase_name(m)) for _, m in bounds)
    print(f"{'phase':{width}}  " + "  ".join(f"{c:>12}" for c in cols))
    for (_, mid), cnt in zip(bounds, counts):
        if all(v == 0 for v in cnt.values()) and mid is not None and mid == MARKER_RANGE - 1:
            continue
        cells = [f"{cnt[c]:12.1f}" if c.endswith("MB") else f"{cnt[c]:12d}" for c in cols]
        print(f"{phase_name(mid):{width}}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", help="keep the trace under this path (without extension)")
    ap.add_argument("--sqlite", help="report an existing exported trace instead of running a command")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.sqlite:
        report(a.sqlite)
        return
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command to profile")
    with tempfile.TemporaryDirectory() as tmp:
        out = a.keep or os.path.join(tmp, "trace")
        report(profile(cmd, out))


if __name__ == "__main__":
    main()
