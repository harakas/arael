#!/usr/bin/env python3
# M3500 2D pose-graph optimization over the generated arael Python
# interface -- the Python twin of examples/m3500_demo.rs and the C++
# demo. The model and solver are Rust (the cxx-examples model crate,
# shared); loading the g2o file, composing the graph, and reporting
# are Python. No randomness, so the results match the Rust and C++
# twins digit for digit.
#
#   python3 main.py [path/to/file.g2o] [--weighted] [--dump out.txt]
#   VERBOSE=1 for solver iteration lines.
#
# Needs the capi cdylib: `cargo build --release -p m3500-demo-capi`
# in ../../cxx-examples/m3500_demo (or set ARAEL_CAPI).
import math
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "model", "python"))

from m3500_demo import graph as g  # noqa: E402
from m3500_demo.arael import g2o  # noqa: E402

DEFAULT_DATASET = os.path.join(_here, "..", "..", "..", "benchmarks", "pgo",
                               "datasets", "input_M3500_g2o.g2o")


def rad_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def load_g2o(path, weighted, graph):
    ds = g2o.Dataset2.load(path)
    # Reserve the known counts: growing a collection one push at a time
    # reallocates repeatedly.
    graph.poses.reserve(len(ds.poses))
    graph.edges.reserve(len(ds.deltas))
    for p in ds.poses:
        pose = graph.poses.push()
        pose.pos = p.t
        pose.rot_angle = p.th
        if graph.prior is None:
            prior = graph.make_prior()
            prior.p = graph.poses.ref_at(0)
            prior.pos = p.t
            prior.th = p.th
    for d in ds.deltas:
        # Exact whitening for any symmetric information matrix: rows of
        # diag(w) * R^T from its eigendecomposition.
        s0 = g2o.vect3d(1.0, 0.0, 0.0)
        s1 = g2o.vect3d(0.0, 1.0, 0.0)
        s2 = g2o.vect3d(0.0, 0.0, 1.0)
        if weighted:
            r, w = d.eigen_sqrt_info()
            s0 = g2o.vect3d(r[0].x * w.x, r[1].x * w.x, r[2].x * w.x)
            s1 = g2o.vect3d(r[0].y * w.y, r[1].y * w.y, r[2].y * w.y)
            s2 = g2o.vect3d(r[0].z * w.z, r[1].z * w.z, r[2].z * w.z)
        e = graph.edges.push()
        e.a = graph.poses.ref_at(d.a)
        e.b = graph.poses.ref_at(d.b)
        e.delta = d.dt
        e.dth = d.dth
        e.s0 = s0
        e.s1 = s1
        e.s2 = s2


def metrics(graph):
    """Plain least-squares cost and the Huber(1.0) block metric,
    straight from the data."""
    ls = 0.0
    huber = 0.0

    def block(r0, r1, r2):
        nonlocal ls, huber
        s = r0 * r0 + r1 * r1 + r2 * r2
        ls += s
        huber += 2.0 * math.sqrt(s) - 1.0 if s > 1.0 else s

    for e in graph.edges:
        a = graph.poses.get(e.a)
        b = graph.poses.get(e.b)
        sa, ca = math.sin(a.rot_angle), math.cos(a.rot_angle)
        delta = e.delta
        dx = b.pos.x - a.pos.x
        dy = b.pos.y - a.pos.y
        lx = ca * dx + sa * dy - delta.x
        ly = -sa * dx + ca * dy - delta.y
        rr = rad_diff(b.rot_angle, a.rot_angle + e.dth)
        s0, s1, s2 = e.s0, e.s1, e.s2
        block(s0.x * lx + s0.y * ly + s0.z * rr,
              s1.x * lx + s1.y * ly + s1.z * rr,
              s2.x * lx + s2.y * ly + s2.z * rr)
    prior = graph.prior
    if prior is not None:
        p = graph.poses.get(prior.p)
        block(p.pos.x - prior.pos.x, p.pos.y - prior.pos.y,
              p.rot_angle - prior.th)
    return ls, huber


def write_eps(before, after, out):
    """Minimal EPS scatter (before = gray, after = black)."""
    xs = [p.x for p in before + after]
    ys = [p.y for p in before + after]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    size = 500.0
    scale = size / max(xmax - xmin, ymax - ymin)
    with open(out, "w") as f:
        f.write("%!PS-Adobe-3.0 EPSF-3.0\n")
        f.write("%%%%BoundingBox: 0 0 %d %d\n" % (int(size) + 20, int(size) + 20))
        for pts, gray in ((before, 0.75), (after, 0.0)):
            f.write("%g setgray\n" % gray)
            for p in pts:
                f.write("%.1f %.1f 1.2 0 360 arc fill\n"
                        % (10.0 + (p.x - xmin) * scale,
                           10.0 + (p.y - ymin) * scale))
        f.write("showpage\n")


def main():
    weighted = "--weighted" in sys.argv[1:]
    dump = None
    path = DEFAULT_DATASET
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dump" and i + 1 < len(args):
            i += 1
            dump = args[i]
        elif not args[i].startswith("-"):
            path = args[i]
        i += 1

    graph = g.Graph()
    load_g2o(path, weighted, graph)
    if weighted:
        print("using information-matrix (sqrt-info) weighting")
    print("%s: %d poses, %d edges" % (path, len(graph.poses), len(graph.edges)))

    ls0, huber0 = metrics(graph)
    print("initial cost: LS=%.6f huber=%.6f" % (ls0, huber0))
    before = [p.pos for p in graph.poses]

    print("parameters: %d" % (len(graph.poses) * g.Pose2.param_count))

    cfg = g.LmConfig.well_conditioned()
    cfg.verbose = os.environ.get("VERBOSE") is not None
    start = time.monotonic()
    r = graph.solve_sparse(cfg)
    elapsed = time.monotonic() - start

    ls1, huber1 = metrics(graph)
    print("%d iterations, cost %.6f -> %.6f"
          % (r.iterations, r.start_cost, r.end_cost))
    print("final cost:   LS=%.6f huber=%.6f" % (ls1, huber1))
    print("solve time: %.3fs" % elapsed)

    after = [p.pos for p in graph.poses]
    out = "m3500_weighted.eps" if weighted else "m3500.eps"
    write_eps(before, after, out)
    print("wrote %s" % out)

    if dump:
        with open(dump, "w") as f:
            for p in graph.poses:
                f.write("%.17g %.17g %.17g\n" % (p.pos.x, p.pos.y, p.rot_angle))
        print("dumped poses to %s" % dump)

    for label, idx in (("x0", 0), ("x1", 1), ("x3499", 3499)):
        if idx < len(graph.poses):
            p = graph.poses[idx]
            print("%s: theta=%.6f x=%.6f y=%.6f"
                  % (label, p.rot_angle, p.pos.x, p.pos.y))


if __name__ == "__main__":
    main()
