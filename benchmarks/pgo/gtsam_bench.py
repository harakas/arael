# GTSAM runner for the 2D pose-graph benchmark.
#
# Reads a g2o file with gtsam.readG2o (which honors the information
# matrices natively), adds the same unit-weight gauge prior on pose 0 as
# the other runners, optimizes with LM or GN, and emits JSON on stdout:
#   { "solve_ms", "first_iter_ms", "iterations", "poses_file" }
# The final poses are dumped as "x y theta" lines for validation by the
# Rust harness. Timing wraps only optimize() -- a C++ call; Python
# overhead outside it is excluded.
#
# usage: gtsam_bench.py <file.g2o> <lm|gn|isam2> <poses_out> <info|unit>
# "unit" replaces the file's information matrices with identity (the
# graph is then built manually -- readG2o always applies the file's
# info), matching the unweighted benchmark configuration.

import json
import sys
import time

import gtsam


def make_optimizer(kind, graph, initial, max_iterations):
    if kind == "lm":
        params = gtsam.LevenbergMarquardtParams()
    else:
        params = gtsam.GaussNewtonParams()
    # Same criterion class as the other systems: stop when a step improves
    # the (half-)cost by less than 1e-5 absolute or 1e-5 relative. These
    # are also GTSAM's own defaults.
    params.setRelativeErrorTol(1e-5)
    params.setAbsoluteErrorTol(1e-5)
    params.setMaxIterations(max_iterations)
    if kind == "lm":
        return gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    return gtsam.GaussNewtonOptimizer(graph, initial, params)


def parse_g2o(g2o_file, unit):
    poses = {}
    edges = []
    with open(g2o_file) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "VERTEX_SE2":
                poses[int(t[1])] = gtsam.Pose2(float(t[2]), float(t[3]), float(t[4]))
            elif t[0] == "EDGE_SE2":
                a, b = int(t[1]), int(t[2])
                delta = gtsam.Pose2(float(t[3]), float(t[4]), float(t[5]))
                if unit:
                    noise = gtsam.noiseModel.Unit.Create(3)
                else:
                    i11, i33 = float(t[6]), float(t[11])
                    noise = gtsam.noiseModel.Diagonal.Precisions([i11, i11, i33])
                edges.append((a, b, delta, noise))
    return poses, edges


def run_isam2(g2o_file, poses_out, unit):
    """Incremental solve, the way GTSAM's own city10000 example drives it:
    one pose per update, initialized by composing the previous estimate
    with the odometry measurement; loop closures join at their later
    endpoint. Timed portion: the update() calls and the final estimate."""
    poses, edges = parse_g2o(g2o_file, unit)
    edges_by_step = {}
    for (a, b, delta, noise) in edges:
        edges_by_step.setdefault(max(a, b), []).append((a, b, delta, noise))
    n = len(poses)

    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    total = 0.0
    first_ms = None
    for k in range(n):
        new_factors = gtsam.NonlinearFactorGraph()
        new_vals = gtsam.Values()
        odo = None
        for (a, b, delta, noise) in edges_by_step.get(k, []):
            new_factors.add(gtsam.BetweenFactorPose2(a, b, delta, noise))
            if {a, b} == {k - 1, k}:
                odo = (a, b, delta)
        if k == 0:
            prior = gtsam.noiseModel.Diagonal.Sigmas([1.0, 1.0, 1.0])
            new_factors.add(gtsam.PriorFactorPose2(0, poses[0], prior))
            new_vals.insert(0, poses[0])
        else:
            if odo is not None:
                a, b, delta = odo
                prev = isam.calculateEstimatePose2(k - 1)
                guess = prev.compose(delta) if a == k - 1 else prev.compose(delta.inverse())
            else:
                guess = poses[k]
            new_vals.insert(k, guess)
        t0 = time.perf_counter()
        isam.update(new_factors, new_vals)
        dt = time.perf_counter() - t0
        total += dt
        if first_ms is None:
            first_ms = dt * 1e3
    t0 = time.perf_counter()
    result = isam.calculateEstimate()
    total += time.perf_counter() - t0

    with open(poses_out, "w") as f:
        for i in range(n):
            p = result.atPose2(i)
            f.write(f"{p.x()} {p.y()} {p.theta()}\n")
    return total * 1e3, first_ms, n


def main():
    g2o_file, kind, poses_out = sys.argv[1], sys.argv[2], sys.argv[3]
    unit = len(sys.argv) > 4 and sys.argv[4] == "unit"
    if kind == "isam2":
        solve_ms, first_iter_ms, iterations = run_isam2(g2o_file, poses_out, unit)
        cpus = "?"
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    cpus = line.split()[1]
        print(json.dumps({
            "solve_ms": solve_ms,
            "first_iter_ms": first_iter_ms,
            "iterations": iterations,
            "cpus_allowed": cpus,
        }))
        return
    if unit:
        poses, edges = parse_g2o(g2o_file, True)
        graph = gtsam.NonlinearFactorGraph()
        for (a, b, delta, noise) in edges:
            graph.add(gtsam.BetweenFactorPose2(a, b, delta, noise))
        initial = gtsam.Values()
        for k in sorted(poses):
            initial.insert(k, poses[k])
    else:
        graph, initial = gtsam.readG2o(g2o_file, False)
    prior = gtsam.noiseModel.Diagonal.Sigmas([1.0, 1.0, 1.0])
    graph.add(gtsam.PriorFactorPose2(0, initial.atPose2(0), prior))

    opt = make_optimizer(kind, graph, initial, 1)
    t0 = time.perf_counter()
    opt.optimize()
    first_iter_ms = (time.perf_counter() - t0) * 1e3

    opt = make_optimizer(kind, graph, initial, 100)
    t0 = time.perf_counter()
    result = opt.optimize()
    solve_ms = (time.perf_counter() - t0) * 1e3
    iterations = opt.iterations()

    with open(poses_out, "w") as f:
        for i in range(initial.size()):
            p = result.atPose2(i)
            f.write(f"{p.x()} {p.y()} {p.theta()}\n")

    cpus = "?"
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("Cpus_allowed_list"):
                cpus = line.split()[1]
    print(json.dumps({
        "solve_ms": solve_ms,
        "first_iter_ms": first_iter_ms,
        "iterations": iterations,
        "cpus_allowed": cpus,
    }))


if __name__ == "__main__":
    main()
