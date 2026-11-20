# The generated Python interface's GPU report, the capi built with its
# gpu-cpu feature (python.rs): the fixture solved on the CPU device with
# the device assembly on and off, and without a GPU. Prints `name value`
# lines for python.rs to check.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "model", "python"))

from cxx_fit import fit  # noqa: E402


def pi(n, v):
    print("%s %d" % (n, v))


def fill(f):
    for i in range(6):
        o = f.obs.push()
        o.x = float(i)
        o.y = 2.0 * i + 1.0 + (0.05 if i % 2 == 0 else -0.05)
    for t, w in zip([1.5, -0.3, 0.7], [1.0, 2.0, 0.5]):
        n = f.items.push()
        n.t = t
        n.w = w
    vn = f.vns.push()
    vn.v = [0.4, -0.1, 0.9, 0.0]
    vn.t = [0.1, 0.2, 0.5, -0.3]
    vn.h = [[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]]
    vn.wp = 0.7
    vn.w = 1.3


def solve(gpu, assembly):
    f = fit.Fit()
    fill(f)
    so = fit.SparseOptions()
    so.block_supernodal = fit.BlockSupernodalMode.ALWAYS
    so.gpu = gpu
    so.gpu_assembly = assembly
    cfg = fit.LmConfig()
    cfg.max_iters = 50
    cfg.gather_timing = True
    return f.solve_sparse(cfg, so)


def show(tag, g):
    pi(tag + "_backend", int(g.backend))
    pi(tag + "_index", g.index)
    pi(tag + "_named", 1 if g.name and g.target else 0)
    pi(tag + "_integrated", 1 if g.integrated else 0)
    pi(tag + "_wide", int(g.wide))
    pi(tag + "_solve", int(g.solve))
    pi(tag + "_f32_system", 1 if g.f32_system else 0)
    pi(tag + "_work", int(g.work))
    pi(tag + "_host_assembly", int(g.host_assembly))
    pi(tag + "_why", len(g.no_device_form))
    pi(tag + "_reserved", 1 if g.reserved_bytes > 0 else 0)
    pi(tag + "_buffers", 1 if g.buffers_created > 0 else 0)


def timing(tag, r):
    """The timed solve's setup parts and the device's own time."""
    t = r.timing
    s, g = t.setup, t.gpu
    pi(tag + "_setup", 0 if s is None else 1)
    if s is not None:
        parts = s.blocks + s.structure + s.gpu_open + s.ordering + s.weighing + s.factor + s.gpu_plan + s.gpu_layout
        pi(tag + "_setup_within", 1 if 0.0 < parts <= t.analysis else 0)
        pi(tag + "_setup_gpu", 1 if s.gpu_plan > 0.0 else 0)
    pi(tag + "_gpu_time", 0 if g is None else 1)
    if g is not None:
        pi(tag + "_attempts_timed", 1 if g.attempt_count == r.iterations and g.attempts > 0.0 else 0)
        pi(tag + "_assembly_timed", g.assembly_count)
        # the conjugate gradients' iterations, within the attempts
        pi(tag + "_cg_timed", 1 if 0.0 < g.cg <= g.attempts and g.cg_rounds >= g.attempt_count else 0)


fit.set_log_level(fit.LogLevel.WARN)
none = solve(fit.GpuBackend.NONE, True)
pi("none", 1 if none.gpu is None else 0)
timing("none", none)
r = solve(fit.GpuBackend.CPU, True)
show("on", r.gpu)
timing("on", r)
pi("on_report", 1 if "  gpu       cpu:0 " in r.report() else 0)
show("off", solve(fit.GpuBackend.CPU, False).gpu)

# Conjugate gradients on the reduced system, on the device.
f = fit.Fit()
fill(f)
so = fit.SparseOptions()
so.schur = fit.SchurPolicy.FORCE
so.schur_solve = fit.SchurSolve.ITERATIVE
so.gpu = fit.GpuBackend.CPU
cfg = fit.LmConfig()
cfg.max_iters = 50
cfg.gather_timing = True
r = f.solve_sparse(cfg, so)
timing("cg", r)
pi("cg_solve", int(r.gpu.solve))
pi("cg_is_cg", 1 if r.gpu.solve == fit.GpuSolve.CONJUGATE_GRADIENTS else 0)
pi("cg_work", int(r.gpu.work))
pi("cg_iters", 1 if (r.plan.cg_iterations or 0) > 0 else 0)
