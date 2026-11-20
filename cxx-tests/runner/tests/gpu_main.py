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
    return f.solve_sparse(cfg, so)


def show(tag, g):
    pi(tag + "_backend", int(g.backend))
    pi(tag + "_index", g.index)
    pi(tag + "_named", 1 if g.name and g.target else 0)
    pi(tag + "_integrated", 1 if g.integrated else 0)
    pi(tag + "_wide", int(g.wide))
    pi(tag + "_f32_factor", 1 if g.f32_factor else 0)
    pi(tag + "_f32_system", 1 if g.f32_system else 0)
    pi(tag + "_work", int(g.work))
    pi(tag + "_cpu_assembly", int(g.cpu_assembly))
    pi(tag + "_why", len(g.no_device_form))
    pi(tag + "_reserved", 1 if g.reserved_bytes > 0 else 0)
    pi(tag + "_buffers", 1 if g.buffers_created > 0 else 0)


fit.set_log_level(fit.LogLevel.WARN)
pi("none", 1 if solve(fit.GpuBackend.NONE, True).gpu is None else 0)
r = solve(fit.GpuBackend.CPU, True)
show("on", r.gpu)
pi("on_report", 1 if "  gpu       cpu:0 " in r.report() else 0)
show("off", solve(fit.GpuBackend.CPU, False).gpu)
