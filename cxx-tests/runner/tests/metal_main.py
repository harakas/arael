# The generated Python interface on Metal, the capi built with its
# gpu-metal feature (python.rs): the fixture's f64 model needs
# mixed_precision there, GpuWide.DEFAULT is the device's df32 and an
# explicit F64 is refused, and the refined solve on the device lands where
# the CPU's does. Prints `name value` lines for python.rs to check.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "model", "python"))

from cxx_fit import fit  # noqa: E402
from cxx_fit.arael import AraelError  # noqa: E402


def p(n, v):
    print("%s %.17e" % (n, v))


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


def solve(gpu, mixed, wide=None):
    """The fixture solved on the block supernodal route: the failure's kind,
    or -1 and the result."""
    f = fit.Fit()
    fill(f)
    so = fit.SparseOptions()
    so.block_supernodal = fit.BlockSupernodalMode.ALWAYS
    so.gpu = gpu
    so.mixed_precision = mixed
    if wide is not None:
        so.gpu_wide = wide
    cfg = fit.LmConfig()
    cfg.max_iters = 50
    try:
        return -1, f.solve_sparse(cfg, so)
    except AraelError as e:
        return (e.failure.kind if e.failure else -99), None


fit.set_log_level(fit.LogLevel.WARN)
pi("default_wide", int(fit.SparseOptions().gpu_wide))
kind, _ = solve(fit.GpuBackend.METAL, False)
pi("f64_unmixed_kind", kind)
kind, _ = solve(fit.GpuBackend.METAL, True, fit.GpuWide.F64)
pi("f64_wide_kind", kind)
_, cpu = solve(fit.GpuBackend.NONE, False)
kind, metal = solve(fit.GpuBackend.METAL, True)
pi("metal_kind", kind)
p("cpu_end", cpu.end_cost)
p("metal_end", metal.end_cost)
pi("metal_gpu_factor", 1 if metal.plan.gpu_factor else 0)
pi("metal_f32_factor", 1 if metal.plan.f32_factor else 0)
