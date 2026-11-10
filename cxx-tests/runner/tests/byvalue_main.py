# The vendored Python math types by value through C: byvalue.c's
# functions return their argument doubled. Prints `<type> ok` or
# `<type> got <values> want <values>` per type (python.rs checks).
import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "model", "python"))

from cxx_fit.arael import math as m  # noqa: E402

lib = ctypes.CDLL(sys.argv[1])


def flat(v):
    if hasattr(v, "rows"):
        return [x for r in v.rows for x in r]
    return list(v)


cases = [
    ("vectnd4", m.vectnd(4), [0.4, -0.1, 0.9, 0.25]),
    ("vectnf4", m.vectnf(4), [0.5, -1.5, 2.5, 4.0]),
    ("vect2d", m.vect2d, [0.5, -0.75]),
    ("matrix2d", m.matrix2d, [[1.5, -2.0], [0.25, 4.0]]),
    ("matrixnd22", m.matrixnd(2, 2), [[1.0, 2.0], [-3.0, 0.5]]),
    ("matrixnd24", m.matrixnd(2, 4), [[1.0, 0.5, 0.0, -0.2], [0.0, 1.0, 0.3, 0.4]]),
    ("matrix3d", m.matrix3d, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5]]),
]
for name, cls, vals in cases:
    f = getattr(lib, "twice_" + name)
    f.argtypes = [cls]
    f.restype = cls
    got = flat(f(cls(vals)))
    want = [2.0 * x for x in flat(cls(vals))]
    print("%s ok" % name if got == want else "%s got %s want %s" % (name, got, want))
