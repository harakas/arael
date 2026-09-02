# The vendored column transport (arael/columns.py) on its own: every
# input shape set_<field> / get_<field> / push_many accept, and every
# refusal, without a cdylib. Exits non-zero on the first failure.
import array
import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "model", "python"))

from cxx_fit.arael import columns as c  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None


class Ref:
    def __init__(self, raw):
        self.raw = raw


def doubles(ptr, stride, n, k=1):
    out = []
    for i in range(n):
        row = (ctypes.c_double * k).from_address(ptr + i * stride)
        out.append(tuple(row) if k > 1 else row[0])
    return out


def ints(ct, ptr, stride, n):
    return [ct.from_address(ptr + i * stride).value for i in range(n)]


# flat: a flat sequence, rows, and the refusal
assert c.flat((1, 2, 3), 3) == (1, 2, 3)
assert c.flat([[1, 2], [3, 4]], 4) == (1, 2, 3, 4)
try:
    c.flat((1, 2), 3)
    raise SystemExit("flat accepted the wrong length")
except TypeError:
    pass

# count: explicit n, first sequence, refusal
assert c.count(4, ()) == 4
assert c.count(None, (("a", 1.0), ("b", [1, 2, 3]))) == 3
assert c.count(None, (("a", Ref(3)), ("b", [1, 2]))) == 2   # a ref is a scalar
try:
    c.count(None, (("a", 1.0), ("b", None)))
    raise SystemExit("count accepted all scalars")
except TypeError:
    pass

# scalar broadcast: float, int, bool, ref
ptr, stride, keep = c.column_in(2.5, "d", 1, 4, "x")
assert stride == 0 and doubles(ptr, 8, 1) == [2.5]
ptr, stride, keep = c.column_in(True, "B", 1, 4, "b")
assert stride == 0 and ints(ctypes.c_uint8, ptr, 1, 1) == [1]
ptr, stride, keep = c.column_in(Ref(7), "I", 1, 4, "r")
assert stride == 0 and ints(ctypes.c_uint32, ptr, 4, 1) == [7]
ptr, stride, keep = c.column_in(-3, "i", 1, 2, "n")
assert ints(ctypes.c_int32, ptr, 4, 1) == [-3]
try:
    c.column_in(2.5, "d", 3, 4, "pos")
    raise SystemExit("a scalar broadcast into a 3-component field")
except TypeError:
    pass

# sequences without numpy: scalars, refs, rows, one math value broadcast
ptr, stride, keep = c.column_in([1.0, 2.0, 3.0], "d", 1, 3, "x")
assert stride == 8 and doubles(ptr, stride, 3) == [1.0, 2.0, 3.0]
ptr, stride, keep = c.column_in([Ref(1), 2, Ref(3)], "I", 1, 3, "r")
assert ints(ctypes.c_uint32, ptr, stride, 3) == [1, 2, 3]
ptr, stride, keep = c.column_in([(1, 2, 3), (4, 5, 6)], "d", 3, 2, "pos")
assert stride == 24 and doubles(ptr, stride, 2, 3) == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
ptr, stride, keep = c.column_in((1, 2, 3), "d", 3, 5, "pos")
assert stride == 0 and doubles(ptr, 24, 1, 3) == [(1.0, 2.0, 3.0)]
ptr, stride, keep = c.column_in([[(1, 2), (3, 4)]], "d", 4, 1, "m")   # rows flatten
assert doubles(ptr, 32, 1, 4) == [(1.0, 2.0, 3.0, 4.0)]
try:
    c.column_in([1.0, 2.0], "d", 1, 3, "x")
    raise SystemExit("wrong length accepted")
except ValueError:
    pass
try:
    c.column_in([(1, 2), (3, 4)], "d", 3, 2, "pos")
    raise SystemExit("wrong component count accepted")
except TypeError:
    pass

# buffers without numpy: array.array in place, a read-only float64
# view copied, a byte string refused (it is a uint8 buffer)
a = array.array("d", [1.0, 2.0, 3.0])
ptr, stride, keep = c.column_in(a, "d", 1, 3, "x")
assert ptr == a.buffer_info()[0] and stride == 8
ro = memoryview(array.array("d", [4.0, 5.0])).toreadonly()
ptr, stride, keep = c.column_in(ro, "d", 1, 2, "x")
assert doubles(ptr, stride, 2) == [4.0, 5.0]
for bad in (bytes(16), array.array("i", [1, 2, 3])):
    try:
        c.column_in(bad, "d", 1, 3, "x")
        raise SystemExit("%r accepted for a float64 field" % (bad,))
    except TypeError:
        pass

# out buffers: with numpy the array shares the buffer; without, the buffer itself
buf, ptr, stride = c.column_out("d", 3, 2)
assert stride == 24 and len(buf) == 6
buf[4] = 9.0
out = c.column_finish(buf, "d", 3, 2)
if np is not None:
    assert out.shape == (2, 3) and out[1, 1] == 9.0 and out.dtype == np.float64
saved = c._np
c._np = None
flat = c.column_finish(buf, "d", 3, 2)
assert flat is buf
c._np = saved

if np is not None:
    # right dtype in place, 2-D math, strided view, 1-D math broadcast
    x = np.arange(4.0)
    ptr, stride, keep = c.column_in(x, "d", 1, 4, "x")
    assert ptr == x.ctypes.data and stride == 8
    P = np.arange(6.0).reshape(2, 3)
    ptr, stride, keep = c.column_in(P, "d", 3, 2, "pos")
    assert ptr == P.ctypes.data and stride == 24
    X = np.arange(8.0).reshape(4, 2)
    ptr, stride, keep = c.column_in(X[:, 1], "d", 1, 4, "x")
    assert stride == 16 and doubles(ptr, stride, 4) == [1.0, 3.0, 5.0, 7.0]
    ptr, stride, keep = c.column_in(np.array([1.0, 2.0, 3.0]), "d", 3, 5, "pos")
    assert stride == 0
    ptr, stride, keep = c.column_in(np.array([True, False]), "B", 1, 2, "b")
    assert ints(ctypes.c_uint8, ptr, stride, 2) == [1, 0]
    ptr, stride, keep = c.column_in(np.array([1, 2], np.uint32), "I", 1, 2, "r")
    assert ints(ctypes.c_uint32, ptr, stride, 2) == [1, 2]
    ptr, stride, keep = c.column_in(np.array([-1, 2], np.int32), "i", 1, 2, "n")
    assert ints(ctypes.c_int32, ptr, stride, 2) == [-1, 2]
    ptr, stride, keep = c.column_in(np.array([0.5, 1.5], np.float32), "f", 1, 2, "s")
    assert ints(ctypes.c_float, ptr, stride, 2) == [0.5, 1.5]
    # refusals: dtype, length, shape, a 2-D array for a scalar field
    for bad, exc in ((np.arange(4), TypeError), (np.arange(3.0), ValueError),
                     (np.zeros((4, 1)), ValueError), (np.zeros((2, 3), np.float32), TypeError)):
        try:
            c.column_in(bad, "d", 1, 4, "x")
            raise SystemExit("accepted %r" % (bad,))
        except exc:
            pass
    try:
        c.column_in(np.zeros((2, 4)), "d", 3, 2, "pos")
        raise SystemExit("wrong component count accepted")
    except ValueError:
        pass
    # a Fortran-ordered (n, 3) block is not one row per element: refused
    F = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    try:
        c.column_in(F, "d", 3, 2, "pos")
        raise SystemExit("column-major block accepted")
    except ValueError:
        pass
    # a read-only numpy array still passes by address
    ro = np.arange(3.0)
    ro.setflags(write=False)
    ptr, stride, keep = c.column_in(ro, "d", 1, 3, "x")
    assert ptr == ro.ctypes.data

print("columns ok")
